use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use tokio::sync::Semaphore;

use chatbot::chat::ChatService;
use chatbot::config::AppConfig;
use chatbot::db::Database;
use chatbot::models::ChatRequest;
use chatbot::ollama::OllamaClient;
use chatbot::qdrant_store::QdrantStore;
use chatbot::retrieval::Retriever;
use chatbot::tantivy_store::TantivyStore;

#[derive(Parser, Debug)]
#[command(name = "eval")]
#[command(about = "Run a local retrieval/answer evaluation set")]
struct Cli {
    #[arg(long, default_value = "eval/prompts.jsonl")]
    file: String,
    #[arg(long, default_value_t = true)]
    strict: bool,
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

#[derive(Debug, Deserialize)]
struct EvalPrompt {
    id: String,
    question: String,
    #[serde(default)]
    strict: Option<bool>,
    #[serde(default)]
    expect_contains: Vec<String>,
    #[serde(default)]
    expect_not_found: Option<bool>,
    #[serde(default)]
    expect_late_part: Option<bool>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = AppConfig::from_env();

    let db = Database::new(&config).await?;
    let ollama = OllamaClient::new(config.ollama_base_url.clone());
    let qdrant = QdrantStore::new(
        config.qdrant_base_url.clone(),
        config.qdrant_collection.clone(),
    );
    let tantivy = TantivyStore::new(config.tantivy_dir());
    let retriever = Retriever::new(
        db.clone(),
        qdrant,
        tantivy,
        ollama.clone(),
        config.models.embedding_model.clone(),
    );

    let chat = ChatService::new(config, db, ollama, retriever, Arc::new(Semaphore::new(1)));

    let prompts = load_prompts(&cli.file)?;
    if prompts.is_empty() {
        anyhow::bail!("no prompts found in {}", cli.file);
    }

    let mut total = 0usize;
    let mut pass_contains = 0usize;
    let mut not_found_count = 0usize;
    let mut with_citations = 0usize;
    let mut late_part_expectations = 0usize;
    let mut late_part_hits = 0usize;

    for prompt in prompts {
        total += 1;
        let request = ChatRequest {
            session_id: format!("eval-{}", uuid::Uuid::new_v4()),
            question: prompt.question.clone(),
            strict: prompt.strict.unwrap_or(cli.strict),
            verbose: true,
        };

        let answer = chat
            .answer(request)
            .await
            .with_context(|| format!("failed eval prompt {}", prompt.id))?;

        let answer_lower = answer.answer_markdown.to_ascii_lowercase();
        let is_not_found = answer_lower.contains("not found in indexed novel sources");
        if is_not_found {
            not_found_count += 1;
        }
        if !answer.citations.is_empty() {
            with_citations += 1;
        }

        let mut contains_pass = true;
        for needle in &prompt.expect_contains {
            if !answer_lower.contains(&needle.to_ascii_lowercase()) {
                contains_pass = false;
                break;
            }
        }

        if prompt.expect_not_found.unwrap_or(false) != is_not_found {
            contains_pass = false;
        }

        if contains_pass {
            pass_contains += 1;
        }

        if prompt.expect_late_part.unwrap_or(false) {
            late_part_expectations += 1;
            if answer
                .citations
                .iter()
                .filter_map(|c| c.chapter.as_ref())
                .any(|chapter| is_late_part_label(chapter))
            {
                late_part_hits += 1;
            }
        }

        if cli.verbose {
            println!("--- {} ---", prompt.id);
            println!("Q: {}", prompt.question);
            println!("A: {}", answer.answer_markdown.replace('\n', " "));
            println!("Citations: {}", answer.citations.len());
            println!();
        }
    }

    let contains_acc = ratio(pass_contains, total);
    let citation_rate = ratio(with_citations, total);
    let not_found_rate = ratio(not_found_count, total);
    let late_part_hit_rate = if late_part_expectations > 0 {
        ratio(late_part_hits, late_part_expectations)
    } else {
        0.0
    };

    println!("Eval prompts: {}", total);
    println!("Contains/expected accuracy: {:.1}%", contains_acc * 100.0);
    println!("Citation rate: {:.1}%", citation_rate * 100.0);
    println!("Not-found rate: {:.1}%", not_found_rate * 100.0);
    if late_part_expectations > 0 {
        println!(
            "Late-part coverage hit-rate: {:.1}% ({}/{})",
            late_part_hit_rate * 100.0,
            late_part_hits,
            late_part_expectations
        );
    }

    Ok(())
}

fn load_prompts(path: &str) -> Result<Vec<EvalPrompt>> {
    let file = File::open(path).with_context(|| format!("failed opening {}", path))?;
    let reader = BufReader::new(file);
    let mut prompts = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parsed: EvalPrompt = serde_json::from_str(trimmed)
            .with_context(|| format!("invalid JSON at {} line {}", path, idx + 1))?;
        prompts.push(parsed);
    }

    Ok(prompts)
}

fn ratio(n: usize, d: usize) -> f32 {
    if d == 0 {
        return 0.0;
    }
    n as f32 / d as f32
}

fn is_late_part_label(label: &str) -> bool {
    let lower = label.to_ascii_lowercase();
    lower.contains("part 6")
        || lower.contains("part 7")
        || lower.contains("part 7a")
        || lower.contains("part 7b")
}
