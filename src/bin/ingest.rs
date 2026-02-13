use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use chatbot::config::AppConfig;
use chatbot::db::Database;
use chatbot::ingest::Ingestor;
use chatbot::models::IngestRequest;
use chatbot::ollama::OllamaClient;
use chatbot::qdrant_store::QdrantStore;
use chatbot::tantivy_store::TantivyStore;

#[derive(Parser, Debug)]
#[command(name = "ingest")]
#[command(about = "Ingest DOCX + PDF novel sources into local indexes")]
struct Cli {
    #[arg(long)]
    docx: String,
    #[arg(long)]
    pdf: String,
    #[arg(long, default_value_t = false)]
    rebuild: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();

    let config = AppConfig::from_env();
    tokio::fs::create_dir_all(&config.data_dir).await?;

    let db = Database::new(&config).await?;
    let ollama = OllamaClient::new(config.ollama_base_url.clone());
    let qdrant = QdrantStore::new(
        config.qdrant_base_url.clone(),
        config.qdrant_collection.clone(),
    );
    let tantivy = TantivyStore::new(config.tantivy_dir());
    let ingestor = Ingestor::new(config, db, ollama, qdrant, tantivy);

    let req = IngestRequest {
        docx_path: cli.docx,
        pdf_path: cli.pdf,
        rebuild: cli.rebuild,
    };

    let job_id = format!("cli-{}", uuid::Uuid::new_v4());
    let result = ingestor
        .ingest(&job_id, req, |status| {
            println!(
                "[{}] {} chunks={} images={} {}",
                status.status,
                status.stage,
                status.chunk_count,
                status.image_count,
                status.message.unwrap_or_default()
            );
        })
        .await?;

    println!(
        "Ingest complete. skipped={} chunks={} images={}",
        result.skipped, result.chunk_count, result.image_count
    );

    Ok(())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}
