use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use regex::Regex;
use tokio::sync::Semaphore;

use crate::config::AppConfig;
use crate::db::Database;
use crate::models::{
    AnswerMode, ChatAnswer, ChatRequest, Citation, QueryClass, RetrievalResult, SourceType,
};
use crate::ollama::OllamaClient;
use crate::retrieval::Retriever;

const NOT_FOUND_MESSAGE: &str = "Not found in indexed novel sources.";

#[derive(Clone)]
struct ContextSource {
    source_id: String,
    result: RetrievalResult,
    part: Option<String>,
}

#[derive(Clone)]
pub struct ChatService {
    config: AppConfig,
    db: Database,
    ollama: OllamaClient,
    retriever: Retriever,
    generation_limit: Arc<Semaphore>,
}

impl ChatService {
    pub fn new(
        config: AppConfig,
        db: Database,
        ollama: OllamaClient,
        retriever: Retriever,
        generation_limit: Arc<Semaphore>,
    ) -> Self {
        Self {
            config,
            db,
            ollama,
            retriever,
            generation_limit,
        }
    }

    pub async fn answer(&self, request: ChatRequest) -> Result<ChatAnswer> {
        let started = Instant::now();

        self.db
            .ensure_session(&request.session_id)
            .await
            .map_err(anyhow::Error::from)?;
        self.db
            .save_message(&request.session_id, "user", &request.question)
            .await
            .map_err(anyhow::Error::from)?;

        let recent_history = self
            .db
            .latest_messages(&request.session_id, 8)
            .await
            .map_err(anyhow::Error::from)?;

        let resolved_question = resolve_question_with_history(&request.question, &recent_history);
        let query_class = self.classify_query(&resolved_question).await;
        let expanded_question = expand_question_for_retrieval(&resolved_question);
        let retrieved = self.retriever.retrieve(&expanded_question, 24).await?;
        let retrieved = self
            .boost_with_anchor_matches(&resolved_question, retrieved)
            .await?;

        let selected = trim_to_context_budget(retrieved, self.config.tokens.max_context_tokens, 10);

        if selected.is_empty() {
            let answer = ChatAnswer {
                answer_markdown: NOT_FOUND_MESSAGE.to_string(),
                citations: vec![],
                mode: AnswerMode::TextOnly,
                confidence: 0.0,
                latency_ms: started.elapsed().as_millis(),
            };

            self.db
                .save_message(&request.session_id, "assistant", &answer.answer_markdown)
                .await
                .map_err(anyhow::Error::from)?;
            return Ok(answer);
        }

        let mut mode = AnswerMode::TextOnly;
        let context_sources = self.build_context_sources(selected).await?;
        let (mut context, _) = build_context(&context_sources);
        let source_ids: Vec<String> = context_sources
            .iter()
            .map(|source| source.source_id.clone())
            .collect();
        let mut confidence = estimate_confidence(
            &context_sources
                .iter()
                .map(|source| source.result.clone())
                .collect::<Vec<_>>(),
        );

        let should_escalate = matches!(query_class, QueryClass::ImageRelevant)
            && confidence < 0.35
            && context_sources.iter().any(|source| {
                matches!(
                    source.result.chunk.kind,
                    SourceType::ImageCaption | SourceType::ImageOcr
                )
            });

        if should_escalate {
            let vision_notes = self
                .vision_notes_for_question(
                    &request.question,
                    &context_sources
                        .iter()
                        .map(|source| source.result.clone())
                        .collect::<Vec<_>>(),
                )
                .await
                .unwrap_or_default();
            if !vision_notes.is_empty() {
                mode = AnswerMode::VisionEscalated;
                context.push_str("\n\n# Additional Vision Notes\n");
                context.push_str(&vision_notes);
                confidence = (confidence + 0.1).min(1.0);
            }
        }

        let history = recent_history;

        let prompt = build_answer_prompt(
            &request.question,
            &context,
            &history,
            request.strict,
            &source_ids,
            self.config.tokens.max_output_tokens,
        );

        let _permit = self.generation_limit.acquire().await?;
        let mut answer_text = self
            .ollama
            .generate_text(
                &self.config.models.answer_model,
                &prompt,
                self.config.tokens.max_output_tokens,
                0.1,
            )
            .await
            .unwrap_or_else(|_| NOT_FOUND_MESSAGE.to_string());
        answer_text = sanitize_model_output(answer_text);

        if request.strict && !has_source_citation(&answer_text) {
            let repaired = self
                .repair_answer_with_citations(
                    &request.question,
                    &context,
                    &answer_text,
                    &source_ids,
                )
                .await
                .unwrap_or_default();

            if has_source_citation(&repaired) {
                answer_text = repaired;
            }
        }
        answer_text = sanitize_model_output(answer_text);

        if let Some(direct_answer) =
            direct_death_answer_from_sources(&request.question, &context_sources)
        {
            answer_text = direct_answer;
        }

        if let Some(part_answer) =
            direct_part_answer_from_sources(&request.question, &context_sources)
        {
            answer_text = part_answer;
        }

        answer_text = enforce_strict_mode(answer_text, request.strict);

        let citations = build_citations(&context_sources, &answer_text, request.strict);
        let response = ChatAnswer {
            answer_markdown: answer_text.clone(),
            citations,
            mode,
            confidence,
            latency_ms: started.elapsed().as_millis(),
        };

        self.db
            .save_message(&request.session_id, "assistant", &answer_text)
            .await
            .map_err(anyhow::Error::from)?;

        Ok(response)
    }

    async fn classify_query(&self, question: &str) -> QueryClass {
        let lower = question.to_ascii_lowercase();
        if has_image_keywords(&lower) {
            return QueryClass::ImageRelevant;
        }

        let prompt = format!(
            "Classify this user question for a novel assistant. Return only one token: text_only or image_relevant.\nQuestion: {question}"
        );

        match self
            .ollama
            .generate_text(&self.config.models.answer_model, &prompt, 8, 0.0)
            .await
        {
            Ok(reply) if reply.trim().to_ascii_lowercase().contains("image") => {
                QueryClass::ImageRelevant
            }
            _ => QueryClass::TextOnly,
        }
    }

    async fn vision_notes_for_question(
        &self,
        question: &str,
        retrieved: &[RetrievalResult],
    ) -> Result<String> {
        let mut notes = Vec::new();

        for item in retrieved.iter().take(2) {
            if !matches!(
                item.chunk.kind,
                SourceType::ImageCaption | SourceType::ImageOcr
            ) {
                continue;
            }

            let Some(path) = &item.chunk.image_path else {
                continue;
            };

            let bytes = match tokio::fs::read(path).await {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };

            let vision_prompt = format!(
                "Answer this about the image in one short paragraph: {question}. Focus only on visible evidence."
            );

            let note = self
                .ollama
                .generate_vision_caption(&self.config.models.vision_model, &vision_prompt, &bytes)
                .await
                .unwrap_or_default();

            if note.trim().is_empty() {
                continue;
            }

            notes.push(format!(
                "- [{}] {}",
                item.chunk.id,
                note.replace('\n', " ").trim()
            ));
        }

        Ok(notes.join("\n"))
    }

    async fn repair_answer_with_citations(
        &self,
        question: &str,
        context: &str,
        draft_answer: &str,
        source_ids: &[String],
    ) -> Result<String> {
        let prompt = format!(
            "You are repairing a strict-citation answer.\n\
             Allowed source tags: {}.\n\
             Rules:\n\
             - Rewrite the answer using only the provided context.\n\
             - Every factual sentence must include at least one source tag like [S1].\n\
             - Use only allowed tags above.\n\
             - If evidence is insufficient, return exactly: {NOT_FOUND_MESSAGE}\n\n\
             Question:\n{question}\n\n\
             Context:\n{context}\n\n\
             Draft answer:\n{draft_answer}\n",
            source_ids.join(", ")
        );

        self.ollama
            .generate_text(
                &self.config.models.answer_model,
                &prompt,
                self.config.tokens.max_output_tokens,
                0.0,
            )
            .await
    }

    async fn boost_with_anchor_matches(
        &self,
        question: &str,
        retrieved: Vec<RetrievalResult>,
    ) -> Result<Vec<RetrievalResult>> {
        let anchors = extract_anchor_terms(question);
        if anchors.is_empty() {
            return Ok(retrieved);
        }

        let question_terms = normalized_question_terms(question);
        let death_question = is_death_question(question);
        let lexical_candidates = self.db.search_chunks_by_terms(&anchors, 160).await?;
        let include_timeline =
            is_overview_query(question) || is_later_query(question) || is_part_query(question);
        let timeline_candidates = if include_timeline {
            self.db.search_chunks_by_terms_chrono(&anchors, 320).await?
        } else {
            vec![]
        };

        let mut merged: HashMap<String, RetrievalResult> = HashMap::new();
        for item in retrieved {
            merged.insert(item.chunk.id.clone(), item);
        }

        for chunk in lexical_candidates {
            let mut bonus = lexical_overlap_score(&chunk.content, &question_terms);
            if bonus <= 0.0 {
                continue;
            }

            if has_any_term(&chunk.content, &anchors) {
                bonus += 0.06;
            }
            if death_question && contains_death_terms(&chunk.content) {
                bonus += 0.18;
            }

            if let Some(existing) = merged.get_mut(&chunk.id) {
                existing.score += bonus;
            } else {
                merged.insert(
                    chunk.id.clone(),
                    RetrievalResult {
                        chunk,
                        score: bonus,
                    },
                );
            }
        }

        if include_timeline {
            let timeline_selected =
                select_timeline_chunks(&timeline_candidates, 12, is_later_query(question));
            for (index, (_rowid, chunk)) in timeline_selected.into_iter().enumerate() {
                let mut bonus = 0.08 + 0.01 * (index as f32);
                if is_later_query(question) {
                    bonus += 0.05;
                }
                if has_any_term(&chunk.content, &anchors) {
                    bonus += 0.04;
                }

                if let Some(existing) = merged.get_mut(&chunk.id) {
                    existing.score += bonus;
                } else {
                    merged.insert(
                        chunk.id.clone(),
                        RetrievalResult {
                            chunk,
                            score: bonus,
                        },
                    );
                }
            }
        }

        let mut merged_items: Vec<RetrievalResult> = merged.into_values().collect();
        merged_items.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(merged_items)
    }

    async fn build_context_sources(
        &self,
        selected: Vec<RetrievalResult>,
    ) -> Result<Vec<ContextSource>> {
        let chunk_ids: Vec<String> = selected.iter().map(|item| item.chunk.id.clone()).collect();
        let rowid_map = self.db.chunk_rowids(&chunk_ids).await?;
        let part_markers = self.db.part_markers().await?;

        let mut out = Vec::with_capacity(selected.len());
        for (index, item) in selected.into_iter().enumerate() {
            let source_id = format!("S{}", index + 1);
            let rowid = rowid_map.get(&item.chunk.id).copied();
            let part = rowid.and_then(|id| infer_part_from_rowid(id, &part_markers));
            out.push(ContextSource {
                source_id,
                result: item,
                part,
            });
        }

        Ok(out)
    }
}

fn direct_death_answer_from_sources(question: &str, sources: &[ContextSource]) -> Option<String> {
    if !is_death_question(question) {
        return None;
    }

    let entity = extract_primary_entity(question)?;
    let entity_display = extract_entity_display(question, &entity);

    for source in sources {
        let content_lower = source.result.chunk.content.to_ascii_lowercase();
        if !content_lower.contains(&entity) || !contains_death_terms(&source.result.chunk.content) {
            continue;
        }

        let sentence = extract_relevant_sentence(&source.result.chunk.content, &entity)
            .unwrap_or_else(|| {
                source
                    .result
                    .chunk
                    .content
                    .split_whitespace()
                    .take(40)
                    .collect::<Vec<_>>()
                    .join(" ")
            });

        if is_when_question(question) && !contains_time_marker(&sentence) {
            return Some(format!(
                "{} is described as having been killed, but this passage does not provide a precise date. {} [{}]",
                entity_display, sentence, source.source_id
            ));
        }

        return Some(format!("{sentence} [{}]", source.source_id));
    }

    None
}

fn direct_part_answer_from_sources(question: &str, sources: &[ContextSource]) -> Option<String> {
    if !is_part_query(question) {
        return None;
    }

    let anchors = extract_anchor_terms(question);
    let mut best_source: Option<&ContextSource> = None;
    let mut best_score: f32 = f32::MIN;
    let mut explicit_date: Option<String> = None;

    for source in sources {
        let text = source.result.chunk.content.to_ascii_lowercase();
        if !anchors.is_empty()
            && !anchors.iter().all(|anchor| {
                text.contains(anchor.as_str()) || question.to_ascii_lowercase().contains(anchor)
            })
        {
            continue;
        }

        let date = extract_date_phrase(&source.result.chunk.content);
        let mut score = source.result.score;
        if date.is_some() {
            score += 0.15;
        }
        if source.part.is_some() {
            score += 0.20;
        }
        if score > best_score {
            best_score = score;
            best_source = Some(source);
            explicit_date = date;
        }
    }

    let source = best_source?;
    let part_text = source
        .part
        .clone()
        .unwrap_or_else(|| "unknown part".to_string());
    let anchor = anchors.first().cloned().unwrap_or_default();
    let sentence =
        extract_relevant_sentence(&source.result.chunk.content, &anchor).unwrap_or_else(|| {
            source
                .result
                .chunk
                .content
                .split_whitespace()
                .take(36)
                .collect::<Vec<_>>()
                .join(" ")
        });

    let mut answer = if let Some(date) = explicit_date {
        format!(
            "This event is in {part_text}. The closest explicit date in the source is {date}. {sentence} [{}]",
            source.source_id
        )
    } else {
        format!(
            "This event appears in {part_text}. {sentence} [{}]",
            source.source_id
        )
    };
    answer = answer.replace("unknown part", "an unspecified part");
    Some(answer)
}

fn resolve_question_with_history(question: &str, history: &[(String, String)]) -> String {
    let anchors = extract_anchor_terms(question);
    if !anchors.is_empty() || !has_pronoun_ref(question) {
        return question.to_string();
    }

    for (role, text) in history.iter().rev() {
        if role != "user" {
            continue;
        }
        if text.trim() == question.trim() {
            continue;
        }
        let previous_anchors = extract_anchor_terms(text);
        if let Some(anchor) = previous_anchors.first() {
            return format!("{question} (subject: {anchor})");
        }
    }

    question.to_string()
}

fn has_pronoun_ref(question: &str) -> bool {
    let q = question.to_ascii_lowercase();
    [
        " he ", " him ", " his ", " she ", " her ", " they ", " them ", " their ",
    ]
    .iter()
    .any(|token| q.contains(token))
        || q.starts_with("what does he")
        || q.starts_with("what did he")
}

fn sanitize_model_output(answer: String) -> String {
    let mut text = answer.trim().to_string();
    if text.starts_with("```") {
        let re = Regex::new(r"(?s)^```[a-zA-Z]*\n(.*)\n```$")
            .unwrap_or_else(|_| Regex::new("^$").unwrap());
        if let Some(caps) = re.captures(&text) {
            if let Some(body) = caps.get(1) {
                text = body.as_str().trim().to_string();
            }
        } else {
            text = text.replace("```", "").trim().to_string();
        }
    }
    text
}

fn expand_question_for_retrieval(question: &str) -> String {
    let mut expanded = question.to_string();
    if is_death_question(question) {
        expanded.push_str(" killed kill died death dead slain executed");
    }

    let anchors = extract_anchor_terms(question);
    if !anchors.is_empty() {
        expanded.push(' ');
        expanded.push_str(&anchors.join(" "));
    }
    expanded
}

fn infer_part_from_rowid(rowid: i64, markers: &[(i64, String)]) -> Option<String> {
    let mut current: Option<String> = None;
    for (marker_rowid, marker_text) in markers {
        if *marker_rowid > rowid {
            break;
        }
        current = normalize_part_marker(marker_text);
    }
    current
}

fn normalize_part_marker(raw: &str) -> Option<String> {
    let cleaned = raw.replace('\u{200B}', "").trim().to_string();
    let re =
        Regex::new(r"(?i)^part\s+([0-9]+[A-B]?)$").unwrap_or_else(|_| Regex::new("^$").unwrap());
    let caps = re.captures(&cleaned)?;
    let suffix = caps.get(1)?.as_str().to_ascii_uppercase();
    Some(format!("Part {suffix}"))
}

fn select_timeline_chunks(
    timeline: &[(i64, crate::models::Chunk)],
    max_take: usize,
    later_bias: bool,
) -> Vec<(i64, crate::models::Chunk)> {
    if timeline.is_empty() || max_take == 0 {
        return vec![];
    }

    let mut selected = Vec::new();
    let n = timeline.len();
    let target = max_take.min(n);

    let indices: Vec<usize> = if later_bias {
        (0..target)
            .map(|i| ((n - 1) * (i + target)) / (2 * target))
            .collect()
    } else {
        (0..target)
            .map(|i| ((n - 1) * i) / (target.saturating_sub(1).max(1)))
            .collect()
    };

    let mut seen = HashSet::new();
    for idx in indices {
        if seen.insert(idx) {
            selected.push(timeline[idx].clone());
        }
    }
    selected
}

fn extract_anchor_terms(question: &str) -> Vec<String> {
    let token_re = Regex::new(r"[A-Za-z0-9_]+").unwrap_or_else(|_| Regex::new("^").unwrap());
    let stopwords: HashSet<&'static str> = [
        "when", "what", "who", "where", "why", "how", "did", "does", "is", "are", "was", "were",
        "the", "a", "an", "in", "on", "to", "of", "for", "it", "its", "and", "or", "if", "then",
    ]
    .into_iter()
    .collect();

    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for m in token_re.find_iter(question) {
        let raw = m.as_str();
        let lower = raw.to_ascii_lowercase();
        if stopwords.contains(lower.as_str()) {
            continue;
        }

        let looks_like_entity = raw.chars().any(|c| c.is_ascii_digit())
            || (raw.len() >= 8 && raw.chars().any(|c| c.is_ascii_uppercase()));
        let useful_long_token = raw.len() >= 7;
        if !looks_like_entity && !useful_long_token {
            continue;
        }

        if seen.insert(lower.clone()) {
            out.push(lower);
        }
    }

    out
}

fn normalized_question_terms(question: &str) -> Vec<String> {
    let token_re = Regex::new(r"[A-Za-z0-9_]+").unwrap_or_else(|_| Regex::new("^").unwrap());
    let stopwords: HashSet<&'static str> = [
        "when", "what", "who", "where", "why", "how", "did", "does", "is", "are", "was", "were",
        "the", "a", "an", "in", "on", "to", "of", "for", "it", "its", "and", "or", "if", "then",
        "part", "chapter",
    ]
    .into_iter()
    .collect();

    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for m in token_re.find_iter(question) {
        let token = m.as_str().to_ascii_lowercase();
        if token.len() < 3 || stopwords.contains(token.as_str()) {
            continue;
        }
        if seen.insert(token.clone()) {
            out.push(token);
        }
    }

    if is_death_question(question) {
        for term in [
            "die", "died", "death", "killed", "slain", "executed", "dead",
        ] {
            if seen.insert(term.to_string()) {
                out.push(term.to_string());
            }
        }
    }

    out
}

fn lexical_overlap_score(content: &str, question_terms: &[String]) -> f32 {
    if question_terms.is_empty() {
        return 0.0;
    }
    let content_lower = content.to_ascii_lowercase();
    let overlap = question_terms
        .iter()
        .filter(|term| content_lower.contains(term.as_str()))
        .count();

    if overlap == 0 {
        0.0
    } else {
        0.03 * overlap as f32
    }
}

fn has_any_term(content: &str, terms: &[String]) -> bool {
    if terms.is_empty() {
        return false;
    }
    let content_lower = content.to_ascii_lowercase();
    terms
        .iter()
        .any(|term| content_lower.contains(term.as_str()))
}

fn is_overview_query(question: &str) -> bool {
    let q = question.to_ascii_lowercase();
    q.starts_with("who is ")
        || q.contains("overall summary")
        || q.contains("summarize")
        || q.contains("character arc")
}

fn is_later_query(question: &str) -> bool {
    let q = question.to_ascii_lowercase();
    q.contains("later in the novel")
        || q.contains("later on")
        || q.contains("towards the end")
        || q.contains("at the end")
}

fn is_part_query(question: &str) -> bool {
    let q = question.to_ascii_lowercase();
    q.contains("which part") || q.contains("what part") || q.contains("part of the book")
}

fn is_death_question(question: &str) -> bool {
    let q = question.to_ascii_lowercase();
    ((q.contains("when did") || q.contains("how did")) && q.contains("die"))
        || q.contains("death of")
        || (q.contains("killed") && (q.contains("who") || q.contains("when") || q.contains("how")))
}

fn is_when_question(question: &str) -> bool {
    let q = question.to_ascii_lowercase();
    q.contains("when")
}

fn contains_death_terms(content: &str) -> bool {
    let c = content.to_ascii_lowercase();
    ["killed", "died", "death", "slain", "executed", "dead"]
        .iter()
        .any(|term| c.contains(term))
}

fn contains_time_marker(sentence: &str) -> bool {
    let s = sentence.to_ascii_lowercase();
    let has_year = Regex::new(r"\b(19|20)\d{2}\b")
        .ok()
        .map(|re| re.is_match(&s))
        .unwrap_or(false);
    let has_time_word = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "part ",
        "chapter ",
    ]
    .iter()
    .any(|term| s.contains(term));

    has_year || has_time_word
}

fn extract_date_phrase(content: &str) -> Option<String> {
    let date_re = Regex::new(
        r"(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(st|nd|rd|th)?(?:,\s*\d{4})?\b",
    )
    .unwrap_or_else(|_| Regex::new("^$").unwrap());
    if let Some(found) = date_re.find(content) {
        return Some(found.as_str().to_string());
    }

    let year_re = Regex::new(r"\b(19|20)\d{2}\b").unwrap_or_else(|_| Regex::new("^$").unwrap());
    year_re.find(content).map(|m| m.as_str().to_string())
}

fn extract_relevant_sentence(content: &str, entity: &str) -> Option<String> {
    let death_terms = ["killed", "died", "death", "slain", "executed", "dead"];
    let entity_lower = entity.to_ascii_lowercase();

    for sentence in content.split(['.', '!', '?']) {
        let s = sentence.trim();
        if s.is_empty() {
            continue;
        }
        let lower = s.to_ascii_lowercase();
        if !lower.contains(&entity_lower) {
            continue;
        }
        if death_terms.iter().any(|term| lower.contains(term)) {
            return Some(s.to_string());
        }
    }
    None
}

fn extract_primary_entity(question: &str) -> Option<String> {
    let anchors = extract_anchor_terms(question);
    anchors.into_iter().max_by_key(|term| term.len())
}

fn extract_entity_display(question: &str, entity_lower: &str) -> String {
    let token_re = Regex::new(r"[A-Za-z0-9_]+").unwrap_or_else(|_| Regex::new("^").unwrap());
    for m in token_re.find_iter(question) {
        let raw = m.as_str();
        if raw.eq_ignore_ascii_case(entity_lower) {
            return raw.to_string();
        }
    }
    entity_lower.to_string()
}

fn has_image_keywords(question: &str) -> bool {
    let keywords = [
        "image",
        "picture",
        "photo",
        "illustration",
        "drawing",
        "panel",
        "visual",
        "what does it look like",
        "scene in the art",
    ];
    keywords.iter().any(|key| question.contains(key))
}

fn trim_to_context_budget(
    retrieved: Vec<RetrievalResult>,
    max_tokens: usize,
    max_chunks: usize,
) -> Vec<RetrievalResult> {
    let mut kept = Vec::new();
    let mut total_tokens = 0usize;

    for item in retrieved.into_iter().take(max_chunks) {
        let chunk_tokens = item.chunk.token_count as usize;
        if total_tokens + chunk_tokens > max_tokens {
            break;
        }
        total_tokens += chunk_tokens;
        kept.push(item);
    }

    kept
}

fn build_context(sources: &[ContextSource]) -> (String, Vec<String>) {
    let mut context = String::new();
    let mut ids = Vec::with_capacity(sources.len());

    for source in sources {
        context.push_str(&format!(
            "[{}] chunk_id={} kind={} chapter={} part={} page={}\n{}\n\n",
            source.source_id,
            source.result.chunk.id,
            source.result.chunk.kind.as_str(),
            source
                .result
                .chunk
                .chapter
                .clone()
                .unwrap_or_else(|| "-".to_string()),
            source.part.clone().unwrap_or_else(|| "-".to_string()),
            source
                .result
                .chunk
                .page
                .map(|p| p.to_string())
                .unwrap_or_else(|| "-".to_string()),
            source.result.chunk.content
        ));
        ids.push(source.source_id.clone());
    }

    (context, ids)
}

fn build_answer_prompt(
    question: &str,
    context: &str,
    history: &[(String, String)],
    strict: bool,
    source_ids: &[String],
    max_tokens: usize,
) -> String {
    let history_text = history
        .iter()
        .map(|(role, text)| format!("{role}: {text}"))
        .collect::<Vec<_>>()
        .join("\n");

    let strict_rules = if strict {
        format!(
            "Rules: Use only the provided context. Cite sources inline like [S1] for every factual sentence. \
             Allowed source tags: {}. If evidence is missing, respond exactly: {NOT_FOUND_MESSAGE}. \
             For overview/later questions, synthesize across early, middle, and late evidence. \
             Never wrap your answer in code fences.",
            source_ids.join(", ")
        )
    } else {
        "Rules: Prefer the provided context. Do not invent events or facts. If context is inconclusive, explicitly say so. \
         Cite source tags like [S1] when possible. For overview/later questions, synthesize across early, middle, and late evidence. \
         Never wrap your answer in code fences.".to_string()
    };

    format!(
        "You are a local novel QA assistant.\n{strict_rules}\n\nConversation:\n{history_text}\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nWrite a concise markdown answer. Max output tokens: {max_tokens}."
    )
}

fn enforce_strict_mode(answer: String, strict: bool) -> String {
    if !strict {
        return answer;
    }

    let trimmed = answer.trim();
    if trimmed.is_empty() {
        return NOT_FOUND_MESSAGE.to_string();
    }

    if trimmed.eq_ignore_ascii_case(NOT_FOUND_MESSAGE) {
        return NOT_FOUND_MESSAGE.to_string();
    }

    let citation_re = Regex::new(r"(?i)\[s\d+\]").unwrap_or_else(|_| Regex::new("^").unwrap());
    if citation_re.is_match(trimmed) {
        trimmed.to_string()
    } else {
        NOT_FOUND_MESSAGE.to_string()
    }
}

fn build_citations(sources: &[ContextSource], answer: &str, strict: bool) -> Vec<Citation> {
    if answer.trim().eq_ignore_ascii_case(NOT_FOUND_MESSAGE) {
        return vec![];
    }

    let mut citations = Vec::new();
    let citation_markers = extract_source_markers(answer);

    for marker in &citation_markers {
        let Some(source) = sources.iter().find(|source| source.source_id == *marker) else {
            continue;
        };

        let snippet = source
            .result
            .chunk
            .content
            .split_whitespace()
            .take(36)
            .collect::<Vec<_>>()
            .join(" ");

        citations.push(Citation {
            chunk_id: source.result.chunk.id.clone(),
            source_type: source.result.chunk.kind,
            chapter: source
                .part
                .clone()
                .or_else(|| source.result.chunk.chapter.clone()),
            page: source.result.chunk.page,
            snippet,
            score: source.result.score,
        });
    }

    if citations.is_empty() {
        if strict {
            return vec![];
        }

        for source in sources.iter().take(3) {
            citations.push(Citation {
                chunk_id: source.result.chunk.id.clone(),
                source_type: source.result.chunk.kind,
                chapter: source
                    .part
                    .clone()
                    .or_else(|| source.result.chunk.chapter.clone()),
                page: source.result.chunk.page,
                snippet: source
                    .result
                    .chunk
                    .content
                    .split_whitespace()
                    .take(24)
                    .collect::<Vec<_>>()
                    .join(" "),
                score: source.result.score,
            });
        }
    }

    citations
}

fn extract_source_markers(answer: &str) -> Vec<String> {
    let re = Regex::new(r"(?i)\[s(\d+)\]").unwrap_or_else(|_| Regex::new("^").unwrap());
    let mut markers = Vec::new();

    for captures in re.captures_iter(answer) {
        let Some(number) = captures.get(1).map(|m| m.as_str()) else {
            continue;
        };
        let marker = format!("S{}", number);
        if !markers.contains(&marker) {
            markers.push(marker);
        }
    }

    markers
}

fn has_source_citation(answer: &str) -> bool {
    let re = Regex::new(r"(?i)\[s\d+\]").unwrap_or_else(|_| Regex::new("^").unwrap());
    re.is_match(answer)
}

fn estimate_confidence(retrieved: &[RetrievalResult]) -> f32 {
    if retrieved.is_empty() {
        return 0.0;
    }

    let top = retrieved[0].score;
    let coverage = (retrieved.len() as f32 / 10.0).min(1.0);
    ((top * 20.0).min(1.0) * 0.7 + coverage * 0.3).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Chunk, SourceType};

    #[test]
    fn strict_mode_rejects_uncited_answer() {
        let answer = "This is unsupported.".to_string();
        let strict = enforce_strict_mode(answer, true);
        assert_eq!(strict, NOT_FOUND_MESSAGE);
    }

    #[test]
    fn strict_mode_accepts_cited_answer() {
        let answer = "It happened at dawn [S1].".to_string();
        let strict = enforce_strict_mode(answer.clone(), true);
        assert_eq!(strict, answer);
    }

    #[test]
    fn citation_builder_uses_referenced_chunks() {
        let chunk_id = "123e4567-e89b-12d3-a456-426614174000".to_string();
        let retrieved = vec![RetrievalResult {
            chunk: Chunk {
                id: chunk_id.clone(),
                content: "Evidence sentence from the novel.".to_string(),
                kind: SourceType::DocxText,
                chapter: Some("Chapter 2".to_string()),
                page: None,
                token_count: 6,
                source_hash: "abc".to_string(),
                image_path: None,
            },
            score: 0.55,
        }];

        let mapped = vec![ContextSource {
            source_id: "S1".to_string(),
            result: retrieved[0].clone(),
            part: Some("Part 2".to_string()),
        }];
        let citations = build_citations(&mapped, "Answer with evidence [S1].", true);

        assert_eq!(citations.len(), 1);
        assert_eq!(citations[0].chunk_id, chunk_id);
        assert_eq!(citations[0].chapter.as_deref(), Some("Part 2"));
    }

    #[test]
    fn not_found_has_no_citations() {
        let mapped: Vec<ContextSource> = vec![];
        let citations = build_citations(&mapped, NOT_FOUND_MESSAGE, true);
        assert!(citations.is_empty());
    }

    #[test]
    fn death_question_uses_direct_evidence_path() {
        let retrieved = RetrievalResult {
            chunk: Chunk {
                id: "abc".to_string(),
                content: "After the long war, Apollo345 was killed and the empire collapsed."
                    .to_string(),
                kind: SourceType::DocxText,
                chapter: Some("Part 1".to_string()),
                page: None,
                token_count: 14,
                source_hash: "h".to_string(),
                image_path: None,
            },
            score: 0.5,
        };
        let mapped = vec![ContextSource {
            source_id: "S1".to_string(),
            result: retrieved,
            part: Some("Part 1".to_string()),
        }];

        let answer = direct_death_answer_from_sources("When did Apollo345 die?", &mapped)
            .expect("expected direct death answer");
        assert!(answer.contains("Apollo345"));
        assert!(answer.contains("[S1]"));
        assert!(answer.contains("does not provide a precise date"));
    }

    #[test]
    fn strips_markdown_fences() {
        let input = "```markdown\nHello world\n```".to_string();
        assert_eq!(sanitize_model_output(input), "Hello world");
    }

    #[test]
    fn part_marker_is_normalized() {
        assert_eq!(normalize_part_marker("Part 7a").as_deref(), Some("Part 7A"));
        assert_eq!(normalize_part_marker("\u{200B} Part 3 ").as_deref(), Some("Part 3"));
    }

    #[test]
    fn part_question_returns_part_answer() {
        let source = ContextSource {
            source_id: "S1".to_string(),
            result: RetrievalResult {
                chunk: Chunk {
                    id: "cid".to_string(),
                    content: "On April 15th, AggravatedCow and company finally arrived on the Cuban shore."
                        .to_string(),
                    kind: SourceType::DocxText,
                    chapter: None,
                    page: None,
                    token_count: 14,
                    source_hash: "h".to_string(),
                    image_path: None,
                },
                score: 0.7,
            },
            part: Some("Part 7A".to_string()),
        };

        let answer =
            direct_part_answer_from_sources("When did AggravatedCow go to Cuba? Which part of the book?", &[source])
                .expect("expected part answer");
        assert!(answer.contains("Part 7A"));
        assert!(answer.contains("[S1]"));
    }
}
