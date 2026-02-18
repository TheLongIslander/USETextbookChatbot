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
    part_index: Option<i64>,
    rowid: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestedPartScope {
    Number(i64),
    Exact(i64),
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
        let requested_part_scope = extract_requested_part_scope(&resolved_question);
        let query_class = self.classify_query(&resolved_question).await;
        let expanded_question = expand_question_for_retrieval(&resolved_question);
        let broad_retrieval = is_broad_retrieval_query(&resolved_question);
        let retrieve_k = if broad_retrieval { 72 } else { 24 };
        let retrieved = self
            .retriever
            .retrieve(&expanded_question, retrieve_k)
            .await?;
        let retrieved = self
            .boost_with_anchor_matches(&resolved_question, retrieved)
            .await?;
        let retrieved = rerank_candidates(&resolved_question, retrieved);

        let broad_query = broad_retrieval;
        let pre_context_tokens = if broad_query {
            self.config.tokens.max_context_tokens.saturating_mul(2)
        } else {
            self.config.tokens.max_context_tokens
        };
        let pre_context_chunks = if broad_query { 24 } else { 12 };
        let selected = trim_to_context_budget(
            &resolved_question,
            retrieved,
            pre_context_tokens,
            pre_context_chunks,
        );

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
        let mut context_sources = self.build_context_sources(selected).await?;
        if let Some(scope) = requested_part_scope {
            context_sources = filter_sources_to_part_scope(context_sources, scope);
        }
        context_sources = rebalance_context_sources(
            &resolved_question,
            context_sources,
            self.config.tokens.max_context_tokens,
            10,
        );
        if let Some(scope) = requested_part_scope {
            context_sources = filter_sources_to_part_scope(context_sources, scope);
        }
        if context_sources.is_empty() {
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
        let max_output_tokens = effective_output_tokens(
            &resolved_question,
            request.verbose,
            self.config.tokens.max_output_tokens,
        );

        let prompt = build_answer_prompt(
            &request.question,
            &context,
            &history,
            request.strict,
            request.verbose,
            &source_ids,
            max_output_tokens,
        );

        let _permit = self.generation_limit.acquire().await?;
        let mut answer_text = self
            .ollama
            .generate_text(
                &self.config.models.answer_model,
                &prompt,
                max_output_tokens,
                0.1,
            )
            .await
            .unwrap_or_else(|_| NOT_FOUND_MESSAGE.to_string());
        answer_text = sanitize_model_output(answer_text);

        if request.strict
            && !strict_citation_requirements_met(&answer_text, &resolved_question, &source_ids)
        {
            let repaired = self
                .repair_answer_with_citations(
                    &request.question,
                    &context,
                    &answer_text,
                    &source_ids,
                    max_output_tokens,
                )
                .await
                .unwrap_or_default();

            if strict_citation_requirements_met(&repaired, &resolved_question, &source_ids) {
                answer_text = repaired;
            }
        }
        answer_text = sanitize_model_output(answer_text);

        if request.strict
            && !strict_citation_requirements_met(&answer_text, &resolved_question, &source_ids)
        {
            answer_text = NOT_FOUND_MESSAGE.to_string();
        }

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

        if request.strict
            && !strict_citation_requirements_met(&answer_text, &resolved_question, &source_ids)
        {
            answer_text = NOT_FOUND_MESSAGE.to_string();
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
        max_output_tokens: usize,
    ) -> Result<String> {
        let min_citations = minimum_required_unique_citations(question, source_ids.len());
        let prompt = format!(
            "You are repairing a strict-citation answer.\n\
             Allowed source tags: {}.\n\
             Rules:\n\
             - Rewrite the answer using only the provided context.\n\
             - Every factual sentence must include at least one source tag like [S1].\n\
             - Use only allowed tags above.\n\
             - Use at least {} distinct source tags when enough evidence exists.\n\
             - If evidence is insufficient, return exactly: {NOT_FOUND_MESSAGE}\n\n\
             Question:\n{question}\n\n\
             Context:\n{context}\n\n\
             Draft answer:\n{draft_answer}\n",
            source_ids.join(", "),
            min_citations
        );

        self.ollama
            .generate_text(
                &self.config.models.answer_model,
                &prompt,
                max_output_tokens,
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
        let strong_anchors = extract_strong_anchor_terms(question);
        let anchor_terms = if anchors.is_empty() {
            strong_anchors.clone()
        } else {
            anchors.clone()
        };

        if anchor_terms.is_empty() {
            return Ok(retrieved);
        }

        let question_terms = normalized_question_terms(question);
        let death_question = is_death_question(question);
        let lexical_candidates = self.db.search_chunks_by_terms(&anchor_terms, 240).await?;
        let conjunctive_terms = pick_conjunctive_terms(&strong_anchors, &anchor_terms);
        let lexical_all_candidates = if conjunctive_terms.len() >= 2 {
            self.db
                .search_chunks_by_all_terms(&conjunctive_terms, 120)
                .await?
        } else {
            vec![]
        };
        let include_timeline = is_broad_retrieval_query(question);
        let timeline_candidates = if include_timeline {
            self.collect_timeline_candidates(&anchor_terms, 220).await?
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

            if has_any_term(&chunk.content, &anchor_terms) {
                bonus += 0.06;
            }
            if death_question && contains_death_terms(&chunk.content) {
                bonus += 0.18;
            }

            add_or_insert_result(&mut merged, chunk, bonus);
        }

        for chunk in lexical_all_candidates {
            let mut bonus = lexical_overlap_score(&chunk.content, &question_terms) + 0.16;
            if has_all_terms(&chunk.content, &conjunctive_terms) {
                bonus += 0.10;
            }
            if death_question && contains_death_terms(&chunk.content) {
                bonus += 0.12;
            }

            add_or_insert_result(&mut merged, chunk, bonus);
        }

        if include_timeline {
            let timeline_selected =
                select_timeline_chunks(&timeline_candidates, 18, is_later_query(question));
            for (index, (_rowid, chunk)) in timeline_selected.into_iter().enumerate() {
                let mut bonus = 0.08 + 0.01 * (index as f32);
                if is_later_query(question) {
                    bonus += 0.05;
                }
                if has_any_term(&chunk.content, &anchor_terms) {
                    bonus += 0.04;
                }

                add_or_insert_result(&mut merged, chunk, bonus);
            }
        }

        if let Some(entity) = summary_focus_entity(question, &strong_anchors, &anchor_terms) {
            let entity_hits = self.db.entity_hits(&entity, 720).await?;
            if !entity_hits.is_empty() {
                let spine = select_entity_summary_spine(&entity_hits, is_later_query(question), 16);
                for hit in entity_hits.iter().take(240) {
                    let mut bonus = 0.06 + (hit.mention_count.max(1) as f32).ln_1p() * 0.05;
                    if is_later_query(question)
                        && temporal_bucket_by_part(hit.chunk.part_index) == 2
                    {
                        bonus += 0.05;
                    }
                    add_or_insert_result(&mut merged, hit.chunk.clone(), bonus);
                }

                for (idx, hit) in spine.into_iter().enumerate() {
                    let mut bonus = 0.24 + 0.01 * (idx as f32);
                    if is_later_query(question)
                        && temporal_bucket_by_part(hit.chunk.part_index) == 2
                    {
                        bonus += 0.08;
                    }
                    add_or_insert_result(&mut merged, hit.chunk, bonus);
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
            let part = item
                .chunk
                .part
                .clone()
                .or_else(|| rowid.and_then(|id| infer_part_from_rowid(id, &part_markers)));
            let part_index = item
                .chunk
                .part_index
                .or_else(|| part.as_deref().and_then(part_index_from_label));
            out.push(ContextSource {
                source_id,
                part_index,
                result: item,
                part,
                rowid,
            });
        }

        Ok(out)
    }

    async fn collect_timeline_candidates(
        &self,
        anchor_terms: &[String],
        slice: i64,
    ) -> Result<Vec<(i64, crate::models::Chunk)>> {
        if anchor_terms.is_empty() || slice <= 0 {
            return Ok(vec![]);
        }

        let total = self.db.count_chunks_by_terms(anchor_terms).await?;
        let asc = self
            .db
            .search_chunks_by_terms_chrono(anchor_terms, slice)
            .await?;
        let desc = self
            .db
            .search_chunks_by_terms_chrono_desc(anchor_terms, slice)
            .await?;
        let middle = if total > slice {
            let offset = (total / 2).saturating_sub(slice / 2).max(0);
            self.db
                .search_chunks_by_terms_chrono_offset(anchor_terms, slice, offset)
                .await?
        } else {
            vec![]
        };

        let mut merged: HashMap<String, (i64, crate::models::Chunk)> = HashMap::new();
        for (rowid, chunk) in asc.into_iter().chain(desc).chain(middle) {
            merged
                .entry(chunk.id.clone())
                .and_modify(|existing| {
                    if rowid < existing.0 {
                        *existing = (rowid, chunk.clone());
                    }
                })
                .or_insert((rowid, chunk));
        }

        let mut out: Vec<(i64, crate::models::Chunk)> = merged.into_values().collect();
        out.sort_by_key(|(rowid, _)| *rowid);
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
    let mut additions = Vec::new();
    let mut seen = HashSet::new();
    let requested_part_scope = extract_requested_part_scope(question);

    if is_death_question(question) {
        for term in [
            "killed", "kill", "died", "death", "dead", "slain", "executed", "murdered",
        ] {
            push_unique_term(&mut additions, &mut seen, term);
        }
    }

    if is_when_question(question) {
        for term in [
            "timeline", "date", "time", "part", "chapter", "before", "after",
        ] {
            push_unique_term(&mut additions, &mut seen, term);
        }
    }

    let q = question.to_ascii_lowercase();
    if [
        "arrive", "arrival", "reached", "reach", "went", "go to", "travel", "sailed",
    ]
    .iter()
    .any(|needle| q.contains(needle))
    {
        for term in [
            "arrive", "arrived", "reached", "went", "travel", "sailed", "journey", "moved",
        ] {
            push_unique_term(&mut additions, &mut seen, term);
        }
    }

    if q.contains(" in ") || q.contains(" to ") || q.contains(" from ") {
        for term in ["location", "place", "city", "country", "island", "region"] {
            push_unique_term(&mut additions, &mut seen, term);
        }
    }

    if is_part_query(question) {
        for term in ["part", "chapter", "section", "timeline"] {
            push_unique_term(&mut additions, &mut seen, term);
        }
    }

    if is_overview_query(question) && requested_part_scope.is_none() {
        for term in [
            "overview",
            "summary",
            "arc",
            "beginning",
            "middle",
            "end",
            "later",
        ] {
            push_unique_term(&mut additions, &mut seen, term);
        }
    }

    if is_later_query(question) && requested_part_scope.is_none() {
        for term in ["later", "end", "final", "aftermath", "eventually"] {
            push_unique_term(&mut additions, &mut seen, term);
        }
    }

    if let Some(scope) = requested_part_scope {
        match scope {
            RequestedPartScope::Number(number) => {
                for term in [
                    format!("part {number}"),
                    format!("part {number}a"),
                    format!("part {number}b"),
                ] {
                    push_unique_term(&mut additions, &mut seen, &term);
                }
            }
            RequestedPartScope::Exact(part_index) => {
                let number = part_index / 10;
                let suffix = match part_index % 10 {
                    1 => "a",
                    2 => "b",
                    _ => "",
                };

                if suffix.is_empty() {
                    let term = format!("part {number}");
                    push_unique_term(&mut additions, &mut seen, &term);
                } else {
                    let exact = format!("part {number}{suffix}");
                    push_unique_term(&mut additions, &mut seen, &exact);
                    let broad = format!("part {number}");
                    push_unique_term(&mut additions, &mut seen, &broad);
                }
            }
        }
    }

    let anchors = extract_anchor_terms(question);
    for anchor in anchors {
        push_unique_term(&mut additions, &mut seen, &anchor);
    }
    for phrase in extract_phrase_anchors(question) {
        push_unique_term(&mut additions, &mut seen, &phrase);
    }

    if !additions.is_empty() {
        expanded.push(' ');
        expanded.push_str(&additions.join(" "));
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

fn part_index_from_label(label: &str) -> Option<i64> {
    let normalized = normalize_part_marker(label)?;
    let re = Regex::new(r"(?i)^part\s+([0-9]+)\s*([ab])?$")
        .unwrap_or_else(|_| Regex::new("^$").unwrap());
    let caps = re.captures(&normalized)?;
    let number = caps.get(1)?.as_str().parse::<i64>().ok()?;
    let suffix = caps
        .get(2)
        .map(|m| m.as_str().to_ascii_uppercase())
        .unwrap_or_default();
    let suffix_offset = match suffix.as_str() {
        "A" => 1,
        "B" => 2,
        _ => 0,
    };
    Some(number * 10 + suffix_offset)
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

fn add_or_insert_result(
    merged: &mut HashMap<String, RetrievalResult>,
    chunk: crate::models::Chunk,
    bonus: f32,
) {
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

fn summary_focus_entity(
    question: &str,
    strong_anchors: &[String],
    anchors: &[String],
) -> Option<String> {
    if extract_requested_part_scope(question).is_some() {
        return None;
    }

    if !(is_overview_query(question)
        || is_comparison_query(question)
        || is_later_query(question)
        || is_part_query(question))
    {
        return None;
    }

    if let Some(entity) = strong_anchors.first() {
        return Some(entity.clone());
    }

    anchors
        .iter()
        .find(|term| term.len() >= 6 && !is_generic_conjunctive_term(term))
        .cloned()
}

fn select_entity_summary_spine(
    hits: &[crate::db::EntityHit],
    later_bias: bool,
    max_take: usize,
) -> Vec<crate::db::EntityHit> {
    if hits.is_empty() || max_take == 0 {
        return vec![];
    }

    let mut by_row = hits.to_vec();
    by_row.sort_by_key(|h| h.rowid);

    let mut selected = Vec::new();
    let mut seen = HashSet::new();

    let pick = |selected: &mut Vec<crate::db::EntityHit>,
                seen: &mut HashSet<String>,
                hit: &crate::db::EntityHit| {
        if seen.insert(hit.chunk.id.clone()) {
            selected.push(hit.clone());
        }
    };

    if let Some(first) = by_row.first() {
        pick(&mut selected, &mut seen, first);
    }
    if let Some(mid) = by_row.get(by_row.len() / 2) {
        pick(&mut selected, &mut seen, mid);
    }
    if let Some(last) = by_row.last() {
        pick(&mut selected, &mut seen, last);
    }

    let mut per_part: HashMap<i64, crate::db::EntityHit> = HashMap::new();
    for hit in hits {
        let Some(part_index) = hit.chunk.part_index else {
            continue;
        };
        let entry = per_part.entry(part_index).or_insert_with(|| hit.clone());
        if hit.mention_count > entry.mention_count
            || (hit.mention_count == entry.mention_count && hit.rowid > entry.rowid)
        {
            *entry = hit.clone();
        }
    }

    let mut part_hits: Vec<crate::db::EntityHit> = per_part.into_values().collect();
    part_hits.sort_by_key(|h| h.chunk.part_index.unwrap_or_default());
    for hit in part_hits {
        if selected.len() >= max_take {
            break;
        }
        pick(&mut selected, &mut seen, &hit);
    }

    if later_bias {
        for hit in by_row.iter().rev().take(max_take) {
            if selected.len() >= max_take {
                break;
            }
            pick(&mut selected, &mut seen, hit);
        }
    }

    selected.sort_by_key(|h| h.rowid);
    selected.truncate(max_take);
    selected
}

fn temporal_bucket_by_part(part_index: Option<i64>) -> usize {
    let Some(part_number) = part_number_from_part_index(part_index) else {
        return 1;
    };
    if part_number <= 3 {
        0
    } else if part_number <= 6 {
        1
    } else {
        2
    }
}

fn part_number_from_part_index(part_index: Option<i64>) -> Option<i64> {
    let index = part_index?;
    if index <= 0 {
        return None;
    }
    Some(index / 10)
}

fn extract_requested_part_scope(question: &str) -> Option<RequestedPartScope> {
    let re = Regex::new(r"(?i)\bpart\s+([0-9]{1,2}|[ivx]{1,5})\s*([ab])?\b")
        .unwrap_or_else(|_| Regex::new("^$").unwrap());
    let caps = re.captures(question)?;
    let number = parse_part_number_token(caps.get(1)?.as_str())?;
    if !(1..=20).contains(&number) {
        return None;
    }

    let suffix = caps
        .get(2)
        .map(|m| m.as_str().to_ascii_uppercase())
        .unwrap_or_default();
    let scope = match suffix.as_str() {
        "A" => RequestedPartScope::Exact(number * 10 + 1),
        "B" => RequestedPartScope::Exact(number * 10 + 2),
        _ => RequestedPartScope::Number(number),
    };

    Some(scope)
}

fn parse_part_number_token(raw: &str) -> Option<i64> {
    if let Ok(value) = raw.parse::<i64>() {
        return Some(value);
    }
    roman_to_int_token(raw)
}

fn roman_to_int_token(raw: &str) -> Option<i64> {
    let value = raw.trim().to_ascii_uppercase();
    if value.is_empty() || value.len() > 6 {
        return None;
    }

    let mut total = 0i64;
    let mut prev = 0i64;
    for ch in value.chars().rev() {
        let n = match ch {
            'I' => 1,
            'V' => 5,
            'X' => 10,
            _ => return None,
        };
        if n < prev {
            total -= n;
        } else {
            total += n;
            prev = n;
        }
    }

    if total <= 0 {
        None
    } else {
        Some(total)
    }
}

fn source_matches_part_scope(source: &ContextSource, scope: RequestedPartScope) -> bool {
    let Some(part_index) = source.part_index else {
        return false;
    };

    match scope {
        RequestedPartScope::Exact(expected) => part_index == expected,
        RequestedPartScope::Number(expected_number) => {
            part_number_from_part_index(Some(part_index)) == Some(expected_number)
        }
    }
}

fn filter_sources_to_part_scope(
    sources: Vec<ContextSource>,
    scope: RequestedPartScope,
) -> Vec<ContextSource> {
    let mut filtered: Vec<ContextSource> = sources
        .into_iter()
        .filter(|source| source_matches_part_scope(source, scope))
        .collect();
    renumber_sources(&mut filtered);
    filtered
}

fn is_broad_retrieval_query(question: &str) -> bool {
    if extract_requested_part_scope(question).is_some() {
        return false;
    }

    is_overview_query(question)
        || is_comparison_query(question)
        || is_later_query(question)
        || is_part_query(question)
}

fn rerank_candidates(question: &str, retrieved: Vec<RetrievalResult>) -> Vec<RetrievalResult> {
    if retrieved.is_empty() {
        return retrieved;
    }

    let question_terms = normalized_question_terms(question);
    let entity = extract_primary_entity(question);
    let has_part_scope = extract_requested_part_scope(question).is_some();
    let later_query = !has_part_scope && is_later_query(question);
    let overview_query =
        !has_part_scope && (is_overview_query(question) || is_comparison_query(question));

    let mut reranked = retrieved;
    for item in &mut reranked {
        let overlap = lexical_overlap_score(&item.chunk.content, &question_terms);
        item.score += overlap * 1.5;

        if let Some(entity_name) = &entity {
            if item
                .chunk
                .content
                .to_ascii_lowercase()
                .contains(entity_name.as_str())
            {
                item.score += 0.12;
            }
        }

        if later_query {
            match temporal_bucket_by_part(item.chunk.part_index) {
                2 => item.score += 0.16,
                0 => item.score -= 0.05,
                _ => {}
            }
        } else if overview_query {
            match temporal_bucket_by_part(item.chunk.part_index) {
                0 | 2 => item.score += 0.03,
                _ => {}
            }
        }
    }

    reranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    reranked
}

fn extract_anchor_terms(question: &str) -> Vec<String> {
    let token_re = Regex::new(r"[A-Za-z0-9_]+").unwrap_or_else(|_| Regex::new("^").unwrap());
    let stopwords = anchor_stopwords();

    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for m in token_re.find_iter(question) {
        let raw = m.as_str();
        let lower = raw.to_ascii_lowercase();
        if stopwords.contains(lower.as_str()) {
            continue;
        }

        let has_digit = raw.chars().any(|c| c.is_ascii_digit());
        let has_internal_upper = raw.chars().skip(1).any(|c| c.is_ascii_uppercase());
        let useful_length = raw.len() >= 4;
        if !has_digit && !has_internal_upper && !useful_length {
            continue;
        }

        if seen.insert(lower.clone()) {
            out.push(lower);
        }
    }

    for phrase in extract_phrase_anchors(question) {
        if seen.insert(phrase.clone()) {
            out.push(phrase);
        }
    }

    out
}

fn extract_strong_anchor_terms(question: &str) -> Vec<String> {
    let token_re = Regex::new(r"[A-Za-z0-9_]+").unwrap_or_else(|_| Regex::new("^").unwrap());
    let stopwords = anchor_stopwords();

    let mut seen = HashSet::new();
    let mut out = Vec::new();

    for m in token_re.find_iter(question) {
        let raw = m.as_str();
        let lower = raw.to_ascii_lowercase();
        if stopwords.contains(lower.as_str()) {
            continue;
        }

        let has_digit = raw.chars().any(|c| c.is_ascii_digit());
        let has_internal_upper = raw.chars().skip(1).any(|c| c.is_ascii_uppercase());
        if has_digit || has_internal_upper {
            if seen.insert(lower.clone()) {
                out.push(lower);
            }
        }
    }

    let loc_re =
        Regex::new(r"(?i)\b(?:in|to|from|at)\s+([A-Z][A-Za-z0-9_]*(?:\s+[A-Z][A-Za-z0-9_]*)?)\b")
            .unwrap_or_else(|_| Regex::new("^$").unwrap());
    for caps in loc_re.captures_iter(question) {
        let Some(m) = caps.get(1) else {
            continue;
        };
        let phrase = m.as_str().trim().to_ascii_lowercase();
        if phrase.len() < 4 {
            continue;
        }
        if seen.insert(phrase.clone()) {
            out.push(phrase);
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

fn has_all_terms(content: &str, terms: &[String]) -> bool {
    if terms.is_empty() {
        return false;
    }
    let content_lower = content.to_ascii_lowercase();
    terms
        .iter()
        .all(|term| content_lower.contains(term.as_str()))
}

fn extract_phrase_anchors(question: &str) -> Vec<String> {
    let re = Regex::new(r"\b([A-Z][A-Za-z0-9_]+(?:\s+[A-Z][A-Za-z0-9_]+)+)\b")
        .unwrap_or_else(|_| Regex::new("^$").unwrap());

    let mut seen = HashSet::new();
    let mut phrases = Vec::new();
    for caps in re.captures_iter(question) {
        let Some(m) = caps.get(1) else {
            continue;
        };
        let phrase = m.as_str().trim().to_ascii_lowercase();
        if phrase.len() < 5 {
            continue;
        }
        if seen.insert(phrase.clone()) {
            phrases.push(phrase);
        }
    }
    phrases
}

fn anchor_stopwords() -> HashSet<&'static str> {
    [
        "when",
        "what",
        "who",
        "where",
        "why",
        "how",
        "did",
        "does",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "in",
        "on",
        "to",
        "of",
        "for",
        "it",
        "its",
        "and",
        "or",
        "if",
        "then",
        "part",
        "chapter",
        "section",
        "summary",
        "summarize",
        "about",
        "book",
        "novel",
        "story",
        "context",
        "provided",
        "give",
        "more",
        "information",
        "tell",
        "please",
    ]
    .into_iter()
    .collect()
}

fn pick_conjunctive_terms(strong: &[String], anchors: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();

    for term in strong {
        if seen.insert(term.clone()) {
            out.push(term.clone());
        }
    }

    for term in anchors {
        if out.len() >= 3 {
            break;
        }
        if term.len() < 4 || is_generic_conjunctive_term(term) {
            continue;
        }
        if seen.insert(term.clone()) {
            out.push(term.clone());
        }
    }

    out
}

fn is_generic_conjunctive_term(term: &str) -> bool {
    [
        "later",
        "overall",
        "summary",
        "character",
        "chapter",
        "part",
        "event",
        "happened",
        "happen",
        "arrive",
        "arrived",
        "went",
        "go",
        "time",
        "date",
    ]
    .contains(&term)
}

fn is_overview_query(question: &str) -> bool {
    let q = question.to_ascii_lowercase();
    q.starts_with("who is ")
        || q.contains("overall summary")
        || q.contains("summarize")
        || q.contains("character arc")
        || q.contains("tell me about")
        || q.contains("more information about")
        || q.contains("profile of")
}

fn is_comparison_query(question: &str) -> bool {
    let q = question.to_ascii_lowercase();
    q.contains("who would win")
        || q.contains(" in a fight")
        || q.contains("versus")
        || q.contains(" vs ")
        || q.contains(" vs. ")
}

fn is_later_query(question: &str) -> bool {
    let q = question.to_ascii_lowercase();
    q.contains("later in the novel")
        || q.contains("later on")
        || q.contains("towards the end")
        || q.contains("at the end")
        || q.contains("later")
        || q.contains("eventually")
        || q.contains("after this")
        || q.contains("after that")
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
    question: &str,
    retrieved: Vec<RetrievalResult>,
    max_tokens: usize,
    max_chunks: usize,
) -> Vec<RetrievalResult> {
    if max_tokens == 0 || max_chunks == 0 {
        return vec![];
    }

    let broad_query = is_broad_retrieval_query(question);
    let candidates: Vec<RetrievalResult> = retrieved
        .into_iter()
        .take(max_chunks.saturating_mul(6).max(max_chunks))
        .collect();

    if candidates.is_empty() {
        return vec![];
    }

    let mut kept = Vec::new();
    let mut seen_ids: HashSet<String> = HashSet::new();
    let mut bucket_counts: HashMap<String, usize> = HashMap::new();
    let mut total_tokens = 0usize;
    let bucket_quota = if broad_query { 1 } else { 2 };

    for pass in 0..2 {
        for item in &candidates {
            if kept.len() >= max_chunks {
                break;
            }

            if seen_ids.contains(&item.chunk.id) {
                continue;
            }

            let chunk_tokens = item.chunk.token_count as usize;
            if chunk_tokens > max_tokens || total_tokens + chunk_tokens > max_tokens {
                continue;
            }

            let bucket = diversity_bucket(&item.chunk);
            let bucket_count = bucket_counts.get(&bucket).copied().unwrap_or(0);
            if pass == 0 && bucket_count >= bucket_quota {
                continue;
            }

            total_tokens += chunk_tokens;
            seen_ids.insert(item.chunk.id.clone());
            *bucket_counts.entry(bucket).or_insert(0) += 1;
            kept.push(item.clone());
        }
        if kept.len() >= max_chunks {
            break;
        }
    }

    kept
}

fn diversity_bucket(chunk: &crate::models::Chunk) -> String {
    if let Some(chapter) = &chunk.chapter {
        let trimmed = chapter.trim();
        if !trimmed.is_empty() {
            return format!("chapter:{trimmed}");
        }
    }

    if let Some(page) = chunk.page {
        return format!("page:{}", page / 5);
    }

    format!("kind:{}", chunk.kind.as_str())
}

fn push_unique_term(terms: &mut Vec<String>, seen: &mut HashSet<String>, term: &str) {
    let cleaned = term.trim().to_ascii_lowercase();
    if cleaned.is_empty() {
        return;
    }
    if seen.insert(cleaned.clone()) {
        terms.push(cleaned);
    }
}

fn rebalance_context_sources(
    question: &str,
    sources: Vec<ContextSource>,
    max_tokens: usize,
    max_chunks: usize,
) -> Vec<ContextSource> {
    if max_tokens == 0 || max_chunks == 0 || sources.is_empty() {
        return vec![];
    }

    if extract_requested_part_scope(question).is_some() && is_overview_query(question) {
        let mut out = select_chronological_spread_sources(sources, max_tokens, max_chunks);
        renumber_sources(&mut out);
        return out;
    }

    let broad_query = is_broad_retrieval_query(question);
    if !broad_query {
        let mut out = trim_context_sources_by_budget(sources, max_tokens, max_chunks);
        renumber_sources(&mut out);
        return out;
    }

    let mut ranked = sources;
    ranked.sort_by(|a, b| {
        b.result
            .score
            .partial_cmp(&a.result.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let rowids: Vec<i64> = ranked.iter().filter_map(|s| s.rowid).collect();
    if rowids.len() < 4 {
        let mut out = trim_context_sources_by_budget(ranked, max_tokens, max_chunks);
        renumber_sources(&mut out);
        return out;
    }

    let min_rowid = rowids.iter().copied().min().unwrap_or(0);
    let max_rowid = rowids.iter().copied().max().unwrap_or(min_rowid);
    let later_bias = is_later_query(question) || question.to_ascii_lowercase().contains("latest");
    let quotas: [usize; 3] = if later_bias { [1, 2, 4] } else { [2, 2, 2] };

    let mut selected = Vec::new();
    let mut bucket_counts = [0usize; 3];
    let mut seen = HashSet::new();
    let mut total_tokens = 0usize;
    let overview_query = is_overview_query(question);

    if overview_query {
        let mut best_by_part: HashMap<i64, ContextSource> = HashMap::new();
        for src in &ranked {
            let Some(part_index) = src.part_index else {
                continue;
            };
            let entry = best_by_part
                .entry(part_index)
                .or_insert_with(|| src.clone());
            if src.result.score > entry.result.score {
                *entry = src.clone();
            }
        }

        let mut part_sources: Vec<ContextSource> = best_by_part.into_values().collect();
        part_sources.sort_by_key(|s| s.part_index.unwrap_or_default());

        for src in part_sources {
            if selected.len() >= max_chunks {
                break;
            }
            if seen.contains(&src.result.chunk.id) {
                continue;
            }
            let chunk_tokens = src.result.chunk.token_count.max(0) as usize;
            if chunk_tokens > max_tokens || total_tokens + chunk_tokens > max_tokens {
                continue;
            }
            total_tokens += chunk_tokens;
            seen.insert(src.result.chunk.id.clone());
            bucket_counts[temporal_bucket(src.rowid, min_rowid, max_rowid)] += 1;
            selected.push(src);
        }
    }

    for src in &ranked {
        if selected.len() >= max_chunks {
            break;
        }

        let chunk_tokens = src.result.chunk.token_count.max(0) as usize;
        if chunk_tokens > max_tokens || total_tokens + chunk_tokens > max_tokens {
            continue;
        }

        let bucket = temporal_bucket(src.rowid, min_rowid, max_rowid);
        if bucket_counts[bucket] >= quotas[bucket] {
            continue;
        }

        if seen.insert(src.result.chunk.id.clone()) {
            selected.push(src.clone());
            bucket_counts[bucket] += 1;
            total_tokens += chunk_tokens;
        }
    }

    for src in &ranked {
        if selected.len() >= max_chunks {
            break;
        }

        if seen.contains(&src.result.chunk.id) {
            continue;
        }

        let chunk_tokens = src.result.chunk.token_count.max(0) as usize;
        if chunk_tokens > max_tokens || total_tokens + chunk_tokens > max_tokens {
            continue;
        }

        if seen.insert(src.result.chunk.id.clone()) {
            selected.push(src.clone());
            total_tokens += chunk_tokens;
        }
    }

    selected.sort_by(|a, b| {
        let a_row = a.rowid.unwrap_or(i64::MAX);
        let b_row = b.rowid.unwrap_or(i64::MAX);
        a_row.cmp(&b_row)
    });

    renumber_sources(&mut selected);
    selected
}

fn select_chronological_spread_sources(
    mut sources: Vec<ContextSource>,
    max_tokens: usize,
    max_chunks: usize,
) -> Vec<ContextSource> {
    if sources.is_empty() || max_tokens == 0 || max_chunks == 0 {
        return vec![];
    }

    sources.sort_by_key(|s| s.rowid.unwrap_or(i64::MAX));
    let n = sources.len();
    let target = n.min(max_chunks);
    let indices: Vec<usize> = (0..target)
        .map(|i| ((n - 1) * i) / (target.saturating_sub(1).max(1)))
        .collect();

    let mut selected = Vec::new();
    let mut seen = HashSet::new();
    let mut total_tokens = 0usize;

    for idx in indices {
        let src = sources[idx].clone();
        if seen.contains(&src.result.chunk.id) {
            continue;
        }

        let chunk_tokens = src.result.chunk.token_count.max(0) as usize;
        if chunk_tokens > max_tokens || total_tokens + chunk_tokens > max_tokens {
            continue;
        }

        seen.insert(src.result.chunk.id.clone());
        total_tokens += chunk_tokens;
        selected.push(src);
    }

    let mut by_score = sources;
    by_score.sort_by(|a, b| {
        b.result
            .score
            .partial_cmp(&a.result.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for src in by_score {
        if selected.len() >= max_chunks {
            break;
        }
        if seen.contains(&src.result.chunk.id) {
            continue;
        }

        let chunk_tokens = src.result.chunk.token_count.max(0) as usize;
        if chunk_tokens > max_tokens || total_tokens + chunk_tokens > max_tokens {
            continue;
        }

        seen.insert(src.result.chunk.id.clone());
        total_tokens += chunk_tokens;
        selected.push(src);
    }

    selected.sort_by_key(|s| s.rowid.unwrap_or(i64::MAX));
    selected
}

fn trim_context_sources_by_budget(
    sources: Vec<ContextSource>,
    max_tokens: usize,
    max_chunks: usize,
) -> Vec<ContextSource> {
    let mut kept = Vec::new();
    let mut total_tokens = 0usize;

    for src in sources {
        if kept.len() >= max_chunks {
            break;
        }
        let chunk_tokens = src.result.chunk.token_count.max(0) as usize;
        if chunk_tokens > max_tokens || total_tokens + chunk_tokens > max_tokens {
            continue;
        }
        total_tokens += chunk_tokens;
        kept.push(src);
    }

    kept
}

fn temporal_bucket(rowid: Option<i64>, min_rowid: i64, max_rowid: i64) -> usize {
    let Some(row) = rowid else {
        return 1;
    };
    let span = (max_rowid - min_rowid).max(1) as f32;
    let position = (row - min_rowid) as f32 / span;
    if position < 0.34 {
        0
    } else if position < 0.67 {
        1
    } else {
        2
    }
}

fn renumber_sources(sources: &mut [ContextSource]) {
    for (idx, source) in sources.iter_mut().enumerate() {
        source.source_id = format!("S{}", idx + 1);
    }
}

fn build_context(sources: &[ContextSource]) -> (String, Vec<String>) {
    let mut context = String::new();
    let mut ids = Vec::with_capacity(sources.len());

    for source in sources {
        context.push_str(&format!(
            "[{}] chunk_id={} kind={} chapter={} part={} part_index={} page={} rowid={}\n{}\n\n",
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
                .part_index
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
            source
                .result
                .chunk
                .page
                .map(|p| p.to_string())
                .unwrap_or_else(|| "-".to_string()),
            source
                .rowid
                .map(|v| v.to_string())
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
    verbose: bool,
    source_ids: &[String],
    max_tokens: usize,
) -> String {
    let requested_part_scope = extract_requested_part_scope(question);
    let history_text = history
        .iter()
        .map(|(role, text)| format!("{role}: {text}"))
        .collect::<Vec<_>>()
        .join("\n");
    let synthesis_hint = if requested_part_scope.is_none()
        && (is_overview_query(question)
            || is_comparison_query(question)
            || is_later_query(question))
    {
        "When answering character/overview/comparison questions, explicitly cover early (Parts 1-3), middle (Parts 4-6), and most recent events (Parts 7A-7B). Prioritize the latest canon events if there is tension."
    } else {
        ""
    };
    let scope_rule = if requested_part_scope.is_some() {
        "The user requested a specific part. Use evidence only from that part scope."
    } else {
        "For overview/comparison/later questions, synthesize across early, middle, and late evidence."
    };

    let strict_rules = if strict {
        format!(
            "Rules: Use only the provided context. Cite sources inline like [S1] for every factual sentence. \
             Allowed source tags: {}. If evidence is missing, respond exactly: {NOT_FOUND_MESSAGE}. \
             {scope_rule} \
             Never wrap your answer in code fences. {}",
            source_ids.join(", "),
            synthesis_hint
        )
    } else {
        format!(
            "Rules: Prefer the provided context. Do not invent events or facts. If context is inconclusive, explicitly say so. \
             Cite source tags like [S1] when possible. {scope_rule} \
             Never wrap your answer in code fences. {}",
            synthesis_hint
        )
    };

    let style_rules = if verbose {
        if requested_part_scope.is_some() {
            "Write a detailed markdown answer focused only on the requested part. Use multiple citations distributed across the answer, and do not use early/middle/latest arc sections unless the user asks for full-book coverage."
        } else {
            "Write a detailed markdown answer. For character/overview/comparison questions, include: (1) early arc, (2) middle arc, (3) latest arc, and (4) current status/implications. Keep claims grounded to cited evidence only."
        }
    } else {
        "Write a concise markdown answer focused on direct evidence."
    };

    format!(
        "You are a local novel QA assistant.\n{strict_rules}\n\nConversation:\n{history_text}\n\nContext:\n{context}\n\nQuestion:\n{question}\n\n{style_rules}\nMax output tokens: {max_tokens}."
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
            source_id: source.source_id.clone(),
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

        for source in sources {
            citations.push(Citation {
                source_id: source.source_id.clone(),
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

fn has_unknown_source_markers(answer: &str, source_ids: &[String]) -> bool {
    if source_ids.is_empty() {
        return !extract_source_markers(answer).is_empty();
    }

    let allowed: HashSet<String> = source_ids
        .iter()
        .map(|id| id.trim().to_ascii_uppercase())
        .collect();

    extract_source_markers(answer)
        .into_iter()
        .any(|marker| !allowed.contains(&marker.to_ascii_uppercase()))
}

fn minimum_required_unique_citations(question: &str, available_sources: usize) -> usize {
    if available_sources <= 1 {
        return available_sources.max(1);
    }

    if extract_requested_part_scope(question).is_some() && is_overview_query(question) {
        return available_sources.min(2).max(1);
    }

    if is_overview_query(question) {
        return available_sources.min(4).max(2);
    }

    if is_comparison_query(question) || is_later_query(question) {
        return available_sources.min(2);
    }

    1
}

fn should_enforce_sentence_level_citations(question: &str) -> bool {
    if extract_requested_part_scope(question).is_some() {
        return false;
    }

    is_overview_query(question) || is_comparison_query(question) || is_later_query(question)
}

fn is_factual_segment(segment: &str) -> bool {
    let trimmed = segment.trim();
    if trimmed.is_empty() {
        return false;
    }

    if trimmed.starts_with('#') || trimmed.starts_with('>') {
        return false;
    }

    let meaningful_words = trimmed
        .split_whitespace()
        .filter(|word| {
            let w = word.trim();
            !w.is_empty() && !w.starts_with("[S") && !w.starts_with("[s")
        })
        .count();
    if meaningful_words < 6 {
        return false;
    }

    let alpha_chars = trimmed.chars().filter(|c| c.is_ascii_alphabetic()).count();
    alpha_chars >= 16
}

fn has_uncited_factual_segment(answer: &str) -> bool {
    let citation_re = Regex::new(r"(?i)\[s\d+\]").unwrap_or_else(|_| Regex::new("^").unwrap());

    answer
        .split('\n')
        .flat_map(|line| line.split(['.', '!', '?']))
        .filter(|segment| is_factual_segment(segment))
        .any(|segment| !citation_re.is_match(segment))
}

fn strict_citation_requirements_met(answer: &str, question: &str, source_ids: &[String]) -> bool {
    if answer.trim().eq_ignore_ascii_case(NOT_FOUND_MESSAGE) {
        return true;
    }

    if !has_source_citation(answer) || has_unknown_source_markers(answer, source_ids) {
        return false;
    }

    let unique_citations = extract_source_markers(answer).len();
    if unique_citations < minimum_required_unique_citations(question, source_ids.len()) {
        return false;
    }

    if should_enforce_sentence_level_citations(question) && has_uncited_factual_segment(answer) {
        return false;
    }

    true
}

fn effective_output_tokens(question: &str, verbose: bool, base: usize) -> usize {
    if base == 0 {
        return 0;
    }

    let mut out = base;
    if verbose {
        out = out.saturating_mul(2);
    }
    if is_broad_retrieval_query(question) {
        out = out.saturating_add(base / 2);
    }

    let cap = base.max(1400);
    out.clamp(base, cap)
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
                part: Some("Part 2".to_string()),
                part_index: Some(20),
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
            part_index: Some(20),
            rowid: Some(20),
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
                part: Some("Part 1".to_string()),
                part_index: Some(10),
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
            part_index: Some(10),
            rowid: Some(5),
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
        assert_eq!(
            normalize_part_marker("\u{200B} Part 3 ").as_deref(),
            Some("Part 3")
        );
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
                    part: Some("Part 7A".to_string()),
                    part_index: Some(71),
                    page: None,
                    token_count: 14,
                    source_hash: "h".to_string(),
                    image_path: None,
                },
                score: 0.7,
            },
            part: Some("Part 7A".to_string()),
            part_index: Some(71),
            rowid: Some(140),
        };

        let answer = direct_part_answer_from_sources(
            "When did AggravatedCow go to Cuba? Which part of the book?",
            &[source],
        )
        .expect("expected part answer");
        assert!(answer.contains("Part 7A"));
        assert!(answer.contains("[S1]"));
    }

    #[test]
    fn rebalance_prefers_temporal_coverage_for_overview() {
        let make_source = |id: &str, score: f32, rowid: i64| ContextSource {
            source_id: "S0".to_string(),
            result: RetrievalResult {
                chunk: Chunk {
                    id: id.to_string(),
                    content: format!("TheLongIslander event {id}"),
                    kind: SourceType::DocxText,
                    chapter: None,
                    part: None,
                    part_index: None,
                    page: None,
                    token_count: 30,
                    source_hash: "h".to_string(),
                    image_path: None,
                },
                score,
            },
            part: None,
            part_index: None,
            rowid: Some(rowid),
        };

        let sources = vec![
            make_source("a", 0.95, 10),
            make_source("b", 0.90, 15),
            make_source("c", 0.85, 20),
            make_source("d", 0.80, 120),
            make_source("e", 0.75, 130),
            make_source("f", 0.70, 240),
            make_source("g", 0.65, 250),
        ];

        let selected = rebalance_context_sources("Who is TheLongIslander?", sources, 220, 6);
        assert!(!selected.is_empty());
        assert!(selected.iter().any(|s| s.rowid.unwrap_or(0) >= 200));
        assert!(selected.iter().any(|s| s.rowid.unwrap_or(0) <= 30));
        assert_eq!(selected[0].source_id, "S1");
    }

    #[test]
    fn rebalance_later_query_biases_late_chunks() {
        let make_source = |id: &str, score: f32, rowid: i64| ContextSource {
            source_id: "S0".to_string(),
            result: RetrievalResult {
                chunk: Chunk {
                    id: id.to_string(),
                    content: format!("Arc event {id}"),
                    kind: SourceType::DocxText,
                    chapter: None,
                    part: None,
                    part_index: None,
                    page: None,
                    token_count: 28,
                    source_hash: "h".to_string(),
                    image_path: None,
                },
                score,
            },
            part: None,
            part_index: None,
            rowid: Some(rowid),
        };

        let sources = vec![
            make_source("a", 1.00, 20),
            make_source("b", 0.98, 30),
            make_source("c", 0.96, 45),
            make_source("d", 0.80, 180),
            make_source("e", 0.70, 240),
            make_source("f", 0.68, 260),
            make_source("g", 0.66, 280),
        ];

        let selected =
            rebalance_context_sources("What does he do later in the novel?", sources, 210, 6);
        let late_count = selected
            .iter()
            .filter(|s| s.rowid.unwrap_or(0) >= 180)
            .count();
        assert!(late_count >= 3);
    }

    #[test]
    fn temporal_bucket_uses_defined_part_ranges() {
        assert_eq!(temporal_bucket_by_part(Some(10)), 0);
        assert_eq!(temporal_bucket_by_part(Some(30)), 0);
        assert_eq!(temporal_bucket_by_part(Some(40)), 1);
        assert_eq!(temporal_bucket_by_part(Some(60)), 1);
        assert_eq!(temporal_bucket_by_part(Some(71)), 2);
        assert_eq!(temporal_bucket_by_part(Some(72)), 2);
    }

    #[test]
    fn detects_requested_part_scope() {
        assert_eq!(
            extract_requested_part_scope("Summarize Part 7"),
            Some(RequestedPartScope::Number(7))
        );
        assert_eq!(
            extract_requested_part_scope("Summarize Part 7A"),
            Some(RequestedPartScope::Exact(71))
        );
        assert_eq!(
            extract_requested_part_scope("Summarize Part VII"),
            Some(RequestedPartScope::Number(7))
        );
    }

    #[test]
    fn scoped_part_summary_is_not_broad_query() {
        assert!(!is_broad_retrieval_query("Summarize Part 7"));
        assert!(is_broad_retrieval_query("Summarize Apollo345"));
    }

    #[test]
    fn part_scope_filter_keeps_only_requested_part() {
        let make_source = |id: &str, part_index: i64| ContextSource {
            source_id: "S0".to_string(),
            result: RetrievalResult {
                chunk: Chunk {
                    id: id.to_string(),
                    content: format!("Event {id}"),
                    kind: SourceType::DocxText,
                    chapter: None,
                    part: None,
                    part_index: Some(part_index),
                    page: None,
                    token_count: 12,
                    source_hash: "h".to_string(),
                    image_path: None,
                },
                score: 0.5,
            },
            part: Some(format!("Part {}", part_index / 10)),
            part_index: Some(part_index),
            rowid: Some(part_index),
        };

        let filtered = filter_sources_to_part_scope(
            vec![
                make_source("p6", 60),
                make_source("p7a", 71),
                make_source("p7b", 72),
            ],
            RequestedPartScope::Number(7),
        );
        assert_eq!(filtered.len(), 2);
        assert!(filtered
            .iter()
            .all(|s| { part_number_from_part_index(s.part_index) == Some(7) }));
        assert_eq!(filtered[0].source_id, "S1");
        assert_eq!(filtered[1].source_id, "S2");
    }

    #[test]
    fn scoped_part_prompt_does_not_force_arc_sections() {
        let prompt = build_answer_prompt(
            "Summarize Part 7",
            "Context",
            &[],
            true,
            true,
            &["S1".to_string()],
            512,
        );

        assert!(prompt.contains("requested a specific part"));
        assert!(!prompt.contains("include: (1) early arc, (2) middle arc, (3) latest arc"));
    }

    #[test]
    fn who_would_win_queries_are_broad() {
        let q = "Who would win in a fight - TheLongIslander or Leafsfan2003?";
        assert!(is_comparison_query(q));
        assert!(is_broad_retrieval_query(q));
    }

    #[test]
    fn detects_unknown_source_markers() {
        let source_ids = vec!["S1".to_string(), "S2".to_string()];
        assert!(!has_unknown_source_markers(
            "Supported [S1] and [S2].",
            &source_ids
        ));
        assert!(has_unknown_source_markers("Unknown [S9].", &source_ids));
    }

    #[test]
    fn strict_citation_requirements_reject_sparse_overview_citations() {
        let source_ids = vec![
            "S1".to_string(),
            "S2".to_string(),
            "S3".to_string(),
            "S4".to_string(),
            "S5".to_string(),
        ];
        let answer = "Part 7A opens with a forced-diversity challenge and a mysterious mission setup [S1]. Part 7B shifts to governance conflicts and retaliation threats tied to Geektown disputes [S2].";

        assert!(!strict_citation_requirements_met(
            answer,
            "Summarize Apollo345's overall arc",
            &source_ids
        ));
    }

    #[test]
    fn strict_citation_requirements_enforce_sentence_level_citations() {
        let source_ids = vec!["S1".to_string(), "S2".to_string(), "S3".to_string()];
        let answer = "Part 7A opens with a forced-diversity challenge and a mission briefing for the trio [S1]. The group advances toward the manor while still uncertain about who chose them. Part 7B records legal grievances around griefing and declared penalties affecting Geektown governance [S2]. The close emphasizes enforcement measures and faction-level consequences [S3].";

        assert!(!strict_citation_requirements_met(
            answer,
            "Summarize Apollo345's overall arc",
            &source_ids
        ));
    }

    #[test]
    fn strict_citation_requirements_accept_dense_overview_citations() {
        let source_ids = vec![
            "S1".to_string(),
            "S2".to_string(),
            "S3".to_string(),
            "S4".to_string(),
            "S5".to_string(),
        ];
        let answer = "Part 7A begins with TheLongIslander, BangladeshiJew, and Leafsfan2003 waking in a tent and finding mission-oriented instructions [S1]. The trio follows the guidance toward the manor while debating trust and intent behind the setup [S2]. Part 7B pivots to documented griefing allegations tied to Tax_Day's old property and disruption of rebuilding work [S3]. The final warnings outline banishment and war escalation as consequences for continued violations [S4].";

        assert!(strict_citation_requirements_met(
            answer,
            "Summarize Part 7 of the book",
            &source_ids
        ));
    }

    #[test]
    fn part_scoped_summary_needs_two_sources_not_four() {
        let source_ids = vec![
            "S1".to_string(),
            "S2".to_string(),
            "S3".to_string(),
            "S4".to_string(),
            "S5".to_string(),
        ];
        let answer = "Part 7A opens with the forced-diversity setup and mission instructions for the trio [S1]. Part 7B shifts to grievance documentation and explicit enforcement threats around Geektown disputes [S2].";

        assert!(strict_citation_requirements_met(
            answer,
            "Summarize Part 7 of the book",
            &source_ids
        ));
    }

    #[test]
    fn part_scoped_summary_does_not_require_every_sentence_cited() {
        let source_ids = vec!["S1".to_string(), "S2".to_string(), "S3".to_string()];
        let answer = "Part 7A opens with a forced-diversity challenge and mission setup [S1]. The group advances toward the manor while still uncertain about who chose them. Part 7B records griefing allegations and declared penalties tied to Geektown governance [S2].";

        assert!(strict_citation_requirements_met(
            answer,
            "Summarize Part 7",
            &source_ids
        ));
    }

    #[test]
    fn effective_output_tokens_expands_for_verbose_broad_queries() {
        let base = 500;
        let q = "Who would win in a fight - TheLongIslander or Leafsfan2003?";
        let expanded = effective_output_tokens(q, true, base);
        assert!(expanded > base);
        assert!(expanded <= 1400);
    }
}
