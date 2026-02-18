pub mod docx;
pub mod pdf;

use std::path::Path;

use anyhow::{Context, Result};
use chrono::Utc;
use regex::Regex;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::config::AppConfig;
use crate::db::{Database, EntityMention};
use crate::models::{Chunk, IngestManifest, IngestRequest, IngestStatus, SourceType, SourceUnit};
use crate::ollama::OllamaClient;
use crate::qdrant_store::{QdrantPayload, QdrantPoint, QdrantStore};
use crate::tantivy_store::TantivyStore;

#[derive(Clone)]
pub struct Ingestor {
    config: AppConfig,
    db: Database,
    ollama: OllamaClient,
    qdrant: QdrantStore,
    tantivy: TantivyStore,
}

#[derive(Debug, Clone)]
pub struct IngestResult {
    pub chunk_count: i64,
    pub image_count: i64,
    pub skipped: bool,
}

impl Ingestor {
    pub fn new(
        config: AppConfig,
        db: Database,
        ollama: OllamaClient,
        qdrant: QdrantStore,
        tantivy: TantivyStore,
    ) -> Self {
        Self {
            config,
            db,
            ollama,
            qdrant,
            tantivy,
        }
    }

    pub async fn ingest<F>(
        &self,
        job_id: &str,
        request: IngestRequest,
        mut progress: F,
    ) -> Result<IngestResult>
    where
        F: FnMut(IngestStatus) + Send,
    {
        let now = Utc::now();
        let mut status = IngestStatus {
            job_id: job_id.to_string(),
            status: "running".to_string(),
            stage: "hashing_sources".to_string(),
            message: None,
            chunk_count: 0,
            image_count: 0,
            started_at: now,
            updated_at: now,
        };

        progress(status.clone());
        self.db.upsert_ingest_status(&status).await?;

        let docx_hash = file_sha256(&request.docx_path).await?;
        let pdf_hash = file_sha256(&request.pdf_path).await?;

        if !request.rebuild {
            if let Some(latest) = self.db.latest_manifest().await? {
                if latest.docx_hash == docx_hash && latest.pdf_hash == pdf_hash {
                    status.status = "completed".to_string();
                    status.stage = "skipped_unchanged".to_string();
                    status.message = Some("Source files unchanged; skipped re-ingest.".to_string());
                    status.chunk_count = latest.chunk_count;
                    status.image_count = latest.image_count;
                    status.updated_at = Utc::now();
                    progress(status.clone());
                    self.db.upsert_ingest_status(&status).await?;

                    return Ok(IngestResult {
                        chunk_count: latest.chunk_count,
                        image_count: latest.image_count,
                        skipped: true,
                    });
                }
            }
        }

        status.stage = "extracting_docx".to_string();
        status.updated_at = Utc::now();
        progress(status.clone());
        self.db.upsert_ingest_status(&status).await?;

        let docx_path = Path::new(&request.docx_path).to_path_buf();
        let docx_hash_clone = docx_hash.clone();
        let mut docx_units = tokio::task::spawn_blocking(move || {
            docx::extract_docx_units(&docx_path, &docx_hash_clone)
        })
        .await
        .context("DOCX extraction task panicked")??;

        status.stage = "extracting_pdf".to_string();
        status.updated_at = Utc::now();
        progress(status.clone());
        self.db.upsert_ingest_status(&status).await?;

        let image_subdir =
            self.config
                .image_dir()
                .join(format!("{}-{}", &docx_hash[..8], &pdf_hash[..8]));

        let (mut pdf_units, image_assets) = pdf::extract_pdf_units(
            Path::new(&request.pdf_path),
            &pdf_hash,
            &image_subdir,
            &self.ollama,
            &self.config.models.vision_model,
        )
        .await?;

        status.image_count = image_assets.len() as i64;

        let mut units: Vec<SourceUnit> = Vec::new();
        units.append(&mut docx_units);
        units.append(&mut pdf_units);

        status.stage = "chunking".to_string();
        status.updated_at = Utc::now();
        progress(status.clone());
        self.db.upsert_ingest_status(&status).await?;

        let chunks = build_chunks(
            units,
            self.config.tokens.chunk_target_tokens,
            self.config.tokens.chunk_overlap_tokens,
        );

        if chunks.is_empty() {
            anyhow::bail!("no chunks generated from DOCX/PDF sources");
        }

        status.chunk_count = chunks.len() as i64;

        status.stage = "embedding_and_indexing".to_string();
        status.updated_at = Utc::now();
        progress(status.clone());
        self.db.upsert_ingest_status(&status).await?;

        let mut points = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            let embedding = self
                .ollama
                .embed(&self.config.models.embedding_model, &chunk.content)
                .await
                .with_context(|| format!("failed embedding for chunk {}", chunk.id))?;

            points.push(QdrantPoint {
                id: chunk.id.clone(),
                vector: embedding,
                payload: QdrantPayload {
                    chunk_id: Some(chunk.id.clone()),
                    id: None,
                    kind: chunk.kind.as_str().to_string(),
                    chapter: chunk.chapter.clone(),
                    part: chunk.part.clone(),
                    part_index: chunk.part_index,
                    page: chunk.page,
                },
            });
        }

        if let Some(first) = points.first() {
            self.qdrant.recreate_collection(first.vector.len()).await?;
        }

        let mut start = 0;
        while start < points.len() {
            let end = (start + 64).min(points.len());
            self.qdrant.upsert_points(&points[start..end]).await?;
            start = end;
        }

        let tantivy = self.tantivy.clone();
        let chunks_for_index = chunks.clone();
        tokio::task::spawn_blocking(move || tantivy.rebuild(&chunks_for_index))
            .await
            .context("tantivy rebuild task panicked")??;

        self.db.clear_chunks().await?;
        self.db.insert_chunks(&chunks).await?;
        self.db.clear_image_assets().await?;
        self.db.insert_image_assets(&image_assets).await?;
        self.db.clear_entity_mentions().await?;
        let mentions = build_entity_mentions(&chunks);
        self.db.insert_entity_mentions(&mentions).await?;

        let manifest = IngestManifest {
            docx_hash,
            pdf_hash,
            created_at: Utc::now(),
            chunk_count: chunks.len() as i64,
            image_count: image_assets.len() as i64,
        };

        self.db.record_manifest(&manifest).await?;

        status.status = "completed".to_string();
        status.stage = "done".to_string();
        status.chunk_count = chunks.len() as i64;
        status.image_count = image_assets.len() as i64;
        status.updated_at = Utc::now();
        progress(status.clone());
        self.db.upsert_ingest_status(&status).await?;

        Ok(IngestResult {
            chunk_count: chunks.len() as i64,
            image_count: image_assets.len() as i64,
            skipped: false,
        })
    }
}

fn build_chunks(units: Vec<SourceUnit>, target_tokens: usize, overlap_tokens: usize) -> Vec<Chunk> {
    let mut chunks = Vec::new();

    let mut current_source_hash: Option<String> = None;
    let mut current_part: Option<String> = None;
    let mut current_part_index: Option<i64> = None;

    for mut unit in units {
        if current_source_hash.as_deref() != Some(unit.source_hash.as_str()) {
            current_source_hash = Some(unit.source_hash.clone());
            current_part = None;
            current_part_index = None;
        }

        let normalized = normalize_text(&unit.content);
        if normalized.is_empty() {
            continue;
        }

        let inline_part = detect_part_marker(&normalized);
        if let Some((part, part_index)) = inline_part.clone() {
            current_part = Some(part);
            current_part_index = Some(part_index);
        }
        if unit.part.is_none() {
            unit.part = current_part.clone();
            unit.part_index = current_part_index;
        } else if let Some(part) = unit.part.clone() {
            current_part = Some(part);
            current_part_index = unit.part_index;
        }

        let tokens: Vec<String> = normalized
            .split_whitespace()
            .map(|token| token.to_string())
            .collect();

        let unit_target_tokens =
            if matches!(unit.kind, SourceType::ImageCaption | SourceType::ImageOcr) {
                target_tokens.min(220)
            } else {
                target_tokens
            };
        let unit_overlap_tokens = overlap_tokens.min(unit_target_tokens.saturating_sub(1));
        let unit_step = unit_target_tokens
            .saturating_sub(unit_overlap_tokens)
            .max(1);

        if inline_part.is_some() && tokens.len() <= 6 {
            continue;
        }

        if tokens.len() <= unit_target_tokens {
            chunks.push(Chunk {
                id: Uuid::new_v4().to_string(),
                content: normalized,
                kind: unit.kind,
                chapter: unit.chapter,
                part: unit.part,
                part_index: unit.part_index,
                page: unit.page,
                token_count: tokens.len() as i64,
                source_hash: unit.source_hash,
                image_path: unit.image_path,
            });
            continue;
        }

        let mut start = 0;
        while start < tokens.len() {
            let end = (start + unit_target_tokens).min(tokens.len());
            let content = tokens[start..end].join(" ");

            chunks.push(Chunk {
                id: Uuid::new_v4().to_string(),
                content,
                kind: unit.kind,
                chapter: unit.chapter.clone(),
                part: unit.part.clone(),
                part_index: unit.part_index,
                page: unit.page,
                token_count: (end - start) as i64,
                source_hash: unit.source_hash.clone(),
                image_path: unit.image_path.clone(),
            });

            if end == tokens.len() {
                break;
            }
            start += unit_step;
        }
    }

    chunks
}

fn normalize_text(text: &str) -> String {
    text.replace('\u{00A0}', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

pub(crate) fn detect_part_marker(line: &str) -> Option<(String, i64)> {
    let cleaned = line.replace('\u{200B}', "").trim().to_string();
    if cleaned.is_empty() || cleaned.len() > 80 {
        return None;
    }
    if cleaned.split_whitespace().count() > 8 {
        return None;
    }

    let re = Regex::new(
        r"(?i)^part\s*([0-9]{1,2}|[ivx]{1,5})\s*([ab])?(?:\s*[:\-–—]?\s*[A-Za-z0-9][A-Za-z0-9 '\-]*)?$",
    )
        .unwrap_or_else(|_| Regex::new("^$").unwrap());
    let caps = re.captures(&cleaned)?;

    let number = parse_part_number(caps.get(1)?.as_str())?;
    if !(1..=20).contains(&number) {
        return None;
    }
    let suffix = caps
        .get(2)
        .map(|m| m.as_str().to_ascii_uppercase())
        .unwrap_or_default();
    let label = format!("Part {number}{suffix}");
    let suffix_offset = match suffix.as_str() {
        "A" => 1,
        "B" => 2,
        _ => 0,
    };

    Some((label, number * 10 + suffix_offset))
}

pub(crate) fn detect_part_marker_anywhere(text: &str) -> Option<(String, i64)> {
    text.lines().take(24).find_map(detect_part_marker)
}

fn parse_part_number(raw: &str) -> Option<i64> {
    if let Ok(v) = raw.parse::<i64>() {
        return Some(v);
    }

    roman_to_int(raw)
}

fn roman_to_int(raw: &str) -> Option<i64> {
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

fn build_entity_mentions(chunks: &[Chunk]) -> Vec<EntityMention> {
    let token_re = Regex::new(r"[A-Za-z0-9_]+").unwrap_or_else(|_| Regex::new("^$").unwrap());
    let mut mentions: Vec<EntityMention> = Vec::new();

    for chunk in chunks {
        let mut per_entity: std::collections::HashMap<String, i64> =
            std::collections::HashMap::new();
        for mat in token_re.find_iter(&chunk.content) {
            let token = mat.as_str();
            if !is_entity_token(token) {
                continue;
            }
            let normalized = token.to_ascii_lowercase();
            *per_entity.entry(normalized).or_insert(0) += 1;
        }

        for (entity, count) in per_entity {
            mentions.push(EntityMention {
                entity,
                chunk_id: chunk.id.clone(),
                mentions: count,
                part_index: chunk.part_index,
            });
        }
    }

    mentions
}

fn is_entity_token(token: &str) -> bool {
    if token.len() < 4 {
        return false;
    }
    let has_digit = token.chars().any(|c| c.is_ascii_digit());
    let has_internal_upper = token.chars().skip(1).any(|c| c.is_ascii_uppercase());
    let has_underscore = token.contains('_');
    has_digit || has_internal_upper || has_underscore
}

async fn file_sha256(path: &str) -> Result<String> {
    let bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("failed reading file for hash: {}", path))?;

    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunking_splits_long_text_with_overlap() {
        let content = (1..=120)
            .map(|n| format!("word{n}"))
            .collect::<Vec<_>>()
            .join(" ");
        let units = vec![SourceUnit {
            kind: SourceType::DocxText,
            chapter: Some("Chapter 1".to_string()),
            part: Some("Part 1".to_string()),
            part_index: Some(10),
            page: None,
            content,
            source_hash: "h".to_string(),
            image_path: None,
        }];

        let chunks = build_chunks(units, 50, 10);
        assert!(chunks.len() >= 3);
        assert_eq!(chunks[0].chapter.as_deref(), Some("Chapter 1"));
        assert_eq!(chunks[0].part_index, Some(10));
        assert!(chunks.iter().all(|c| c.token_count > 0));
    }

    #[test]
    fn image_units_stay_single_chunk() {
        let units = vec![SourceUnit {
            kind: SourceType::ImageCaption,
            chapter: None,
            part: None,
            part_index: None,
            page: Some(3),
            content: "A single image caption chunk".to_string(),
            source_hash: "h".to_string(),
            image_path: Some("/tmp/image.png".to_string()),
        }];

        let chunks = build_chunks(units, 10, 2);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, SourceType::ImageCaption);
        assert_eq!(chunks[0].page, Some(3));
    }

    #[test]
    fn detects_part_labels() {
        let part = detect_part_marker("Part 7a: Return Arc").expect("part");
        assert_eq!(part.0, "Part 7A");
        assert_eq!(part.1, 71);
    }

    #[test]
    fn detects_part_label_variants() {
        let part = detect_part_marker("Part IV Prelude").expect("roman part");
        assert_eq!(part.0, "Part 4");
        assert_eq!(part.1, 40);

        let part = detect_part_marker("Part 5 Finale").expect("subtitle part");
        assert_eq!(part.0, "Part 5");
        assert_eq!(part.1, 50);
    }

    #[test]
    fn rejects_narrative_part_reference() {
        assert!(
            detect_part_marker("Part 1 was, by far, the most chaotic part of the game.").is_none()
        );
    }

    #[test]
    fn part_state_resets_when_source_changes() {
        let units = vec![
            SourceUnit {
                kind: SourceType::DocxText,
                chapter: None,
                part: Some("Part 7B".to_string()),
                part_index: Some(72),
                page: None,
                content: "Late docx section".to_string(),
                source_hash: "docx_hash".to_string(),
                image_path: None,
            },
            SourceUnit {
                kind: SourceType::PdfText,
                chapter: None,
                part: None,
                part_index: None,
                page: Some(1),
                content: "Early pdf page".to_string(),
                source_hash: "pdf_hash".to_string(),
                image_path: None,
            },
        ];

        let chunks = build_chunks(units, 64, 8);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].part.as_deref(), Some("Part 7B"));
        assert_eq!(chunks[1].part, None);
    }

    #[test]
    fn builds_entity_mentions_for_named_tokens() {
        let chunks = vec![Chunk {
            id: "c1".to_string(),
            content: "TheLongIslander met AggravatedCow and Apollo345".to_string(),
            kind: SourceType::DocxText,
            chapter: None,
            part: Some("Part 1".to_string()),
            part_index: Some(10),
            page: None,
            token_count: 7,
            source_hash: "h".to_string(),
            image_path: None,
        }];

        let mentions = build_entity_mentions(&chunks);
        assert!(mentions.iter().any(|m| m.entity == "thelongislander"));
        assert!(mentions.iter().any(|m| m.entity == "apollo345"));
    }
}
