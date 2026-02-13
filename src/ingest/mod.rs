pub mod docx;
pub mod pdf;

use std::path::Path;

use anyhow::{Context, Result};
use chrono::Utc;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::config::AppConfig;
use crate::db::Database;
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
    let step = target_tokens.saturating_sub(overlap_tokens).max(1);

    for unit in units {
        let normalized = normalize_text(&unit.content);
        if normalized.is_empty() {
            continue;
        }

        let tokens: Vec<String> = normalized
            .split_whitespace()
            .map(|token| token.to_string())
            .collect();

        let is_image_unit = matches!(unit.kind, SourceType::ImageCaption | SourceType::ImageOcr);
        if is_image_unit || tokens.len() <= target_tokens {
            chunks.push(Chunk {
                id: Uuid::new_v4().to_string(),
                content: normalized,
                kind: unit.kind,
                chapter: unit.chapter,
                page: unit.page,
                token_count: tokens.len() as i64,
                source_hash: unit.source_hash,
                image_path: unit.image_path,
            });
            continue;
        }

        let mut start = 0;
        while start < tokens.len() {
            let end = (start + target_tokens).min(tokens.len());
            let content = tokens[start..end].join(" ");

            chunks.push(Chunk {
                id: Uuid::new_v4().to_string(),
                content,
                kind: unit.kind,
                chapter: unit.chapter.clone(),
                page: unit.page,
                token_count: (end - start) as i64,
                source_hash: unit.source_hash.clone(),
                image_path: unit.image_path.clone(),
            });

            if end == tokens.len() {
                break;
            }
            start += step;
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
            page: None,
            content,
            source_hash: "h".to_string(),
            image_path: None,
        }];

        let chunks = build_chunks(units, 50, 10);
        assert!(chunks.len() >= 3);
        assert_eq!(chunks[0].chapter.as_deref(), Some("Chapter 1"));
        assert!(chunks.iter().all(|c| c.token_count > 0));
    }

    #[test]
    fn image_units_stay_single_chunk() {
        let units = vec![SourceUnit {
            kind: SourceType::ImageCaption,
            chapter: None,
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
}
