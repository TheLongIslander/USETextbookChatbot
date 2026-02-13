use std::cmp::Ordering;
use std::collections::HashMap;

use anyhow::Result;

use crate::db::Database;
use crate::models::RetrievalResult;
use crate::ollama::OllamaClient;
use crate::qdrant_store::QdrantStore;
use crate::tantivy_store::TantivyStore;

const VECTOR_RECALL_K: usize = 72;
const BM25_RECALL_K: usize = 72;

#[derive(Clone)]
pub struct Retriever {
    db: Database,
    qdrant: QdrantStore,
    tantivy: TantivyStore,
    ollama: OllamaClient,
    embedding_model: String,
}

impl Retriever {
    pub fn new(
        db: Database,
        qdrant: QdrantStore,
        tantivy: TantivyStore,
        ollama: OllamaClient,
        embedding_model: impl Into<String>,
    ) -> Self {
        Self {
            db,
            qdrant,
            tantivy,
            ollama,
            embedding_model: embedding_model.into(),
        }
    }

    pub async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<RetrievalResult>> {
        let embedding = self.ollama.embed(&self.embedding_model, query).await?;
        let recall_k = top_k.saturating_mul(3).max(VECTOR_RECALL_K);
        let qdrant_fut = self.qdrant.search(&embedding, recall_k);

        let tantivy = self.tantivy.clone();
        let query_text = query.to_string();
        let bm25_recall_k = top_k.saturating_mul(3).max(BM25_RECALL_K);
        let bm25_fut =
            tokio::task::spawn_blocking(move || tantivy.search(&query_text, bm25_recall_k));

        let (vector_hits, bm25_hits) = tokio::join!(qdrant_fut, bm25_fut);

        let vector_hits = vector_hits.unwrap_or_default();
        let bm25_hits = bm25_hits.unwrap_or_else(|_| Ok(vec![])).unwrap_or_default();

        let mut fused_scores: HashMap<String, f32> = HashMap::new();
        let rrf_k = 60.0f32;
        let (vector_weight, bm25_weight) = fusion_weights_for_query(query);

        for (rank, hit) in vector_hits.iter().enumerate() {
            let rank_score = vector_weight / (rrf_k + (rank + 1) as f32);
            let similarity_bonus = hit.score.clamp(0.0, 1.0) * 0.04;
            let score = rank_score + similarity_bonus;
            *fused_scores.entry(hit.chunk_id.clone()).or_insert(0.0) += score;
        }

        for (rank, (chunk_id, raw_score)) in bm25_hits.iter().enumerate() {
            let rank_score = bm25_weight / (rrf_k + (rank + 1) as f32);
            let lexical_bonus = raw_score.max(0.0).ln_1p() * 0.012;
            let score = rank_score + lexical_bonus;
            *fused_scores.entry(chunk_id.clone()).or_insert(0.0) += score;
        }

        let mut ranked: Vec<(String, f32)> = fused_scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let selected_ids: Vec<String> = ranked
            .iter()
            .take(top_k)
            .map(|(chunk_id, _)| chunk_id.clone())
            .collect();

        let chunks = self.db.get_chunks_by_ids(&selected_ids).await?;
        let score_map: HashMap<&str, f32> = ranked
            .iter()
            .map(|(chunk_id, score)| (chunk_id.as_str(), *score))
            .collect();

        let mut out = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            out.push(RetrievalResult {
                score: *score_map.get(chunk.id.as_str()).unwrap_or(&0.0),
                chunk,
            });
        }

        out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        Ok(out)
    }
}

fn fusion_weights_for_query(query: &str) -> (f32, f32) {
    let lower = query.to_ascii_lowercase();
    let lexical_intent = [
        "when ",
        "where ",
        "which part",
        "what part",
        "date",
        "year",
        "arrive",
        "arrival",
        "reached",
        "went",
        "go to",
        "in ",
        "to ",
        "chapter",
        "part ",
        "killed",
        "died",
        "who is",
    ]
    .iter()
    .any(|term| lower.contains(term));

    let semantic_intent = ["theme", "motif", "symbol", "tone", "meaning", "message"]
        .iter()
        .any(|term| lower.contains(term));

    let entity_signal = query.chars().any(|c| c.is_ascii_digit())
        || query
            .split_whitespace()
            .any(|tok| tok.chars().skip(1).any(|c| c.is_ascii_uppercase()));

    if semantic_intent {
        (1.2, 0.95)
    } else if lexical_intent || entity_signal {
        (0.95, 1.35)
    } else {
        (1.0, 1.0)
    }
}
