use std::sync::Arc;

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct VectorHit {
    pub chunk_id: String,
    pub score: f32,
}

#[derive(Clone)]
pub struct QdrantStore {
    client: Client,
    base_url: String,
    collection: String,
    known_vector_size: Arc<RwLock<Option<usize>>>,
}

impl QdrantStore {
    pub fn new(base_url: impl Into<String>, collection: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            collection: collection.into(),
            known_vector_size: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn recreate_collection(&self, vector_size: usize) -> Result<()> {
        let delete_url = format!("{}/collections/{}", self.base_url, self.collection);
        let _ = self.client.delete(&delete_url).send().await;

        self.ensure_collection(vector_size).await?;
        Ok(())
    }

    pub async fn ensure_collection(&self, vector_size: usize) -> Result<()> {
        {
            let known = self.known_vector_size.read().await;
            if let Some(existing) = *known {
                if existing == vector_size {
                    return Ok(());
                }
            }
        }

        let create_url = format!("{}/collections/{}", self.base_url, self.collection);
        let payload = json!({
            "vectors": {
                "size": vector_size,
                "distance": "Cosine"
            }
        });

        self.client
            .put(create_url)
            .json(&payload)
            .send()
            .await
            .context("failed to contact qdrant while creating collection")?
            .error_for_status()
            .context("qdrant failed to create collection")?;

        *self.known_vector_size.write().await = Some(vector_size);
        Ok(())
    }

    pub async fn upsert_points(&self, points: &[QdrantPoint]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        let vector_size = points[0].vector.len();
        self.ensure_collection(vector_size).await?;

        let upsert_url = format!(
            "{}/collections/{}/points?wait=true",
            self.base_url, self.collection
        );
        let body = json!({ "points": points });

        self.client
            .put(upsert_url)
            .json(&body)
            .send()
            .await
            .context("failed to contact qdrant during upsert")?
            .error_for_status()
            .context("qdrant upsert returned non-success status")?;

        Ok(())
    }

    pub async fn search(&self, vector: &[f32], limit: usize) -> Result<Vec<VectorHit>> {
        if vector.is_empty() {
            return Ok(vec![]);
        }

        let url = format!(
            "{}/collections/{}/points/search",
            self.base_url, self.collection
        );

        let body = json!({
            "vector": vector,
            "limit": limit,
            "with_payload": true,
        });

        let response = self
            .client
            .post(url)
            .json(&body)
            .send()
            .await
            .context("failed to contact qdrant during search")?
            .error_for_status()
            .context("qdrant search returned non-success status")?
            .json::<QdrantSearchResponse>()
            .await
            .context("failed to decode qdrant search response")?;

        Ok(response
            .result
            .into_iter()
            .filter_map(|point| {
                let payload = point.payload?;
                let chunk_id = payload.chunk_id.or_else(|| payload.id)?;
                Some(VectorHit {
                    chunk_id,
                    score: point.score,
                })
            })
            .collect())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct QdrantPoint {
    pub id: String,
    pub vector: Vec<f32>,
    pub payload: QdrantPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantPayload {
    pub chunk_id: Option<String>,
    pub id: Option<String>,
    pub kind: String,
    pub chapter: Option<String>,
    pub part: Option<String>,
    pub part_index: Option<i64>,
    pub page: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct QdrantSearchResponse {
    result: Vec<QdrantResultPoint>,
}

#[derive(Debug, Deserialize)]
struct QdrantResultPoint {
    score: f32,
    payload: Option<QdrantPayload>,
}
