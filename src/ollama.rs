use anyhow::{Context, Result};
use base64::Engine;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct OllamaClient {
    client: Client,
    base_url: String,
}

impl OllamaClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
        }
    }

    pub async fn embed(&self, model: &str, text: &str) -> Result<Vec<f32>> {
        let input = text.trim();
        if input.is_empty() {
            anyhow::bail!("cannot embed empty text input");
        }

        match self.embed_with_endpoint_fallback(model, input).await {
            Ok(vector) => Ok(vector),
            Err(err) => {
                if !is_context_length_error(&err) {
                    return Err(err);
                }

                let word_count = input.split_whitespace().count();
                let mut last_err = err;
                for max_words in [1400usize, 1000, 800, 600, 450, 320, 240, 180, 120] {
                    if word_count <= max_words {
                        continue;
                    }

                    let truncated = truncate_to_word_limit(input, max_words);
                    match self.embed_with_endpoint_fallback(model, &truncated).await {
                        Ok(vector) => return Ok(vector),
                        Err(next_err) => {
                            if !is_context_length_error(&next_err) {
                                return Err(next_err);
                            }
                            last_err = next_err;
                        }
                    }
                }

                Err(anyhow::anyhow!(
                    "ollama embedding exceeded context length even after adaptive truncation \
                     (original_words={word_count}). last error: {last_err}"
                ))
            }
        }
    }

    async fn embed_with_endpoint_fallback(&self, model: &str, text: &str) -> Result<Vec<f32>> {
        // Newer Ollama releases use /api/embed, while older versions use /api/embeddings.
        // Try the new route first and fall back to the legacy route for compatibility.
        match self.embed_modern(model, text).await {
            Ok(vector) => Ok(vector),
            Err(modern_err) => match self.embed_legacy(model, text).await {
                Ok(vector) => Ok(vector),
                Err(legacy_err) => Err(anyhow::anyhow!(
                    "ollama embedding failed via /api/embed and /api/embeddings. \
                     modern error: {modern_err}; legacy error: {legacy_err}; \
                     ensure the embedding model is pulled (e.g. `ollama pull {model}`)"
                )),
            },
        }
    }

    async fn embed_modern(&self, model: &str, text: &str) -> Result<Vec<f32>> {
        #[derive(Serialize)]
        struct EmbedReq<'a> {
            model: &'a str,
            input: &'a str,
        }

        #[derive(Deserialize)]
        struct EmbedResp {
            embeddings: Vec<Vec<f32>>,
        }

        let url = format!("{}/api/embed", self.base_url);
        let response = self
            .client
            .post(url)
            .json(&EmbedReq { model, input: text })
            .send()
            .await
            .context("failed to call ollama embed endpoint")?;

        if response.status() != StatusCode::OK {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "ollama /api/embed returned {status}: {}",
                normalize_err_body(&body)
            );
        }

        let response = response
            .json::<EmbedResp>()
            .await
            .context("failed to decode ollama /api/embed response")?;

        let vector =
            response.embeddings.into_iter().next().ok_or_else(|| {
                anyhow::anyhow!("ollama /api/embed returned empty embeddings array")
            })?;

        Ok(vector)
    }

    async fn embed_legacy(&self, model: &str, text: &str) -> Result<Vec<f32>> {
        #[derive(Serialize)]
        struct EmbeddingReq<'a> {
            model: &'a str,
            prompt: &'a str,
        }

        #[derive(Deserialize)]
        struct EmbeddingResp {
            embedding: Vec<f32>,
        }

        let url = format!("{}/api/embeddings", self.base_url);
        let response = self
            .client
            .post(url)
            .json(&EmbeddingReq {
                model,
                prompt: text,
            })
            .send()
            .await
            .context("failed to call ollama embeddings endpoint")?;

        if response.status() != StatusCode::OK {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "ollama /api/embeddings returned {status}: {}",
                normalize_err_body(&body)
            );
        }

        let response = response
            .json::<EmbeddingResp>()
            .await
            .context("failed to decode ollama embeddings response")?;

        Ok(response.embedding)
    }

    pub async fn generate_text(
        &self,
        model: &str,
        prompt: &str,
        num_predict: usize,
        temperature: f32,
    ) -> Result<String> {
        #[derive(Serialize)]
        struct GenerateReq<'a> {
            model: &'a str,
            prompt: &'a str,
            stream: bool,
            options: GenerateOptions,
        }

        #[derive(Serialize)]
        struct GenerateOptions {
            num_predict: usize,
            temperature: f32,
        }

        #[derive(Deserialize)]
        struct GenerateResp {
            response: String,
        }

        let url = format!("{}/api/generate", self.base_url);
        let response = self
            .client
            .post(url)
            .json(&GenerateReq {
                model,
                prompt,
                stream: false,
                options: GenerateOptions {
                    num_predict,
                    temperature,
                },
            })
            .send()
            .await
            .context("failed to call ollama generate endpoint")?
            .error_for_status()
            .context("ollama generate returned non-success status")?
            .json::<GenerateResp>()
            .await
            .context("failed to decode ollama generate response")?;

        Ok(response.response.trim().to_string())
    }

    pub async fn generate_vision_caption(
        &self,
        model: &str,
        prompt: &str,
        image_bytes: &[u8],
    ) -> Result<String> {
        #[derive(Serialize)]
        struct VisionReq<'a> {
            model: &'a str,
            prompt: &'a str,
            stream: bool,
            images: Vec<String>,
            options: VisionOptions,
        }

        #[derive(Serialize)]
        struct VisionOptions {
            num_predict: usize,
            temperature: f32,
        }

        #[derive(Deserialize)]
        struct VisionResp {
            response: String,
        }

        let encoded = base64::engine::general_purpose::STANDARD.encode(image_bytes);

        let url = format!("{}/api/generate", self.base_url);
        let response = self
            .client
            .post(url)
            .json(&VisionReq {
                model,
                prompt,
                stream: false,
                images: vec![encoded],
                options: VisionOptions {
                    num_predict: 180,
                    temperature: 0.1,
                },
            })
            .send()
            .await
            .context("failed to call ollama vision generate endpoint")?
            .error_for_status()
            .context("ollama vision generate returned non-success status")?
            .json::<VisionResp>()
            .await
            .context("failed to decode ollama vision response")?;

        Ok(response.response.trim().to_string())
    }
}

fn normalize_err_body(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return "<empty body>".to_string();
    }

    if let Ok(json) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(err) = json.get("error").and_then(|v| v.as_str()) {
            return err.to_string();
        }
    }

    trimmed.to_string()
}

fn is_context_length_error(err: &anyhow::Error) -> bool {
    let msg = err.to_string().to_ascii_lowercase();
    msg.contains("input length exceeds the context length")
        || (msg.contains("context length") && msg.contains("input length"))
}

fn truncate_to_word_limit(text: &str, max_words: usize) -> String {
    text.split_whitespace()
        .take(max_words)
        .collect::<Vec<_>>()
        .join(" ")
}
