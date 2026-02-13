use std::env;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub answer_model: String,
    pub embedding_model: String,
    pub vision_model: String,
}

#[derive(Clone, Debug)]
pub struct TokenConfig {
    pub max_context_tokens: usize,
    pub max_output_tokens: usize,
    pub chunk_target_tokens: usize,
    pub chunk_overlap_tokens: usize,
}

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub bind_addr: String,
    pub data_dir: PathBuf,
    pub ollama_base_url: String,
    pub qdrant_base_url: String,
    pub qdrant_collection: String,
    pub models: ModelConfig,
    pub tokens: TokenConfig,
}

impl AppConfig {
    pub fn from_env() -> Self {
        let data_dir = env::var("NOVEL_CHATBOT_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./data"));

        Self {
            bind_addr: env::var("NOVEL_CHATBOT_BIND")
                .unwrap_or_else(|_| "127.0.0.1:8080".to_string()),
            data_dir,
            ollama_base_url: env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string()),
            qdrant_base_url: env::var("QDRANT_BASE_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:6333".to_string()),
            qdrant_collection: env::var("QDRANT_COLLECTION")
                .unwrap_or_else(|_| "novel_chunks".to_string()),
            models: ModelConfig {
                answer_model: env::var("ANSWER_MODEL")
                    .unwrap_or_else(|_| "qwen2.5:14b-instruct".to_string()),
                embedding_model: env::var("EMBEDDING_MODEL")
                    .unwrap_or_else(|_| "mxbai-embed-large".to_string()),
                vision_model: env::var("VISION_MODEL").unwrap_or_else(|_| "llava:7b".to_string()),
            },
            tokens: TokenConfig {
                max_context_tokens: env::var("MAX_CONTEXT_TOKENS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(4_000),
                max_output_tokens: env::var("MAX_OUTPUT_TOKENS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(500),
                chunk_target_tokens: env::var("CHUNK_TARGET_TOKENS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(600),
                chunk_overlap_tokens: env::var("CHUNK_OVERLAP_TOKENS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(80),
            },
        }
    }

    pub fn tantivy_dir(&self) -> PathBuf {
        self.data_dir.join("tantivy")
    }

    pub fn image_dir(&self) -> PathBuf {
        self.data_dir.join("images")
    }

    pub fn sqlite_dsn(&self) -> String {
        format!(
            "sqlite://{}",
            self.data_dir.join("chatbot.sqlite3").display()
        )
    }
}
