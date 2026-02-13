use std::sync::Arc;

use anyhow::Result;
use tokio::sync::Semaphore;
use tracing_subscriber::EnvFilter;

use chatbot::chat::ChatService;
use chatbot::db::Database;
use chatbot::ingest::Ingestor;
use chatbot::ollama::OllamaClient;
use chatbot::qdrant_store::QdrantStore;
use chatbot::retrieval::Retriever;
use chatbot::tantivy_store::TantivyStore;
use chatbot::{run_server, AppConfig};

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let config = AppConfig::from_env();
    tokio::fs::create_dir_all(&config.data_dir).await?;

    let db = Database::new(&config).await?;
    let ollama = OllamaClient::new(config.ollama_base_url.clone());
    let qdrant = QdrantStore::new(
        config.qdrant_base_url.clone(),
        config.qdrant_collection.clone(),
    );
    let tantivy = TantivyStore::new(config.tantivy_dir());

    let retriever = Retriever::new(
        db.clone(),
        qdrant.clone(),
        tantivy.clone(),
        ollama.clone(),
        config.models.embedding_model.clone(),
    );

    let generation_limit = Arc::new(Semaphore::new(1));

    let chat = ChatService::new(
        config.clone(),
        db.clone(),
        ollama.clone(),
        retriever,
        generation_limit,
    );

    let ingestor = Ingestor::new(config.clone(), db.clone(), ollama, qdrant, tantivy);

    run_server(config, db, chat, ingestor).await
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}
