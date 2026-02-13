use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use askama::Template;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use chrono::Utc;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use uuid::Uuid;

use crate::chat::ChatService;
use crate::config::AppConfig;
use crate::db::Database;
use crate::ingest::{IngestResult, Ingestor};
use crate::models::{
    ChatRequest, IngestRequest, IngestResponse, IngestStatus, SessionRequest, SessionResponse,
};

#[derive(Clone)]
struct AppState {
    db: Database,
    chat: ChatService,
    ingestor: Ingestor,
    jobs: Arc<Mutex<HashMap<String, IngestStatus>>>,
}

pub async fn run_server(
    config: AppConfig,
    db: Database,
    chat_service: ChatService,
    ingestor: Ingestor,
) -> Result<()> {
    tokio::fs::create_dir_all(&config.data_dir).await?;

    let state = AppState {
        db,
        chat: chat_service,
        ingestor,
        jobs: Arc::new(Mutex::new(HashMap::new())),
    };

    let app = Router::new()
        .route("/", get(index_page))
        .route("/api/ingest", post(start_ingest))
        .route("/api/ingest/:job_id", get(get_ingest_status))
        .route("/api/chat", post(chat_handler))
        .route("/api/sources/:chunk_id", get(get_source))
        .route("/api/session", post(create_session))
        .nest_service("/static", ServeDir::new("static"))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = config.bind_addr.parse()?;
    tracing::info!("listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index_page(State(state): State<AppState>) -> Result<Html<String>, ApiError> {
    let session_id = state.db.create_session().await.map_err(ApiError::from)?;

    let template = IndexTemplate { session_id };
    let body = template.render().map_err(ApiError::from)?;

    Ok(Html(body))
}

async fn start_ingest(
    State(state): State<AppState>,
    Json(request): Json<IngestRequest>,
) -> Result<Json<IngestResponse>, ApiError> {
    let job_id = Uuid::new_v4().to_string();
    let now = Utc::now();

    let initial = IngestStatus {
        job_id: job_id.clone(),
        status: "started".to_string(),
        stage: "queued".to_string(),
        message: None,
        chunk_count: 0,
        image_count: 0,
        started_at: now,
        updated_at: now,
    };

    {
        let mut jobs = state
            .jobs
            .lock()
            .map_err(|_| ApiError::from(anyhow::anyhow!("lock poisoned")))?;
        jobs.insert(job_id.clone(), initial.clone());
    }
    state.db.upsert_ingest_status(&initial).await?;

    let state_for_task = state.clone();
    let job_id_for_task = job_id.clone();
    tokio::spawn(async move {
        let jobs = state_for_task.jobs.clone();

        let callback = |status: IngestStatus| {
            if let Ok(mut guard) = jobs.lock() {
                guard.insert(status.job_id.clone(), status);
            }
        };

        let result: Result<IngestResult> = state_for_task
            .ingestor
            .ingest(&job_id_for_task, request, callback)
            .await;

        if let Err(err) = result {
            let failed_status = IngestStatus {
                job_id: job_id_for_task.clone(),
                status: "failed".to_string(),
                stage: "error".to_string(),
                message: Some(err.to_string()),
                chunk_count: 0,
                image_count: 0,
                started_at: now,
                updated_at: Utc::now(),
            };

            if let Ok(mut guard) = state_for_task.jobs.lock() {
                guard.insert(job_id_for_task.clone(), failed_status.clone());
            }
            let _ = state_for_task.db.upsert_ingest_status(&failed_status).await;
            tracing::error!("ingest job {} failed: {}", job_id_for_task, err);
        }
    });

    Ok(Json(IngestResponse {
        job_id,
        status: "started".to_string(),
    }))
}

async fn get_ingest_status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<IngestStatus>, ApiError> {
    if let Some(status) = state
        .jobs
        .lock()
        .map_err(|_| ApiError::from(anyhow::anyhow!("lock poisoned")))?
        .get(&job_id)
        .cloned()
    {
        return Ok(Json(status));
    }

    let status = state.db.get_ingest_status(&job_id).await?;
    match status {
        Some(status) => Ok(Json(status)),
        None => Err(ApiError::not_found(format!(
            "ingest job not found: {}",
            job_id
        ))),
    }
}

async fn chat_handler(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<crate::models::ChatAnswer>, ApiError> {
    let answer = state.chat.answer(request).await?;
    Ok(Json(answer))
}

async fn get_source(
    State(state): State<AppState>,
    Path(chunk_id): Path<String>,
) -> Result<Json<crate::models::Chunk>, ApiError> {
    let chunk = state.db.get_chunk(&chunk_id).await?;
    match chunk {
        Some(chunk) => Ok(Json(chunk)),
        None => Err(ApiError::not_found(format!(
            "chunk not found: {}",
            chunk_id
        ))),
    }
}

async fn create_session(
    State(state): State<AppState>,
    Json(request): Json<SessionRequest>,
) -> Result<Json<SessionResponse>, ApiError> {
    if request.reset.unwrap_or(false) {
        if let Some(session_id) = request.session_id {
            state.db.ensure_session(&session_id).await?;
            state.db.delete_session_messages(&session_id).await?;
            return Ok(Json(SessionResponse { session_id }));
        }
    }

    let session_id = state.db.create_session().await?;
    Ok(Json(SessionResponse { session_id }))
}

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {
    session_id: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn not_found(message: String) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message,
        }
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(value: anyhow::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: value.to_string(),
        }
    }
}

impl From<askama::Error> for ApiError {
    fn from(value: askama::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: value.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let body = serde_json::json!({ "error": self.message });
        (self.status, Json(body)).into_response()
    }
}
