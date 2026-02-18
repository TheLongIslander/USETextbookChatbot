use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    DocxText,
    PdfText,
    ImageCaption,
    ImageOcr,
}

impl SourceType {
    pub fn as_str(self) -> &'static str {
        match self {
            SourceType::DocxText => "docx_text",
            SourceType::PdfText => "pdf_text",
            SourceType::ImageCaption => "image_caption",
            SourceType::ImageOcr => "image_ocr",
        }
    }

    pub fn from_db(value: &str) -> Self {
        match value {
            "docx_text" => SourceType::DocxText,
            "pdf_text" => SourceType::PdfText,
            "image_caption" => SourceType::ImageCaption,
            "image_ocr" => SourceType::ImageOcr,
            _ => SourceType::PdfText,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub content: String,
    pub kind: SourceType,
    pub chapter: Option<String>,
    pub part: Option<String>,
    pub part_index: Option<i64>,
    pub page: Option<i64>,
    pub token_count: i64,
    pub source_hash: String,
    pub image_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageAsset {
    pub id: String,
    pub page: Option<i64>,
    pub file_path: String,
    pub ocr_text: String,
    pub caption: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub source_id: String,
    pub chunk_id: String,
    pub source_type: SourceType,
    pub chapter: Option<String>,
    pub page: Option<i64>,
    pub snippet: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerMode {
    TextOnly,
    VisionEscalated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatAnswer {
    pub answer_markdown: String,
    pub citations: Vec<Citation>,
    pub mode: AnswerMode,
    pub confidence: f32,
    pub latency_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestManifest {
    pub docx_hash: String,
    pub pdf_hash: String,
    pub created_at: DateTime<Utc>,
    pub chunk_count: i64,
    pub image_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryClass {
    TextOnly,
    ImageRelevant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestRequest {
    pub docx_path: String,
    pub pdf_path: String,
    #[serde(default)]
    pub rebuild: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResponse {
    pub job_id: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestStatus {
    pub job_id: String,
    pub status: String,
    pub stage: String,
    pub message: Option<String>,
    pub chunk_count: i64,
    pub image_count: i64,
    pub started_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub session_id: String,
    pub question: String,
    #[serde(default = "default_true")]
    pub strict: bool,
    #[serde(default = "default_true")]
    pub verbose: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRequest {
    pub session_id: Option<String>,
    pub reset: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionResponse {
    pub session_id: String,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub chunk: Chunk,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct SourceUnit {
    pub kind: SourceType,
    pub chapter: Option<String>,
    pub part: Option<String>,
    pub part_index: Option<i64>,
    pub page: Option<i64>,
    pub content: String,
    pub source_hash: String,
    pub image_path: Option<String>,
}
