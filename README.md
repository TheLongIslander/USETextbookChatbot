# Offline Multimodal Novel Chatbot (Rust + Ollama)

Local-first RAG chatbot for a novel exported from Google Docs as `DOCX + PDF`.

## What this includes

- Rust `axum` server bound to localhost by default.
- Local web UI for ingest + chat.
- Ingestion pipeline:
  - DOCX parsing with heading-aware chapter context.
  - PDF text extraction.
  - PDF image extraction, OCR (`tesseract`), and local image captioning via Ollama vision model.
- Hybrid retrieval:
  - Vector search in Qdrant.
  - BM25 search in Tantivy.
  - Reciprocal rank fusion.
- Strict citation-mode answers with source metadata.
- SQLite metadata store for chunks, manifests, sessions, messages, and image assets.
- CLI ingestion command (`cargo run --bin ingest -- ...`).

## Requirements

- macOS (Apple Silicon supported).
- Rust toolchain (`rustup`, `cargo`).
- [Ollama](https://ollama.com/) running locally.
- Qdrant running locally (recommended via `docker compose`).

Optional but recommended for better PDF ingestion:

- `poppler` utilities (`pdfinfo`, `pdftotext`, `pdfimages`).
- `tesseract` for OCR.

Example install on macOS:

```bash
brew install poppler tesseract
```

## Quick start

1. Copy environment defaults:

```bash
cp .env.example .env
```

2. Start Qdrant:

```bash
docker compose up -d
```

3. Ensure Ollama is running and pull models:

```bash
ollama pull qwen2.5:14b-instruct
ollama pull mxbai-embed-large
ollama pull llava:7b
```

4. Run the server:

```bash
cargo run
```

5. Open `http://127.0.0.1:8080`.

## Ingest from CLI

```bash
cargo run --bin ingest -- \
  --docx "./Full USE Textbook-2.docx" \
  --pdf "./Full USE Textbook.pdf"
```

Add `--rebuild` to force re-index even if source hashes are unchanged.

## API

- `POST /api/ingest`
  - body: `{ "docx_path": "...", "pdf_path": "...", "rebuild": false }`
- `GET /api/ingest/:job_id`
- `POST /api/chat`
  - body: `{ "session_id": "...", "question": "...", "strict": true }`
- `GET /api/sources/:chunk_id`
- `POST /api/session`
  - create: `{ "reset": false }`
  - reset existing: `{ "session_id": "...", "reset": true }`

## Notes on strict mode

In strict mode, the assistant is expected to answer only from indexed novel sources and include chunk citations. If insufficient evidence exists, it returns:

`Not found in indexed novel sources.`

## Data layout

By default data is written under `./data`:

- `chatbot.sqlite3` (metadata + sessions)
- `tantivy/` (BM25 index)
- `images/` (extracted PDF images)
- `qdrant/` (if using provided docker compose)

## Dev checks

```bash
cargo check
cargo test
```
