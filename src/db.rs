use std::collections::HashMap;
use std::str::FromStr;

use anyhow::Result;
use chrono::Utc;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions, SqliteRow};
use sqlx::{QueryBuilder, Row, Sqlite, SqlitePool, Transaction};
use uuid::Uuid;

use crate::config::AppConfig;
use crate::models::{Chunk, IngestManifest, IngestStatus, SourceType};

#[derive(Clone)]
pub struct Database {
    pool: SqlitePool,
}

impl Database {
    pub async fn new(config: &AppConfig) -> Result<Self> {
        tokio::fs::create_dir_all(&config.data_dir).await?;

        let options = SqliteConnectOptions::from_str(&config.sqlite_dsn())?
            .create_if_missing(true)
            .foreign_keys(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(10)
            .connect_with(options)
            .await?;

        let db = Self { pool };
        db.migrate().await?;
        Ok(db)
    }

    async fn migrate(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                kind TEXT NOT NULL,
                chapter TEXT,
                page INTEGER,
                token_count INTEGER NOT NULL,
                source_hash TEXT NOT NULL,
                image_path TEXT
            );

            CREATE TABLE IF NOT EXISTS manifests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                docx_hash TEXT NOT NULL,
                pdf_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                image_count INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS image_assets (
                id TEXT PRIMARY KEY,
                page INTEGER,
                file_path TEXT NOT NULL,
                ocr_text TEXT NOT NULL,
                caption TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS ingest_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                message TEXT,
                chunk_count INTEGER NOT NULL,
                image_count INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn clear_chunks(&self) -> Result<()> {
        sqlx::query("DELETE FROM chunks")
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn insert_chunks(&self, chunks: &[Chunk]) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        for chunk in chunks {
            insert_chunk_tx(&mut tx, chunk).await?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn get_chunk(&self, chunk_id: &str) -> Result<Option<Chunk>> {
        let row = sqlx::query(
            r#"
            SELECT id, content, kind, chapter, page, token_count, source_hash, image_path
            FROM chunks
            WHERE id = ?
            "#,
        )
        .bind(chunk_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(row_to_chunk))
    }

    pub async fn get_chunks_by_ids(&self, ids: &[String]) -> Result<Vec<Chunk>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }

        let mut qb: QueryBuilder<Sqlite> = QueryBuilder::new(
            "SELECT id, content, kind, chapter, page, token_count, source_hash, image_path FROM chunks WHERE id IN (",
        );
        let mut separated = qb.separated(",");
        for id in ids {
            separated.push_bind(id);
        }
        separated.push_unseparated(")");

        let rows: Vec<SqliteRow> = qb.build().fetch_all(&self.pool).await?;
        let mut chunks: Vec<Chunk> = rows.into_iter().map(row_to_chunk).collect();

        chunks.sort_by_key(|chunk| {
            ids.iter()
                .position(|id| id == &chunk.id)
                .unwrap_or(usize::MAX)
        });

        Ok(chunks)
    }

    pub async fn search_chunks_by_terms(&self, terms: &[String], limit: i64) -> Result<Vec<Chunk>> {
        if terms.is_empty() || limit <= 0 {
            return Ok(vec![]);
        }

        let mut qb: QueryBuilder<Sqlite> = QueryBuilder::new(
            "SELECT id, content, kind, chapter, page, token_count, source_hash, image_path FROM chunks WHERE ",
        );

        for (idx, term) in terms.iter().enumerate() {
            if idx > 0 {
                qb.push(" OR ");
            }
            qb.push("lower(content) LIKE ");
            qb.push_bind(format!("%{}%", term.to_ascii_lowercase()));
        }

        qb.push(" LIMIT ");
        qb.push_bind(limit);

        let rows: Vec<SqliteRow> = qb.build().fetch_all(&self.pool).await?;
        Ok(rows.into_iter().map(row_to_chunk).collect())
    }

    pub async fn search_chunks_by_terms_chrono(
        &self,
        terms: &[String],
        limit: i64,
    ) -> Result<Vec<(i64, Chunk)>> {
        if terms.is_empty() || limit <= 0 {
            return Ok(vec![]);
        }

        let mut qb: QueryBuilder<Sqlite> = QueryBuilder::new(
            "SELECT rowid, id, content, kind, chapter, page, token_count, source_hash, image_path FROM chunks WHERE ",
        );

        for (idx, term) in terms.iter().enumerate() {
            if idx > 0 {
                qb.push(" OR ");
            }
            qb.push("lower(content) LIKE ");
            qb.push_bind(format!("%{}%", term.to_ascii_lowercase()));
        }

        qb.push(" ORDER BY rowid ASC LIMIT ");
        qb.push_bind(limit);

        let rows: Vec<SqliteRow> = qb.build().fetch_all(&self.pool).await?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            out.push((row.get::<i64, _>("rowid"), row_to_chunk(row)));
        }
        Ok(out)
    }

    pub async fn chunk_rowids(&self, ids: &[String]) -> Result<HashMap<String, i64>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut qb: QueryBuilder<Sqlite> =
            QueryBuilder::new("SELECT rowid, id FROM chunks WHERE id IN (");
        let mut separated = qb.separated(",");
        for id in ids {
            separated.push_bind(id);
        }
        separated.push_unseparated(")");

        let rows: Vec<SqliteRow> = qb.build().fetch_all(&self.pool).await?;
        let mut out = HashMap::with_capacity(rows.len());
        for row in rows {
            out.insert(row.get::<String, _>("id"), row.get::<i64, _>("rowid"));
        }
        Ok(out)
    }

    pub async fn part_markers(&self) -> Result<Vec<(i64, String)>> {
        let rows = sqlx::query(
            r#"
            SELECT rowid, content
            FROM chunks
            WHERE lower(trim(content)) LIKE 'part %'
            ORDER BY rowid ASC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let out = rows
            .into_iter()
            .map(|row| (row.get::<i64, _>("rowid"), row.get::<String, _>("content")))
            .collect();
        Ok(out)
    }

    pub async fn record_manifest(&self, manifest: &IngestManifest) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO manifests (docx_hash, pdf_hash, created_at, chunk_count, image_count)
            VALUES (?, ?, ?, ?, ?)
            "#,
        )
        .bind(&manifest.docx_hash)
        .bind(&manifest.pdf_hash)
        .bind(manifest.created_at.to_rfc3339())
        .bind(manifest.chunk_count)
        .bind(manifest.image_count)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn clear_image_assets(&self) -> Result<()> {
        sqlx::query("DELETE FROM image_assets")
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn insert_image_assets(&self, assets: &[crate::models::ImageAsset]) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        for asset in assets {
            sqlx::query(
                r#"
                INSERT INTO image_assets (id, page, file_path, ocr_text, caption)
                VALUES (?, ?, ?, ?, ?)
                "#,
            )
            .bind(&asset.id)
            .bind(asset.page)
            .bind(&asset.file_path)
            .bind(&asset.ocr_text)
            .bind(&asset.caption)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn latest_manifest(&self) -> Result<Option<IngestManifest>> {
        let row = sqlx::query(
            r#"
            SELECT docx_hash, pdf_hash, created_at, chunk_count, image_count
            FROM manifests
            ORDER BY id DESC
            LIMIT 1
            "#,
        )
        .fetch_optional(&self.pool)
        .await?;

        let manifest = row.map(|r| IngestManifest {
            docx_hash: r.get::<String, _>("docx_hash"),
            pdf_hash: r.get::<String, _>("pdf_hash"),
            created_at: chrono::DateTime::parse_from_rfc3339(&r.get::<String, _>("created_at"))
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            chunk_count: r.get::<i64, _>("chunk_count"),
            image_count: r.get::<i64, _>("image_count"),
        });

        Ok(manifest)
    }

    pub async fn create_session(&self) -> Result<String> {
        let session_id = Uuid::new_v4().to_string();
        sqlx::query("INSERT INTO sessions (id, created_at) VALUES (?, ?)")
            .bind(&session_id)
            .bind(Utc::now().to_rfc3339())
            .execute(&self.pool)
            .await?;

        Ok(session_id)
    }

    pub async fn ensure_session(&self, session_id: &str) -> Result<()> {
        sqlx::query("INSERT OR IGNORE INTO sessions (id, created_at) VALUES (?, ?)")
            .bind(session_id)
            .bind(Utc::now().to_rfc3339())
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    pub async fn delete_session_messages(&self, session_id: &str) -> Result<()> {
        sqlx::query("DELETE FROM messages WHERE session_id = ?")
            .bind(session_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn save_message(&self, session_id: &str, role: &str, content: &str) -> Result<()> {
        self.ensure_session(session_id).await?;
        sqlx::query(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        )
        .bind(session_id)
        .bind(role)
        .bind(content)
        .bind(Utc::now().to_rfc3339())
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn latest_messages(
        &self,
        session_id: &str,
        limit: i64,
    ) -> Result<Vec<(String, String)>> {
        let rows = sqlx::query(
            r#"
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            "#,
        )
        .bind(session_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        let mut out: Vec<(String, String)> = rows
            .into_iter()
            .map(|r| (r.get::<String, _>("role"), r.get::<String, _>("content")))
            .collect();
        out.reverse();
        Ok(out)
    }

    pub async fn upsert_ingest_status(&self, status: &IngestStatus) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO ingest_jobs (job_id, status, stage, message, chunk_count, image_count, started_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                status = excluded.status,
                stage = excluded.stage,
                message = excluded.message,
                chunk_count = excluded.chunk_count,
                image_count = excluded.image_count,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(&status.job_id)
        .bind(&status.status)
        .bind(&status.stage)
        .bind(&status.message)
        .bind(status.chunk_count)
        .bind(status.image_count)
        .bind(status.started_at.to_rfc3339())
        .bind(status.updated_at.to_rfc3339())
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_ingest_status(&self, job_id: &str) -> Result<Option<IngestStatus>> {
        let row = sqlx::query(
            r#"
            SELECT job_id, status, stage, message, chunk_count, image_count, started_at, updated_at
            FROM ingest_jobs
            WHERE job_id = ?
            "#,
        )
        .bind(job_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| IngestStatus {
            job_id: r.get("job_id"),
            status: r.get("status"),
            stage: r.get("stage"),
            message: r.get("message"),
            chunk_count: r.get("chunk_count"),
            image_count: r.get("image_count"),
            started_at: chrono::DateTime::parse_from_rfc3339(&r.get::<String, _>("started_at"))
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            updated_at: chrono::DateTime::parse_from_rfc3339(&r.get::<String, _>("updated_at"))
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        }))
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

async fn insert_chunk_tx(tx: &mut Transaction<'_, Sqlite>, chunk: &Chunk) -> Result<()> {
    sqlx::query(
        r#"
        INSERT INTO chunks (id, content, kind, chapter, page, token_count, source_hash, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(&chunk.id)
    .bind(&chunk.content)
    .bind(chunk.kind.as_str())
    .bind(&chunk.chapter)
    .bind(chunk.page)
    .bind(chunk.token_count)
    .bind(&chunk.source_hash)
    .bind(&chunk.image_path)
    .execute(&mut **tx)
    .await?;

    Ok(())
}

fn row_to_chunk(row: SqliteRow) -> Chunk {
    Chunk {
        id: row.get("id"),
        content: row.get("content"),
        kind: SourceType::from_db(&row.get::<String, _>("kind")),
        chapter: row.get("chapter"),
        page: row.get("page"),
        token_count: row.get("token_count"),
        source_hash: row.get("source_hash"),
        image_path: row.get("image_path"),
    }
}
