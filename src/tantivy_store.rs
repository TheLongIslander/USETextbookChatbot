use std::path::PathBuf;

use anyhow::Result;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, Value, STORED, STRING, TEXT};
use tantivy::{doc, Index};

use crate::models::Chunk;

#[derive(Clone)]
pub struct TantivyStore {
    index_dir: PathBuf,
}

#[derive(Clone, Copy)]
struct TantivyFields {
    chunk_id: Field,
    content: Field,
    kind: Field,
    chapter: Field,
    page: Field,
}

impl TantivyStore {
    pub fn new(index_dir: PathBuf) -> Self {
        Self { index_dir }
    }

    pub fn rebuild(&self, chunks: &[Chunk]) -> Result<()> {
        if self.index_dir.exists() {
            std::fs::remove_dir_all(&self.index_dir)?;
        }
        std::fs::create_dir_all(&self.index_dir)?;

        let (schema, fields) = build_schema();
        let index = Index::create_in_dir(&self.index_dir, schema)?;
        let mut writer = index.writer(50_000_000)?;

        for chunk in chunks {
            writer.add_document(doc!(
                fields.chunk_id => chunk.id.clone(),
                fields.content => chunk.content.clone(),
                fields.kind => chunk.kind.as_str().to_string(),
                fields.chapter => chunk.chapter.clone().unwrap_or_default(),
                fields.page => chunk.page.unwrap_or_default(),
            ))?;
        }

        writer.commit()?;
        Ok(())
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<(String, f32)>> {
        if !self.index_dir.exists() {
            return Ok(vec![]);
        }

        let index = Index::open_in_dir(&self.index_dir)?;
        let schema = index.schema();
        let fields = resolve_fields(&schema)?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&index, vec![fields.content, fields.chapter]);
        let query = query_parser.parse_query(query)?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;

        let mut out = Vec::with_capacity(top_docs.len());
        for (score, addr) in top_docs {
            let doc = searcher.doc::<tantivy::schema::TantivyDocument>(addr)?;
            if let Some(chunk_id) = doc
                .get_first(fields.chunk_id)
                .and_then(|value| value.as_str())
            {
                out.push((chunk_id.to_string(), score));
            }
        }

        Ok(out)
    }
}

fn build_schema() -> (Schema, TantivyFields) {
    let mut builder = Schema::builder();

    let chunk_id = builder.add_text_field("chunk_id", STRING | STORED);
    let content = builder.add_text_field("content", TEXT | STORED);
    let kind = builder.add_text_field("kind", STRING | STORED);
    let chapter = builder.add_text_field("chapter", TEXT | STORED);
    let page = builder.add_i64_field("page", STORED);

    (
        builder.build(),
        TantivyFields {
            chunk_id,
            content,
            kind,
            chapter,
            page,
        },
    )
}

fn resolve_fields(schema: &Schema) -> Result<TantivyFields> {
    Ok(TantivyFields {
        chunk_id: schema
            .get_field("chunk_id")
            .map_err(|err| anyhow::anyhow!(err.to_string()))?,
        content: schema
            .get_field("content")
            .map_err(|err| anyhow::anyhow!(err.to_string()))?,
        kind: schema
            .get_field("kind")
            .map_err(|err| anyhow::anyhow!(err.to_string()))?,
        chapter: schema
            .get_field("chapter")
            .map_err(|err| anyhow::anyhow!(err.to_string()))?,
        page: schema
            .get_field("page")
            .map_err(|err| anyhow::anyhow!(err.to_string()))?,
    })
}
