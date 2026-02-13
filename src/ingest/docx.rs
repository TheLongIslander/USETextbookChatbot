use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};
use roxmltree::Document;
use zip::ZipArchive;

use crate::models::{SourceType, SourceUnit};

pub fn extract_docx_units(path: &Path, source_hash: &str) -> Result<Vec<SourceUnit>> {
    let file =
        File::open(path).with_context(|| format!("failed to open DOCX: {}", path.display()))?;
    let mut archive = ZipArchive::new(file).context("DOCX is not a valid ZIP archive")?;

    let mut document_xml = String::new();
    archive
        .by_name("word/document.xml")
        .context("DOCX missing word/document.xml")?
        .read_to_string(&mut document_xml)
        .context("failed to read word/document.xml")?;

    let doc = Document::parse(&document_xml).context("failed to parse DOCX XML")?;

    let mut current_chapter: Option<String> = None;
    let mut units = Vec::new();

    for paragraph in doc
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "p")
    {
        let style = paragraph
            .descendants()
            .find(|node| node.is_element() && node.tag_name().name() == "pStyle")
            .and_then(|node| {
                node.attributes()
                    .find(|attr| attr.name().ends_with("val"))
                    .map(|attr| attr.value().to_string())
            });

        let text = paragraph
            .descendants()
            .filter(|node| node.is_element() && node.tag_name().name() == "t")
            .filter_map(|node| node.text())
            .collect::<Vec<_>>()
            .join("");

        let normalized = normalize_text(&text);
        if normalized.is_empty() {
            continue;
        }

        if style
            .as_ref()
            .map(|style| style.to_ascii_lowercase().contains("heading"))
            .unwrap_or(false)
        {
            current_chapter = Some(normalized);
            continue;
        }

        units.push(SourceUnit {
            kind: SourceType::DocxText,
            chapter: current_chapter.clone(),
            page: None,
            content: normalized,
            source_hash: source_hash.to_string(),
            image_path: None,
        });
    }

    Ok(units)
}

fn normalize_text(input: &str) -> String {
    input
        .replace(['\u{2018}', '\u{2019}'], "'")
        .replace(['\u{201C}', '\u{201D}'], "\"")
        .replace('\u{00A0}', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}
