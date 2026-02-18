use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};
use roxmltree::Document;
use zip::ZipArchive;

use super::detect_part_marker;
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
    let mut current_part: Option<String> = None;
    let mut current_part_index: Option<i64> = None;
    let mut inferred_part_number: Option<i64> = None;
    let mut last_heading_was_part_marker = false;
    let mut units = Vec::new();

    for paragraph in doc
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "p")
    {
        let style = paragraph_style(paragraph);
        let centered = paragraph_is_centered(paragraph);
        let max_font_size_half_points = paragraph_max_font_size_half_points(paragraph);

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

        if is_heading_paragraph(
            style.as_deref(),
            centered,
            max_font_size_half_points,
            &normalized,
        ) {
            current_chapter = Some(normalized.clone());
            if let Some((part, part_index)) = detect_part_marker(&normalized) {
                current_part = Some(part);
                current_part_index = Some(part_index);
                inferred_part_number = Some(part_index / 10);
                last_heading_was_part_marker = true;
                continue;
            }

            if is_introduction_heading(&normalized) && !last_heading_was_part_marker {
                let next_part = inferred_part_number.map(|n| n + 1).unwrap_or(1);
                inferred_part_number = Some(next_part);
                current_part = Some(format!("Part {next_part}"));
                current_part_index = Some(next_part * 10);
            }

            last_heading_was_part_marker = false;
            continue;
        }

        if let Some((part, part_index)) = detect_part_marker(&normalized) {
            current_part = Some(part);
            current_part_index = Some(part_index);
            inferred_part_number = Some(part_index / 10);
            last_heading_was_part_marker = true;
            continue;
        }

        last_heading_was_part_marker = false;

        units.push(SourceUnit {
            kind: SourceType::DocxText,
            chapter: current_chapter.clone(),
            part: current_part.clone(),
            part_index: current_part_index,
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

fn paragraph_style(paragraph: roxmltree::Node<'_, '_>) -> Option<String> {
    paragraph
        .descendants()
        .find(|node| node.is_element() && node.tag_name().name() == "pStyle")
        .and_then(|node| {
            node.attributes()
                .find(|attr| attr.name().ends_with("val"))
                .map(|attr| attr.value().to_string())
        })
}

fn paragraph_is_centered(paragraph: roxmltree::Node<'_, '_>) -> bool {
    paragraph
        .descendants()
        .find(|node| node.is_element() && node.tag_name().name() == "jc")
        .and_then(|node| {
            node.attributes()
                .find(|attr| attr.name().ends_with("val"))
                .map(|attr| attr.value().to_ascii_lowercase())
        })
        .map(|value| value == "center")
        .unwrap_or(false)
}

fn paragraph_max_font_size_half_points(paragraph: roxmltree::Node<'_, '_>) -> Option<i64> {
    paragraph
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "sz")
        .filter_map(|node| {
            node.attributes()
                .find(|attr| attr.name().ends_with("val"))
                .and_then(|attr| attr.value().parse::<i64>().ok())
        })
        .max()
}

fn is_heading_paragraph(
    style: Option<&str>,
    centered: bool,
    max_font_size_half_points: Option<i64>,
    text: &str,
) -> bool {
    if style
        .map(|s| s.to_ascii_lowercase().contains("heading"))
        .unwrap_or(false)
    {
        return true;
    }

    if !centered {
        return false;
    }

    let words = text.split_whitespace().count();
    if words == 0 || words > 10 || text.len() > 96 {
        return false;
    }
    if text.ends_with('.') || text.ends_with('!') || text.ends_with('?') {
        return false;
    }

    let size = max_font_size_half_points.unwrap_or(0);
    size >= 30 || (size >= 26 && words <= 5)
}

fn is_introduction_heading(text: &str) -> bool {
    let cleaned = text
        .trim()
        .trim_matches(|c: char| matches!(c, ':' | '-' | '–' | '—'))
        .trim();
    cleaned.eq_ignore_ascii_case("introduction")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn introduction_heading_is_detected() {
        assert!(is_introduction_heading("Introduction"));
        assert!(is_introduction_heading(" introduction "));
        assert!(is_introduction_heading("Introduction:"));
        assert!(!is_introduction_heading("The Introduction Arc"));
    }

    #[test]
    fn heading_heuristic_prefers_short_centered_titles() {
        assert!(is_heading_paragraph(
            None,
            true,
            Some(36),
            "The Geek Empire"
        ));
        assert!(is_heading_paragraph(None, true, Some(30), "PranavM I"));
        assert!(!is_heading_paragraph(
            None,
            false,
            Some(36),
            "The Geek Empire"
        ));
        assert!(!is_heading_paragraph(
            None,
            true,
            Some(24),
            "This is a long narrative paragraph that should not be considered a heading."
        ));
    }
}
