use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use regex::Regex;
use tokio::process::Command;
use uuid::Uuid;
use walkdir::WalkDir;

use super::detect_part_marker_anywhere;
use crate::models::{ImageAsset, SourceType, SourceUnit};
use crate::ollama::OllamaClient;

pub async fn extract_pdf_units(
    pdf_path: &Path,
    source_hash: &str,
    image_dir: &Path,
    ollama: &OllamaClient,
    vision_model: &str,
) -> Result<(Vec<SourceUnit>, Vec<ImageAsset>)> {
    let mut units = extract_pdf_text_units(pdf_path, source_hash).await?;
    let page_part_map = build_page_part_map(&units);

    let (image_units, image_assets) = extract_image_units(
        pdf_path,
        source_hash,
        image_dir,
        ollama,
        vision_model,
        &page_part_map,
    )
    .await?;

    units.extend(image_units);
    Ok((units, image_assets))
}

async fn extract_pdf_text_units(pdf_path: &Path, source_hash: &str) -> Result<Vec<SourceUnit>> {
    let mut units = Vec::new();
    let mut current_part: Option<String> = None;
    let mut current_part_index: Option<i64> = None;

    if has_command("pdftotext").await {
        let page_count = get_pdf_page_count(pdf_path).await.unwrap_or(0);
        if page_count > 0 {
            for page in 1..=page_count {
                let output = Command::new("pdftotext")
                    .arg("-f")
                    .arg(page.to_string())
                    .arg("-l")
                    .arg(page.to_string())
                    .arg("-layout")
                    .arg("-nopgbrk")
                    .arg(pdf_path)
                    .arg("-")
                    .output()
                    .await
                    .with_context(|| format!("failed to run pdftotext for page {}", page))?;

                if !output.status.success() {
                    continue;
                }

                let raw_text = String::from_utf8_lossy(&output.stdout).to_string();
                let content = normalize_text(&raw_text);
                if content.is_empty() {
                    continue;
                }

                if let Some((part, part_index)) = detect_part_marker_anywhere(&raw_text) {
                    current_part = Some(part);
                    current_part_index = Some(part_index);
                }

                units.push(SourceUnit {
                    kind: SourceType::PdfText,
                    chapter: None,
                    part: current_part.clone(),
                    part_index: current_part_index,
                    page: Some(page as i64),
                    content,
                    source_hash: source_hash.to_string(),
                    image_path: None,
                });
            }
        }
    }

    if units.is_empty() {
        let pdf_path = pdf_path.to_path_buf();
        let extracted = tokio::task::spawn_blocking(move || pdf_extract::extract_text(&pdf_path))
            .await
            .context("PDF extraction task panicked")?
            .context("failed to extract text from PDF")?;

        let content = normalize_text(&extracted);
        if !content.is_empty() {
            let (part, part_index) = detect_part_marker_anywhere(&extracted)
                .map_or((None, None), |(p, idx)| (Some(p), Some(idx)));
            units.push(SourceUnit {
                kind: SourceType::PdfText,
                chapter: None,
                part,
                part_index,
                page: None,
                content,
                source_hash: source_hash.to_string(),
                image_path: None,
            });
        }
    }

    Ok(units)
}

async fn extract_image_units(
    pdf_path: &Path,
    source_hash: &str,
    image_dir: &Path,
    ollama: &OllamaClient,
    vision_model: &str,
    page_part_map: &HashMap<i64, (Option<String>, Option<i64>)>,
) -> Result<(Vec<SourceUnit>, Vec<ImageAsset>)> {
    if !has_command("pdfimages").await {
        return Ok((vec![], vec![]));
    }

    tokio::fs::create_dir_all(image_dir).await?;

    let page_mapping = parse_pdfimages_page_mapping(pdf_path)
        .await
        .unwrap_or_default();
    let prefix = image_dir.join("img");
    let output = Command::new("pdfimages")
        .arg("-png")
        .arg(pdf_path)
        .arg(&prefix)
        .output()
        .await
        .context("failed to run pdfimages")?;

    if !output.status.success() {
        return Ok((vec![], vec![]));
    }

    let mut files: Vec<PathBuf> = WalkDir::new(image_dir)
        .min_depth(1)
        .max_depth(1)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.path().to_path_buf())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| {
                    matches!(
                        ext.to_ascii_lowercase().as_str(),
                        "png" | "jpg" | "jpeg" | "pbm" | "ppm" | "tiff"
                    )
                })
                .unwrap_or(false)
        })
        .collect();

    files.sort();

    let mut units = Vec::new();
    let mut assets = Vec::new();

    for (index, file_path) in files.iter().enumerate() {
        let page = page_mapping.get(index).copied();
        let (part, part_index) = page
            .and_then(|p| page_part_map.get(&p).cloned())
            .unwrap_or((None, None));
        let ocr_text = extract_ocr_text(file_path).await.unwrap_or_default();

        let image_bytes = tokio::fs::read(file_path).await.unwrap_or_default();
        let caption = if image_bytes.is_empty() {
            String::new()
        } else {
            ollama
                .generate_vision_caption(
                    vision_model,
                    "Describe this image from a novel manuscript. Focus on characters, objects, locations, and text that matters for answering story questions.",
                    &image_bytes,
                )
                .await
                .unwrap_or_default()
        };

        let asset = ImageAsset {
            id: Uuid::new_v4().to_string(),
            page,
            file_path: file_path.display().to_string(),
            ocr_text: ocr_text.clone(),
            caption: caption.clone(),
        };

        if !caption.trim().is_empty() {
            units.push(SourceUnit {
                kind: SourceType::ImageCaption,
                chapter: None,
                part: part.clone(),
                part_index,
                page,
                content: normalize_text(&caption),
                source_hash: source_hash.to_string(),
                image_path: Some(asset.file_path.clone()),
            });
        }

        if !ocr_text.trim().is_empty() {
            units.push(SourceUnit {
                kind: SourceType::ImageOcr,
                chapter: None,
                part: part.clone(),
                part_index,
                page,
                content: normalize_text(&ocr_text),
                source_hash: source_hash.to_string(),
                image_path: Some(asset.file_path.clone()),
            });
        }

        assets.push(asset);
    }

    Ok((units, assets))
}

async fn get_pdf_page_count(pdf_path: &Path) -> Result<usize> {
    let output = Command::new("pdfinfo")
        .arg(pdf_path)
        .output()
        .await
        .context("failed to run pdfinfo")?;

    if !output.status.success() {
        return Err(anyhow::anyhow!("pdfinfo exited with non-zero status"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let regex = Regex::new(r"(?m)^Pages:\s+(\d+)\s*$")?;
    let pages = regex
        .captures(&stdout)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse::<usize>().ok())
        .ok_or_else(|| anyhow::anyhow!("unable to parse page count from pdfinfo"))?;

    Ok(pages)
}

async fn parse_pdfimages_page_mapping(pdf_path: &Path) -> Result<Vec<i64>> {
    let output = Command::new("pdfimages")
        .arg("-list")
        .arg(pdf_path)
        .output()
        .await
        .context("failed to run pdfimages -list")?;

    if !output.status.success() {
        return Err(anyhow::anyhow!("pdfimages -list failed"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut pages = Vec::new();

    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("page") || trimmed.starts_with("---") {
            continue;
        }

        let cols: Vec<&str> = trimmed.split_whitespace().collect();
        if cols.is_empty() {
            continue;
        }

        if let Ok(page) = cols[0].parse::<i64>() {
            pages.push(page);
        }
    }

    Ok(pages)
}

async fn extract_ocr_text(image_path: &Path) -> Result<String> {
    if !has_command("tesseract").await {
        return Ok(String::new());
    }

    let output = Command::new("tesseract")
        .arg(image_path)
        .arg("stdout")
        .arg("--dpi")
        .arg("300")
        .output()
        .await
        .context("failed to run tesseract")?;

    if !output.status.success() {
        return Ok(String::new());
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

async fn has_command(binary: &str) -> bool {
    // Some binaries (e.g. poppler's pdfimages) return non-zero for --version,
    // so check PATH presence via `which` instead of probing a specific flag.
    Command::new("which")
        .arg(binary)
        .output()
        .await
        .map(|out| out.status.success() && !out.stdout.is_empty())
        .unwrap_or(false)
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

fn build_page_part_map(units: &[SourceUnit]) -> HashMap<i64, (Option<String>, Option<i64>)> {
    let mut map = HashMap::new();
    for unit in units {
        let Some(page) = unit.page else {
            continue;
        };
        map.insert(page, (unit.part.clone(), unit.part_index));
    }
    map
}
