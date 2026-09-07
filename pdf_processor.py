from __future__ import annotations

from dataclasses import dataclass

import pymupdf as fitz


@dataclass(frozen=True)
class DocumentChunk:
    source_document: str
    page_number: int
    text: str
    chunk_index: int


def extract_pages(pdf_bytes: bytes, source_document: str) -> list[tuple[int, str]]:
    """Return (1-based page_number, text) pairs from a PDF."""
    pages: list[tuple[int, str]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for index, page in enumerate(document, start=1):
            text = page.get_text("text") or ""
            pages.append((index, _normalize_whitespace(text)))
    if not pages:
        raise ValueError(f"No pages found in {source_document}")
    return pages


def chunk_pages(
    pages: list[tuple[int, str]],
    source_document: str,
    *,
    max_chars: int = 1800,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """Split page text into chunks. Each chunk stays on a single page."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and smaller than max_chars")

    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page_number, text in pages:
        if not text.strip():
            continue
        page_chunks = _split_text(text, max_chars=max_chars, overlap=overlap)
        for piece in page_chunks:
            chunks.append(
                DocumentChunk(
                    source_document=source_document,
                    page_number=page_number,
                    text=piece,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

    return chunks


def process_pdf(
    pdf_bytes: bytes,
    source_document: str,
    *,
    max_chars: int = 1800,
    overlap: int = 200,
) -> list[DocumentChunk]:
    pages = extract_pages(pdf_bytes, source_document)
    return chunk_pages(
        pages,
        source_document,
        max_chars=max_chars,
        overlap=overlap,
    )


def _split_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return [stripped]

    pieces: list[str] = []
    start = 0
    length = len(stripped)

    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            break_at = _best_break(stripped, start, end)
            end = break_at
        piece = stripped[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        next_start = max(end - overlap, start + 1)
        start = next_start

    return pieces


def _best_break(text: str, start: int, end: int) -> int:
    window = text[start:end]
    for separator in ("\n\n", "\n", ". ", "; "):
        index = window.rfind(separator)
        if index >= max_chars_floor(window):
            return start + index + len(separator)
    return end


def max_chars_floor(window: str) -> int:
    return max(0, int(len(window) * 0.6))


def _normalize_whitespace(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    collapsed: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if not blank and collapsed:
                collapsed.append("")
                blank = True
            continue
        collapsed.append(line)
        blank = False
    return "\n".join(collapsed).strip()
