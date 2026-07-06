"""
PDF text extraction using PyMuPDF (fitz).

Returns structured output: per-page text + full concatenated text + page count.
"""

from __future__ import annotations

import fitz  # PyMuPDF


def extract_pdf(file_path: str) -> dict:
    """
    Extract text from a PDF file.

    Args:
        file_path: Absolute or relative path to the .pdf file.

    Returns:
        {
            "pages": [{"page_num": int, "text": str}, ...],
            "full_text": str,
            "page_count": int,
        }

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file cannot be opened as a PDF.
    """
    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {file_path}") from exc

    pages = []
    full_text_parts = []

    page_count = len(doc)
    for page_index in range(page_count):
        page = doc[page_index]
        try:
            raw_text = page.get_text("text")
        except Exception:
            # Some pages (images-only, corrupted) may fail; skip them.
            continue

        cleaned = _clean_page_text(raw_text)
        if not cleaned:
            # Skip completely empty pages
            continue

        page_num = page_index + 1
        pages.append({"page_num": page_num, "text": cleaned})
        full_text_parts.append(cleaned)

    doc.close()

    return {
        "pages": pages,
        "full_text": "\n\n".join(full_text_parts),
        "page_count": page_count,  # total pages in file, not just non-empty
    }


def _clean_page_text(text: str) -> str:
    """
    Remove obvious noise from a raw PDF page text.

    - Collapses runs of blank lines to a single blank line.
    - Strips leading/trailing whitespace per line.
    - Removes lines that are pure whitespace or single-character noise.
    """
    lines = text.splitlines()
    cleaned_lines = []
    prev_blank = False

    for line in lines:
        stripped = line.strip()

        # Drop lines that are purely noise (single chars, empty)
        if len(stripped) <= 1:
            if not prev_blank:
                cleaned_lines.append("")  # preserve paragraph breaks
            prev_blank = True
            continue

        cleaned_lines.append(stripped)
        prev_blank = False

    return "\n".join(cleaned_lines).strip()
