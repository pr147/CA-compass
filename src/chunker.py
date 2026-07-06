"""
Text chunking utility.

Strategy:
  1. Split on double-newlines (paragraph boundaries).
  2. Merge short paragraphs into the previous chunk until max_chars is reached.
  3. If a single paragraph exceeds max_chars, split it on sentence boundaries.
  4. Drop chunks that fall below min_chars (noise / headers).
"""

from __future__ import annotations

import re


def chunk_text(
    text: str,
    max_chars: int = 2500,
    min_chars: int = 200,
) -> list[str]:
    """
    Split extracted PDF text into LLM-friendly chunks.

    Args:
        text:      Full extracted document text.
        max_chars: Soft upper bound on characters per chunk.
        min_chars: Minimum characters for a chunk to be kept.

    Returns:
        List of non-empty string chunks.
    """
    if not text or not text.strip():
        return []

    # Step 1: split into paragraphs on blank lines
    paragraphs = re.split(r"\n{2,}", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Step 2: merge short paragraphs; split oversized ones
    raw_chunks: list[str] = []
    buffer = ""

    for para in paragraphs:
        # If a single paragraph is too long, break it at sentence boundaries
        if len(para) > max_chars:
            sentences = _split_sentences(para)
            for sentence in sentences:
                if len(buffer) + len(sentence) + 1 <= max_chars:
                    buffer = (buffer + " " + sentence).strip()
                else:
                    if buffer:
                        raw_chunks.append(buffer)
                    buffer = sentence
        else:
            if len(buffer) + len(para) + 2 <= max_chars:
                buffer = (buffer + "\n\n" + para).strip() if buffer else para
            else:
                if buffer:
                    raw_chunks.append(buffer)
                buffer = para

    if buffer:
        raw_chunks.append(buffer)

    # Step 3: filter out chunks that are too small to be useful
    chunks = [c for c in raw_chunks if len(c) >= min_chars]

    return chunks


def _split_sentences(text: str) -> list[str]:
    """
    Naive sentence splitter on common English punctuation.
    Good enough for news/current-affairs text.
    """
    # Split after . ! ? followed by whitespace and an uppercase letter
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]
