"""
preprocessor.py
---------------
Text cleaning, normalisation, and semantic chunking utilities.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# ── Tunable constants ──────────────────────────────────────────────
CHUNK_SIZE    = 400   # target words per chunk
CHUNK_OVERLAP = 50    # words shared between consecutive chunks
MIN_CHUNK_LEN = 30    # discard chunks shorter than this (words)


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise raw extracted text:
      - Collapse multiple blank lines / excessive whitespace.
      - Remove non-printable characters.
      - Fix hyphenated line-breaks common in PDFs.
    """
    if not text:
        return ""

    # Fix PDF hyphen line-breaks  (e.g. "exam-\nple" → "example")
    text = re.sub(r"-\n", "", text)

    # Replace newlines with spaces
    text = text.replace("\n", " ")

    # Remove non-printable / control characters
    text = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split cleaned text into overlapping word-based chunks.

    Overlapping chunks improve recall — a sentence that straddles
    a boundary still appears whole in at least one chunk.

    Args:
        text:       Cleaned plain-text string.
        chunk_size: Approximate word count per chunk.
        overlap:    Word overlap between consecutive chunks.

    Returns:
        List of non-empty chunk strings.
    """
    if not text:
        return []

    words  = text.split()
    chunks = []
    start  = 0
    step   = max(chunk_size - overlap, 1)

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])

        # Only keep chunks long enough to be meaningful
        if len(words[start:end]) >= MIN_CHUNK_LEN:
            chunks.append(chunk)

        if end == len(words):
            break
        start += step

    logger.debug("Chunked text into %d chunk(s).", len(chunks))
    return chunks


def get_text_stats(text: str) -> dict:
    """Return basic statistics about a text block."""
    words      = text.split()
    sentences  = re.split(r"[.!?]+", text)
    sentences  = [s for s in sentences if s.strip()]
    return {
        "characters": len(text),
        "words":      len(words),
        "sentences":  len(sentences),
    }
