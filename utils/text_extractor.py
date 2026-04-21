"""
text_extractor.py
-----------------
Handles text extraction from PDF, DOCX, and TXT files.
Supports both PyPDF2 and pdfplumber for robust PDF parsing.
"""

import io
import os
import logging

import PyPDF2
import pdfplumber
from docx import Document

# Configure module-level logger
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Dispatch extraction based on file extension.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename (used to determine type).

    Returns:
        Extracted plain text string. Empty string on failure.
    """
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".pdf":
            return _extract_pdf(file_bytes)
        elif ext == ".docx":
            return _extract_docx(file_bytes)
        elif ext == ".txt":
            return _extract_txt(file_bytes)
        else:
            logger.warning("Unsupported file type: %s", ext)
            return ""
    except Exception as exc:
        logger.error("Extraction failed for %s: %s", filename, exc)
        return ""


# ──────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────

def _extract_pdf(file_bytes: bytes) -> str:
    """
    Try pdfplumber first (better layout handling),
    fall back to PyPDF2 if pdfplumber returns nothing.
    """
    text = _pdf_pdfplumber(file_bytes)
    if not text.strip():
        text = _pdf_pypdf2(file_bytes)
    return text


def _pdf_pdfplumber(file_bytes: bytes) -> str:
    """Extract text from PDF using pdfplumber."""
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
    return "\n".join(pages)


def _pdf_pypdf2(file_bytes: bytes) -> str:
    """Fallback PDF extraction using PyPDF2."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return "\n".join(pages)


def _extract_docx(file_bytes: bytes) -> str:
    """Extract text from a .docx file using python-docx."""
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def _extract_txt(file_bytes: bytes) -> str:
    """Decode a plain-text file, trying UTF-8 then latin-1."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")
