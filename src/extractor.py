# Handles text extraction and normalization from contract PDFs

import re
from pathlib import Path
import logging
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Cleans raw text extracted from a PDF.
    Converts smart quotes, various dash types, and ligature artifacts
    to their plain ASCII equivalents, then collapses excess whitespace.
    """
    if not text:
        return ""

    # Smart double quotes → standard double quotes
    text = re.sub(r'[\u201c\u201d\u201f\u00ab\u00bb]', '"', text)

    # Smart single quotes and apostrophes → standard apostrophe
    text = re.sub(r'[\u2018\u2019\u201a\u201b`´]', "'", text)

    # En-dash, em-dash, minus sign → hyphen
    text = re.sub(r'[\u2013\u2014\u2015\u2212]', '-', text)

    # Common PDF symbol artifacts
    text = text.replace('\uf0b7', '-')
    text = text.replace('\uf02d', '-')

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Strip trailing spaces from each line
    lines = [line.strip() for line in text.split('\n')]

    # Collapse runs of blank lines to a single blank line
    output = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 1:
                output.append("")
        else:
            blank_count = 0
            output.append(re.sub(r'[ \t]+', ' ', line))

    return "\n".join(output).strip()


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extracts all text from a PDF file, page by page.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    parts = []
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                parts.append(page_text)
            else:
                logger.debug(f"Page {i + 1} of '{pdf_path.name}' returned no text (may be scanned).")
        return "\n".join(parts)
    except Exception as e:
        logger.error(f"Failed to read '{pdf_path.name}': {e}")
        raise


def process_contract_file(pdf_path: Path) -> str:
    """Extracts and normalizes text from a single contract PDF."""
    raw = extract_text_from_pdf(pdf_path)
    return normalize_text(raw)
