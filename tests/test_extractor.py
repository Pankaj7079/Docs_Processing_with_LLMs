# Unit tests for PDF text extractor and normalizer
"""
Unit tests for the extractor module.
"""


import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.extractor import normalize_text, extract_text_from_pdf, process_contract_file


def test_normalize_text_smart_quotes():
    """
    Tests that smart quotes are correctly converted to standard ASCII quotes.
    """
    input_text = "The “agreement” signed by ‘John’."
    expected = "The \"agreement\" signed by 'John'."
    assert normalize_text(input_text) == expected


def test_normalize_text_ligatures_and_dashes():
    """
    Tests that dashes and ligatures are converted.
    """
    # Unicode en-dash (\u2013) and em-dash (\u2014)
    input_text = "Term – 1 Year — Condition"
    expected = "Term - 1 Year - Condition"
    assert normalize_text(input_text) == expected


def test_normalize_text_excess_spacing():
    """
    Tests that consecutive spaces are reduced and line spacing is normalized.
    """
    input_text = "Section 1.   Definition of   Terms.   \n\n\n\n  Next paragraph text. "
    # Expect double spaces inside line reduced to single, trailing spaces stripped, and max 1 empty line
    expected = "Section 1. Definition of Terms.\n\nNext paragraph text."
    assert normalize_text(input_text) == expected


@patch("src.extractor.PdfReader")
@patch("src.extractor.Path.exists")
def test_extract_text_from_pdf_mocked(mock_exists, mock_pdf_reader):
    """
    Tests PDF extraction with a mocked PdfReader.
    """
    # Setup mocks
    mock_exists.return_value = True
    mock_reader_instance = MagicMock()
    mock_pdf_reader.return_value = mock_reader_instance
    
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page 1 content."
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Page 2 content."
    
    mock_reader_instance.pages = [mock_page1, mock_page2]
    
    # Run text extraction
    result = extract_text_from_pdf(Path("dummy.pdf"))
    
    assert "Page 1 content." in result
    assert "Page 2 content." in result
    mock_pdf_reader.assert_called_once_with(Path("dummy.pdf"))
