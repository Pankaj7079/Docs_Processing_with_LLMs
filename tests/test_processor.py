# Unit tests for Groq-based LLM clause extraction and summarization
"""
Unit tests for the LLM processor module.
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from src.llm_processor import build_extraction_prompt, build_system_instruction, process_contract_with_llm, extract_relevant_context


def test_build_system_instruction():
    """
    Verifies the system instruction is a non-empty string.
    """
    instruction = build_system_instruction()
    assert isinstance(instruction, str)
    assert "legal document analyst" in instruction.lower()


def test_build_extraction_prompt():
    """
    Verifies that the prompt contains the input text and formatting instructions.
    """
    sample_text = "THIS IS A CONTRACT FOR SOFTWARE DEVELOPMENT SERVICES."
    prompt = build_extraction_prompt(sample_text)
    
    assert sample_text in prompt
    assert "termination_clause" in prompt
    assert "confidentiality_clause" in prompt
    assert "liability_clause" in prompt
    assert "summary" in prompt


def test_extract_relevant_context_filtering():
    """
    Verifies that extract_relevant_context keeps intro and filters keyword paragraphs.
    """
    # Create paragraphs where some have keywords, others don't
    p_intro = "This Agreement is made on Jan 1, 2026."
    p_random1 = "The weather today is sunny and mild."
    p_keyword1 = "The term of this contract is 1 year and either party can terminate it."
    p_random2 = "Apples are delicious and nutritious fruits."
    p_keyword2 = "Liability hereunder shall be strictly capped."
    
    full_text = "\n\n".join([p_intro, p_random1, p_keyword1, p_random2, p_keyword2])
    
    # We enforce a small max_chars to force filtering
    condensed = extract_relevant_context(full_text, max_chars=200)
    
    # The condensed output should always contain the intro
    assert p_intro in condensed
    # Should contain keyword paragraphs
    assert p_keyword1 in condensed
    # Should not contain unrelated random paragraphs if they exceed length limits or are skipped
    assert p_random2 not in condensed



@patch("src.llm_processor.Groq")
@patch.dict("os.environ", {"GROQ_API_KEY": "test_key"})
def test_process_contract_with_llm_success(mock_groq_class):
    """
    Tests successful processing of a contract with a mocked Groq API.
    """
    # Create mock client and response
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    
    # Setup the completion choices return structure
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "summary": "This is a summary.",
        "termination_clause": "Termination clause text.",
        "confidentiality_clause": "Confidentiality clause text.",
        "liability_clause": "Liability clause text."
    })
    
    mock_completions = MagicMock()
    mock_completions.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completions
    
    # Directly override the global client with our mock
    import src.llm_processor
    src.llm_processor.client = mock_client
    
    # Run processor
    result = process_contract_with_llm("Sample contract text")
    
    assert result["summary"] == "This is a summary."
    assert result["termination_clause"] == "Termination clause text."
    assert result["confidentiality_clause"] == "Confidentiality clause text."
    assert result["liability_clause"] == "Liability clause text."
    
    # Reset global client to None for other tests/runs
    src.llm_processor.client = None
    
    mock_client.chat.completions.create.assert_called_once()
