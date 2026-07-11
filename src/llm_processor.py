# Uses the Groq API to extract key clauses and a summary from contract text

import os
import json
import logging
import time
from typing import Dict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama-3.1-8b-instant"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found in environment. Set it in .env before running.")


def build_system_instruction() -> str:
    return (
        "You are an expert legal document analyst. Read the contract provided, "
        "extract specific clauses verbatim from the text, and write a concise summary. "
        "Always return your response as a valid JSON object with no extra text or markdown."
    )


def build_extraction_prompt(contract_text: str) -> str:
    prompt = f"""Analyze the legal contract below and return a JSON object with these exact keys:

1. "summary" — A 100-150 word summary covering:
   - Purpose of the agreement
   - Key obligations of each party
   - Notable risks or penalties

2. "termination_clause" — Verbatim extract of how/when the contract can be terminated.
3. "confidentiality_clause" — Verbatim extract covering non-disclosure and confidentiality duties.
4. "liability_clause" — Verbatim extract covering liability caps, indemnification, or exclusions.

If a clause is not present, set its value to "NOT FOUND".

--- EXAMPLE ---
CONTRACT:
NON-DISCLOSURE AGREEMENT between Alpha Corp and Beta Inc, dated January 1, 2026.
1. Purpose. The parties wish to evaluate a potential software development partnership.
2. Confidentiality. Each party shall hold the other's Confidential Information in strict confidence for 3 years from disclosure.
3. Term and Termination. This Agreement is effective for one year. Either party may terminate with 30 days written notice.
4. Limitation of Liability. EXCEPT FOR BREACHES OF CONFIDENTIALITY, NEITHER PARTY SHALL BE LIABLE FOR INDIRECT DAMAGES. TOTAL LIABILITY SHALL NOT EXCEED $50,000.

EXPECTED OUTPUT:
{{
  "summary": "This Mutual NDA between Alpha Corp and Beta Inc establishes a one-year framework for sharing confidential information to evaluate a potential software development partnership. Both parties must protect shared information with reasonable care for three years. Either party can terminate with 30 days written notice. Key risk: confidentiality breaches are carved out of the $50,000 liability cap, meaning unlimited exposure for such violations.",
  "termination_clause": "Term and Termination. This Agreement is effective for one year. Either party may terminate with 30 days written notice.",
  "confidentiality_clause": "Confidentiality. Each party shall hold the other's Confidential Information in strict confidence for 3 years from disclosure.",
  "liability_clause": "Limitation of Liability. EXCEPT FOR BREACHES OF CONFIDENTIALITY, NEITHER PARTY SHALL BE LIABLE FOR INDIRECT DAMAGES. TOTAL LIABILITY SHALL NOT EXCEED $50,000."
}}
--- END EXAMPLE ---

Now analyze this contract and return only the JSON:

{contract_text}
"""
    return prompt


def extract_relevant_context(text: str, max_chars: int = 12000) -> str:
    """
    Trims a long contract to its most legally relevant sections.
    Keeps the opening paragraphs and any paragraphs containing key legal terms,
    staying within the character budget to avoid token limit errors.
    """
    if len(text) <= max_chars:
        return text

    paragraphs = text.split("\n\n")
    if len(paragraphs) <= 1:
        paragraphs = text.split("\n")

    keywords = [
        "terminate", "termination", "expiry", "expiration", "duration",
        "confidential", "non-disclosure", "proprietary", "disclosure",
        "liability", "indemnif", "damages", "consequential", "limitation of",
    ]

    # Always keep the opening section
    intro_count = min(3, len(paragraphs))
    selected = list(paragraphs[:intro_count])
    current_len = sum(len(p) for p in selected)

    for para in paragraphs[intro_count:]:
        if any(kw in para.lower() for kw in keywords):
            if current_len + len(para) < max_chars:
                selected.append(para)
                current_len += len(para)
            else:
                break

    # If keyword filtering found nothing useful, just truncate
    if len(selected) <= intro_count:
        return text[:max_chars]

    return "\n\n...\n\n".join(selected)


def process_contract_with_llm(
    contract_text: str,
    model_name: str = DEFAULT_MODEL,
    max_retries: int = 3,
    retry_delay: int = 2
) -> Dict[str, str]:
    """
    Sends a contract to the Groq LLM and returns structured clause data.

    Returns a dict with keys: summary, termination_clause, confidentiality_clause, liability_clause.
    """
    global client

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not set. Please configure your .env file.")

    if client is None:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    condensed = extract_relevant_context(contract_text)
    system_msg = build_system_instruction()
    user_msg = build_extraction_prompt(condensed)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Sending to Groq (model={model_name}, attempt {attempt}/{max_retries})...")

            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                model=model_name,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            response_text = completion.choices[0].message.content.strip()
            parsed = json.loads(response_text)

            required_keys = ["summary", "termination_clause", "confidentiality_clause", "liability_clause"]
            return {key: parsed.get(key, "NOT FOUND") for key in required_keys}

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise
        except Exception as e:
            logger.warning(f"API error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise
            time.sleep(retry_delay * attempt)

    return {
        "summary": "Extraction failed.",
        "termination_clause": "Extraction failed.",
        "confidentiality_clause": "Extraction failed.",
        "liability_clause": "Extraction failed.",
    }
