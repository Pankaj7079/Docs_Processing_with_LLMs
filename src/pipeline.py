# Orchestrates the full pipeline: download → extract → LLM → CSV

import time
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

from src.downloader import setup_contracts_subset
from src.extractor import process_contract_file
from src.llm_processor import process_contract_with_llm

logger = logging.getLogger(__name__)


def run_pipeline(
    data_dir: Path = Path("data"),
    subset_size: int = 50,
    output_filename: str = "extracted_clauses.csv",
    delay_between_requests: float = 4.0,
) -> Path:
    """
    Runs the legal contract processing pipeline end-to-end.

    Steps:
    1. Download (or generate) 50 contract PDFs.
    2. Extract and normalize text from each PDF.
    3. Call the Groq LLM to extract clauses and a summary.
    4. Write results to a CSV incrementally and a final time when complete.

    Returns the path to the output CSV file.
    """
    contracts_dir = data_dir / "contracts"
    output_path = data_dir / output_filename

    logger.info("=" * 60)
    logger.info("STARTING LEGAL CONTRACT PROCESSING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Ensure we have our working set of PDFs
    logger.info("Step 1: Setting up contract PDFs...")
    setup_contracts_subset(data_dir=data_dir, contracts_dir=contracts_dir, subset_size=subset_size)

    # Step 2: Collect and sort the PDF list deterministically
    pdf_files = sorted(contracts_dir.glob("*.pdf"))[:subset_size]

    if not pdf_files:
        raise FileNotFoundError(f"No contract PDFs found in '{contracts_dir}'.")

    logger.info(f"Step 2: Processing {len(pdf_files)} contracts...")

    results = []
    success_count = 0

    for pdf_path in tqdm(pdf_files, desc="Processing contracts", unit="file"):
        contract_id = pdf_path.stem

        try:
            # Extract and normalize text
            text = process_contract_file(pdf_path)

            if not text.strip():
                logger.warning(f"'{pdf_path.name}' yielded no text — skipping LLM call.")
                results.append({
                    "contract_id": contract_id,
                    "summary": "SKIPPED — empty text",
                    "termination_clause": "NOT FOUND",
                    "confidentiality_clause": "NOT FOUND",
                    "liability_clause": "NOT FOUND",
                })
                continue

            # Call LLM for clause extraction and summarization
            extracted = process_contract_with_llm(text)
            results.append({"contract_id": contract_id, **extracted})
            success_count += 1

            # Respect API rate limits
            if len(results) < len(pdf_files):
                time.sleep(delay_between_requests)

        except Exception as e:
            logger.error(f"Failed on '{pdf_path.name}': {e}")
            results.append({
                "contract_id": contract_id,
                "summary": "EXTRACTION FAILED",
                "termination_clause": "EXTRACTION FAILED",
                "confidentiality_clause": "EXTRACTION FAILED",
                "liability_clause": "EXTRACTION FAILED",
            })

        # Write after every contract so results are visible immediately
        try:
            _write_csv(results, output_path)
        except Exception as write_err:
            logger.warning(f"Incremental CSV write failed: {write_err}")

    # Final clean write with logged confirmation
    _write_csv(results, output_path)
    logger.info(f"Step 3: Done. {success_count}/{len(pdf_files)} contracts processed successfully.")
    logger.info(f"Output saved to '{output_path}'.")
    logger.info("=" * 60)

    return output_path


def _write_csv(results: list, path: Path) -> None:
    """Writes the results list to CSV with the correct column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["contract_id", "summary", "termination_clause", "confidentiality_clause", "liability_clause"]
    pd.DataFrame(results)[cols].to_csv(path, index=False)
