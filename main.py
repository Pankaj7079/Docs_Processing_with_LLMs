# Main entrypoint — run the pipeline or search extracted clauses

import os
import argparse
import sys
from pathlib import Path
import logging

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import run_pipeline
from src.semantic_search import ClauseSearchEngine

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("main")


def check_api_key() -> bool:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY is not set. Please add it to your .env file.")
        return False
    return True


def handle_search(query: str, csv_path: Path, top_k: int) -> None:
    if not csv_path.exists():
        logger.error(f"Output file '{csv_path}' not found. Run the pipeline first.")
        return

    try:
        engine = ClauseSearchEngine()
        indexed_count = engine.index_csv(csv_path)
        if indexed_count == 0:
            logger.warning("No clauses were indexed. The output file may be empty.")
            return

        logger.info(f"Searching for: '{query}'")
        results = engine.search(query, top_k=top_k)

        print("\n" + "=" * 80)
        print(f" Top {top_k} results for: '{query}'")
        print("=" * 80)

        for i, (doc, score) in enumerate(results, 1):
            print(f"\n[{i}]  Score: {score:.4f}")
            print(f"     Contract : {doc['contract_id']}")
            print(f"     Clause   : {doc['clause_type']}")
            snippet = doc["text"]
            if len(snippet) > 400:
                snippet = snippet[:400] + "..."
            for line in snippet.split("\n"):
                print(f"     {line}")
            print("-" * 80)

    except Exception as e:
        logger.error(f"Search failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Legal contract processing pipeline — extract clauses and search them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="Run the full extraction pipeline.")
    group.add_argument("--search", type=str, metavar="QUERY", help="Search extracted clauses by query string.")

    parser.add_argument("--subset-size", type=int, default=50, help="Number of contracts to process.")
    parser.add_argument("--output", type=str, default="data/extracted_clauses.csv", help="Output CSV path.")
    parser.add_argument("--delay", type=float, default=4.0, help="Seconds to wait between LLM calls.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of search results to show.")

    args = parser.parse_args()

    if args.run:
        if not check_api_key():
            sys.exit(1)
        try:
            output_path = Path(args.output)
            run_pipeline(
                subset_size=args.subset_size,
                output_filename=output_path.name,
                data_dir=output_path.parent,
                delay_between_requests=args.delay
            )
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            sys.exit(1)

    elif args.search:
        handle_search(args.search, Path(args.output), args.top_k)


if __name__ == "__main__":
    main()
