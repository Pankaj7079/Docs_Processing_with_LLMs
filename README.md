# Legal Contract Processing Pipeline

A Python pipeline that extracts and summarizes key clauses from legal contracts using an LLM.

It works with the [CUAD dataset](https://www.atticusprojectai.org/cuad) — a public collection of over 500 commercial contracts annotated with 41 clause categories. The pipeline processes a 50-contract subset, extracts termination, confidentiality, and liability clauses verbatim, and generates a short plain-English summary for each.

---

## How it works

```
CUAD dataset (Zenodo)
        │
        ▼
  PDF text extraction
  + text normalization   (pypdf)
        │
        ▼
  Keyword-based context filtering
  (trims to ~3k tokens to stay within free-tier API limits)
        │
        ▼
  Groq LLM — llama-3.1-8b-instant
  (few-shot prompting + JSON mode)
        │
        ▼
  extracted_clauses.csv
  [contract_id | summary | termination_clause | confidentiality_clause | liability_clause]
        │
        ▼
  Semantic search   (sentence-transformers / all-MiniLM-L6-v2)
  Runs fully local — no extra API needed
```

---

## Project structure

```
.
├── main.py                  # CLI entrypoint
├── requirements.txt
├── .env.template            # Copy to .env and fill in your Groq key
├── src/
│   ├── downloader.py        # Downloads CUAD zip; generates synthetic PDFs as fallback
│   ├── extractor.py         # PDF text extraction and normalization
│   ├── llm_processor.py     # Groq API calls, few-shot prompting, JSON parsing
│   ├── pipeline.py          # Orchestration: ties all modules together
│   └── semantic_search.py   # Local embedding index and cosine similarity search
├── tests/
│   ├── test_extractor.py
│   └── test_processor.py
└── data/                    # Auto-generated at runtime, not committed to git
    ├── contracts/           # 50 contract PDFs
    └── extracted_clauses.csv
```

---

## Setup

**Requirements**: Python 3.10+, [`uv`](https://github.com/astral-sh/uv)

Install `uv` if you don't have it:
```bash
pip install uv
```

Clone the repo and install dependencies:
```bash
git clone https://github.com/Pankaj7079/Docs_Processing_with_LLMs
cd Docs_Processing_with_LLMs

uv venv
uv pip install -r requirements.txt
```

Create your `.env` file from the template:

```bash
# Linux / macOS
cp .env.template .env

# Windows
copy .env.template .env
```

Then open `.env` and set your [Groq API key](https://console.groq.com):
```
GROQ_API_KEY=your_key_here
```

---

## Usage

**Run the full extraction pipeline:**

```bash
uv run python main.py --run
```

Downloads the contracts, extracts text, calls the Groq LLM for each contract, and writes results to `data/extracted_clauses.csv`. Results are saved incrementally after every contract, so a partial run is never lost.

Optional flags:
- `--subset-size N` — process N contracts instead of the default 50
- `--output path/to/output.csv` — custom output file path
- `--delay 4.0` — seconds to pause between LLM calls (reduces rate limit errors)

---

**Search extracted clauses semantically:**

```bash
uv run python main.py --search "limitation of liability"
```

Builds a local embedding index from the CSV and returns the most relevant clause matches.

Optional flags:
- `--top-k 5` — number of results to return (default: 3)
- `--output path/to/output.csv` — path to the CSV to search (default: `data/extracted_clauses.csv`)

---

## Running tests

```bash
uv run python -m pytest -v
```

8 unit tests covering text normalization, mocked PDF extraction, prompt construction, and LLM JSON parsing.

---

## Notes

- **Rate limits**: Groq's free tier allows 6,000 tokens per minute. Long contracts are filtered down to their most legally relevant paragraphs before the API call to stay within this limit.
- **Offline fallback**: If Zenodo is unreachable during download, the pipeline automatically generates 50 realistic synthetic contracts using `reportlab` and continues normally.
- **No data in git**: The `data/` folder and `logs/` are excluded from version control via `.gitignore`. Run the pipeline once to populate them locally.