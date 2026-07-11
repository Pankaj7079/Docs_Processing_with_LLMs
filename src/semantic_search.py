# Semantic search over extracted contract clauses using local sentence embeddings

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ClauseSearchEngine:
    """
    Indexes extracted contract clauses and serves semantic search queries.
    Embeddings are generated locally using sentence-transformers.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        logger.info("Embedding model ready.")

    def index_csv(self, csv_path: Path) -> int:
        """
        Loads extracted clauses from the pipeline output CSV and builds a search index.
        Returns the number of clauses indexed.
        """
        if not csv_path.exists():
            logger.error(f"CSV not found at '{csv_path}'. Run the pipeline first.")
            return 0

        logger.info(f"Indexing clauses from '{csv_path}'...")
        df = pd.read_csv(csv_path)

        self.documents = []
        clause_cols = ["termination_clause", "confidentiality_clause", "liability_clause"]

        for _, row in df.iterrows():
            contract_id = row.get("contract_id", "Unknown")
            for col in clause_cols:
                text = str(row.get(col, "")).strip()
                if not text or text.upper() in {"NOT FOUND", "EXTRACTION FAILED", "NAN"}:
                    continue
                self.documents.append({
                    "contract_id": contract_id,
                    "clause_type": col.replace("_", " ").title(),
                    "text": text,
                })

        if not self.documents:
            logger.warning("No valid clauses found to index.")
            return 0

        texts = [d["text"] for d in self.documents]
        logger.info(f"Generating embeddings for {len(texts)} clauses...")
        self.embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        logger.info("Indexing complete.")
        return len(self.documents)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Returns the top_k most semantically similar clauses for a given query.
        Each result is a (document_dict, similarity_score) tuple.
        """
        if self.embeddings is None or not self.documents:
            logger.warning("Index is empty — call index_csv() first.")
            return []

        query_vec = self.model.encode(query, convert_to_numpy=True)

        # Cosine similarity
        dot = np.dot(self.embeddings, query_vec)
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8
        scores = dot / norms

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], float(scores[i])) for i in top_indices]
