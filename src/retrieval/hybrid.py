"""Hybrid retrieval — dense + sparse, union merged."""

import pickle
from pathlib import Path

import duckdb
import faiss
import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class Chunk(BaseModel):
    """
    Represents a discrete text segment retrieved from the vector store.

    Attributes:
        chunk_id (int): The absolute index mapped across all persistence layers.
        text (str): The raw token string of the segment.
        company (str): The corporate entity name.
        ticker (str): The stock ticker symbol.
        fiscal_year (int): The financial reporting year.
        item_section (str): The SEC 10-K section header.
        source_url (str): The EDGAR origin URL.
        score (float): The dense retrieval cosine similarity score.
        found_by (list[str]): The retrieval arms that surfaced this chunk ('dense', 'sparse').
    """
    chunk_id: int
    text: str
    company: str
    ticker: str
    fiscal_year: int
    item_section: str
    source_url: str
    score: float = 0.0
    found_by: list[str] = Field(default_factory=list)


class HybridRetriever:

    def __init__(self, faiss_path: Path, bm25_path: Path, duckdb_path: Path, embedder: SentenceTransformer):
        """
        Initializes the retrieval indices and embedding infrastructure.

        Args:
            faiss_path (Path): Filepath to the serialized FAISS vector index.
            bm25_path (Path): Filepath to the serialized BM25 sparse index.
            duckdb_path (Path): Filepath to the DuckDB relational metadata catalog.
            embedder (SentenceTransformer): The model used for dense semantic encoding.
        """
        self.index = faiss.read_index(str(faiss_path))
        with open(bm25_path, "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)
        self.duckdb_path = str(duckdb_path)
        self.embedder = embedder

    def search(self, query: str, company: str = None, fiscal_year: int = None, item_section: str = None, k: int = 6) -> list[Chunk]:
        """
        Executes a constrained hybrid search pipeline.

        Args:
            query (str): The semantic natural language search query.
            company (str, optional): Metadata constraint for the ticker.
            fiscal_year (int, optional): Metadata constraint for the reporting year.
            item_section (str, optional): Metadata constraint for the 10-K section.
            k (int): The maximum number of terminal chunks to return. Defaults to 6.

        Returns:
            list[Chunk]: The highest-scoring combined chunk payloads.
        """
        with duckdb.connect(self.duckdb_path, read_only=True) as conn:

            sql = "SELECT chunk_id FROM chunks WHERE 1=1"
            params = []
            if company is not None:
                sql += " AND (company = ? OR ticker = ?)"
                params.extend([company, company])
            if fiscal_year is not None:
                sql += " AND fiscal_year = ?"
                params.append(fiscal_year)
            if item_section is not None:
                sql += " AND item_section = ?"
                params.append(item_section)

            ids = {r[0] for r in conn.execute(sql, params).fetchall()}
            if not ids:
                return []

            dense_ids, dense_scores = self._dense(query, ids, k)
            sparse_ids = self._sparse(query, ids, k)

            merged = dense_ids | sparse_ids
            if not merged:
                return []

            ph = ", ".join("?" for _ in merged)
            rows = conn.execute(
                f"SELECT chunk_id, text, company, ticker, fiscal_year, item_section, source_url "
                f"FROM chunks WHERE chunk_id IN ({ph})",
                list(merged),
            ).fetchall()

            chunks = [
                Chunk(chunk_id=r[0], text=r[1], company=r[2], ticker=r[3],
                      fiscal_year=r[4], item_section=r[5], source_url=r[6])
                for r in rows
            ]

            for c in chunks:
                if c.chunk_id in dense_ids:
                    c.found_by.append("dense")
                    c.score = dense_scores.get(c.chunk_id, 0.0)
                if c.chunk_id in sparse_ids:
                    c.found_by.append("sparse")

            chunks.sort(key=lambda c: c.score, reverse=True)
            return chunks[:k]

    def _dense(self, query: str, candidate_ids: set, k: int) -> tuple[set, dict]:
        """
        Executes dense vector similarity search via FAISS.

        Args:
            query (str): The semantic search query.
            candidate_ids (set): Pre-filtered valid chunk IDs.
            k (int): Result limit.

        Returns:
            tuple[set, dict]: The set of retrieved IDs and a mapping of ID to cosine score.
        """
        vec = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        selector = faiss.IDSelectorBatch(np.array(sorted(candidate_ids), dtype=np.int64))
        params = faiss.SearchParameters(sel=selector)

        scores, ids = self.index.search(vec, min(k, len(candidate_ids)), params=params)

        results = {}
        for id, score in zip(ids[0], scores[0]):
            if id >= 0:
                results[int(id)] = float(score)

        return set(results.keys()), results

    def _sparse(self, query: str, candidate_ids: set, k: int) -> set:
        """
        Executes sparse keyword matching via BM25.

        Args:
            query (str): The lexical search query.
            candidate_ids (set): Pre-filtered valid chunk IDs.
            k (int): Result limit.

        Returns:
            set: The set of retrieved chunk IDs.
        """
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        scored = [(i, s) for i, s in enumerate(scores) if i in candidate_ids and s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return {i for i, _ in scored[:k]}
