"""Document ingestion pipeline for SEC 10-K filings.

Fetches filings from EDGAR, parses text on Item boundaries, generates vector
embeddings, and persists synchronized FAISS, DuckDB, and BM25 index artifacts.
"""

import pickle
from pathlib import Path

import duckdb
import faiss
import numpy as np
import pandas as pd
from edgar import Company, set_identity
from langchain_text_splitters import TextSplitter
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class ChunkRecord(BaseModel):
    """Structured record representing a processed SEC filing document chunk.

    Attributes:
        chunk_id (int): Zero-indexed positional ID shared across DuckDB, FAISS, and BM25.
        company (str): Full legal corporate entity name.
        ticker (str): Equity ticker symbol in uppercase.
        fiscal_year (int): Four-digit reporting fiscal year.
        item_section (str): SEC filing item section heading code.
        source_url (str): Canonical SEC EDGAR source document URL.
        text (str): Extracted narrative chunk text.
    """

    chunk_id: int
    company: str
    ticker: str
    fiscal_year: int
    item_section: str
    source_url: str
    text: str


class IndexStore:
    """Manager for aligned DuckDB, FAISS, and BM25 persistence artifacts.

    Attributes:
        faiss_path (Path): File path destination for vector index file.
        bm25_path (Path): File path destination for sparse index pickle.
        conn (duckdb.DuckDBPyConnection): Connection handle to DuckDB database.
    """

    def __init__(self, duckdb_path: Path, faiss_path: Path, bm25_path: Path) -> None:
        """Initializes storage directories, connections, and database schema.

        Args:
            duckdb_path (Path): File path destination for metadata catalog database.
            faiss_path (Path): File path destination for vector index file.
            bm25_path (Path): File path destination for sparse index pickle.
        """
        self.faiss_path = faiss_path
        self.bm25_path = bm25_path

        duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(duckdb_path))

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY,
                company TEXT NOT NULL,
                ticker TEXT NOT NULL,
                fiscal_year INTEGER NOT NULL,
                item_section TEXT NOT NULL,
                source_url TEXT NOT NULL,
                text TEXT NOT NULL
            )
            """
        )
        self.conn.execute("DELETE FROM chunks")

    def persist(self, chunks: list[ChunkRecord], embeddings: np.ndarray) -> None:
        """Writes chunk records and embeddings to DuckDB, FAISS, and BM25 artifacts.

        Args:
            chunks (list[ChunkRecord]): Sequentially ordered chunk data models.
            embeddings (np.ndarray): Dense embedding matrix aligned with chunks.
        """
        if not chunks:
            return

        # 1. DuckDB metadata catalog
        df = pd.DataFrame([c.model_dump() for c in chunks])
        self.conn.execute("INSERT INTO chunks SELECT * FROM df")

        # 2. FAISS dense index
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(embeddings.astype(np.float32))
        faiss.write_index(faiss_index, str(self.faiss_path))

        # 3. BM25 sparse index
        tokenized_corpus = [c.text.lower().split() for c in chunks]
        bm25_model = BM25Okapi(tokenized_corpus)
        with open(self.bm25_path, "wb") as f:
            pickle.dump(bm25_model, f)


class IngestionPipeline:
    """Orchestrator executing document fetching, parsing, and storage.

    Attributes:
        embedder (SentenceTransformer): Loaded vector embedding model.
        store (IndexStore): Multi-backend persistence manager.
        splitter (TextSplitter): Token-aware chunker sized to the embedder.
        target_items (dict[str, str]): SEC section items to parse.
        min_section_chars (int): Shortest Item worth chunking at all.
        min_chunk_chars (int): Shortest chunk worth indexing.
    """

    def __init__(
        self,
        embedder: SentenceTransformer,
        store: IndexStore,
        splitter: TextSplitter,
        target_items: dict[str, str],
        min_section_chars: int,
        min_chunk_chars: int,
    ) -> None:
        """Initializes pipeline with collaborator instances and tunables.

        Args:
            embedder (SentenceTransformer): Vector embedding model instance.
            store (IndexStore): Storage layer manager instance.
            splitter (TextSplitter): Chunker whose budget matches the embedder.
            target_items (dict[str, str]): SEC section items to parse.
            min_section_chars (int): Shortest Item worth chunking at all.
            min_chunk_chars (int): Shortest chunk worth indexing.
        """
        self.embedder = embedder
        self.store = store
        self.splitter = splitter
        self.target_items = target_items
        self.min_section_chars = min_section_chars
        self.min_chunk_chars = min_chunk_chars

    def run(self, tickers: list[str], target_years: list[int], user_agent: str) -> None:
        """Executes full ingestion pipeline for given companies and years.

        Args:
            tickers (list[str]): List of stock ticker symbols to fetch.
            target_years (list[int]): Target fiscal reporting years.
            user_agent (str): User agent identity string for SEC EDGAR access.
        """
        set_identity(user_agent)
        all_chunks: list[ChunkRecord] = []
        chunk_id = 0

        for ticker in tickers:
            company = Company(ticker)
            filings = company.get_filings(form="10-K")

            for filing in filings:
                year = filing.filing_date.year
                if year not in target_years:
                    continue

                tenk = filing.obj()
                for item_key in self.target_items.keys():
                    try:
                        # edgartools returns the section as plain text already
                        section_text = tenk[item_key]
                        if not section_text or len(section_text.strip()) < self.min_section_chars:
                            continue

                        for sub_text in self.splitter.split_text(section_text):
                            if len(sub_text) < self.min_chunk_chars:
                                continue

                            all_chunks.append(
                                ChunkRecord(
                                    chunk_id=chunk_id,
                                    company=company.name,
                                    ticker=ticker,
                                    fiscal_year=year,
                                    item_section=item_key,
                                    source_url=filing.url,
                                    text=sub_text,
                                )
                            )
                            chunk_id += 1
                    except Exception:
                        continue  # Skip missing or malformed sections

        if all_chunks:
            texts = [c.text for c in all_chunks]
            # Normalization ensures exact cosine similarity with IndexFlatIP
            embeddings = self.embedder.encode(texts, normalize_embeddings=True)
            self.store.persist(all_chunks, embeddings)