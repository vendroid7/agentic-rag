"""Entry point for executing the SEC 10-K document ingestion pipeline.

Initializes configuration, dependencies, and storage layers, then triggers
the download and indexing process. Can be run for a specific ticker during
development or the full default corpus for the final demo.
"""

import argparse

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import config
from ingest.ingestion import IndexStore, IngestionPipeline


def main() -> None:
    """Parses arguments and executes the ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Build the Agentic RAG index.")
    parser.add_argument(
        "--ticker",
        type=str,
        help="Optional single ticker to ingest for fast testing (e.g., AAPL).",
    )
    args = parser.parse_args()

    # Determine which tickers to fetch based on CLI arguments
    tickers_to_fetch = [args.ticker] if args.ticker else config.DEFAULT_TICKERS

    print(f"Initializing embedding model: {config.DEFAULT_EMBEDDING_MODEL}")
    embedder = SentenceTransformer(config.DEFAULT_EMBEDDING_MODEL)

    # Measuring chunks with the embedder's own tokenizer is what makes the
    # token budget a guarantee: a chunk can no longer overflow the model's
    # window and lose its tail to silent truncation at encode time.
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        embedder.tokenizer,
        chunk_size=config.MAX_CHUNK_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
        separators=config.CHUNK_SEPARATORS,
    )

    print("Initializing local storage layers (DuckDB, FAISS, BM25)...")
    store = IndexStore(
        duckdb_path=config.DUCKDB_PATH,
        faiss_path=config.FAISS_INDEX_PATH,
        bm25_path=config.BM25_PATH,
    )

    pipeline = IngestionPipeline(
        embedder=embedder,
        store=store,
        splitter=splitter,
        target_items=config.TARGET_ITEMS,
        min_section_chars=config.MIN_SECTION_CHARS,
        min_chunk_chars=config.MIN_CHUNK_CHARS,
    )

    print(f"Starting ingestion for {tickers_to_fetch}...")
    pipeline.run(
        tickers=tickers_to_fetch,
        target_years=config.DEFAULT_YEARS,
        user_agent=config.SEC_USER_AGENT,
    )
    
    print("✅ Ingestion complete. Indexes are ready for the agent.")


if __name__ == "__main__":
    main()