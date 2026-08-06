"""Single source of truth for every tunable value in the project.

Values that are free to live in version control are defined here as module
constants. The one value that is not, the SEC identity string, is read from a
local `.env` file (see `.env.example`) so that no personal detail is committed.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root, two folders up from this file (agentic_rag_th/src/config/config.py).
ROOT = Path(__file__).resolve().parents[2]

load_dotenv(ROOT / ".env")

# --- SEC EDGAR access ---
# EDGAR rejects unidentified traffic, so it requires a "Name email" string on
# every request. Kept out of the repository because it is a personal detail.
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")

# --- LLM (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# One model for every call. llama-3.1-8b was tried for the cheaper steps, but it
# echoes a nested schema back instead of filling it in and repeats itself when
# writing answers, so nothing was left for a second tier to do.
MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 2048

# --- Agent Settings ---
MAX_RETRIES = 2  # How many times the Verifier can reject chunks before forcing an answer
MAX_SUBQUESTIONS = 4  # Maximum number of sub-questions the planner can generate
MAX_CLARIFICATIONS = 1  # How many times the agent may stop and ask the user before it has to proceed

# --- Corpus to ingest ---
DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA", "TSLA"]
DEFAULT_YEARS = [2024, 2025]

# The 10-K item sections worth indexing, mapped to the display names the agent
# cites and filters on. Anything outside this set (exhibits, signatures,
# boilerplate) carries no answerable narrative and is skipped.
TARGET_ITEMS = {
    "Item 1": "Business",
    "Item 1A": "Risk Factors",
    "Item 3": "Legal Proceedings",
    "Item 5": "Market for Registrant's Common Equity",
    "Item 7": "Management's Discussion and Analysis",
    "Item 7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "Item 8": "Financial Statements",
}

# --- Embeddings ---
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# --- Chunking ---
# Chunks are measured in the embedding model's own tokens, not characters, so
# that no chunk can silently overflow its 512-token window and be truncated at
# encode time. The margin below 512 leaves room for the [CLS]/[SEP] markers.
MAX_CHUNK_TOKENS = 480

# A filing's argument often straddles a chunk boundary, so each chunk repeats
# the tail of the one before it and no sentence is left without its lead-in.
CHUNK_OVERLAP_TOKENS = 64

# Tried in order: keep paragraphs whole, then lines, then sentences, then
# words. The empty string is the fallback that splits mid-word, and it is what
# guarantees an unbroken wall of text still gets cut down to size.
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# An Item that parses to less than this is a cross-reference stub ("see Note 7")
# rather than narrative, and there is nothing in it to answer a question with.
MIN_SECTION_CHARS = 100

# Splitting on headings can leave a heading stranded as its own chunk ("22",
# "OPERATIONAL RISKS"). Such fragments carry no meaning and only add noise.
MIN_CHUNK_CHARS = 120

# --- Retrieval ---
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Each arm (dense + sparse) fetches this many candidates independently.
# The union is typically 30-40 unique chunks for the reranker to judge.
TOP_K_PER_ARM = 20

# How many chunks survive the cross-encoder rerank and reach the answer.
FINAL_K = 4

# --- Index artifacts ---
# All three derive from one directory so they cannot be pointed at different
# places and drift out of sync with each other.
INDEX_DIR = ROOT / "data" / "index"
FAISS_INDEX_PATH = INDEX_DIR / "vectors.faiss"
DUCKDB_PATH = INDEX_DIR / "meta.duckdb"
BM25_PATH = INDEX_DIR / "bm25.pkl"
