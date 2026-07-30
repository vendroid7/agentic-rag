# Agentic RAG over SEC 10-K Filings

An interactive, agentic Retrieval-Augmented Generation (RAG) system. Unlike a standard linear pipeline, this system uses a stateful agent reasoning loop powered by LangGraph to dynamically break down questions, detect ambiguity, and verify its own retrievals.

---

## 🚀 Setup & Run Instructions

### 1. Install Dependencies
This project uses `uv` for lightning-fast Python dependency management.
```bash
uv sync
```

### 2. Environment Variables
Copy the example environment file:
```bash
cp .env.example .env
```
Open the `.env` file and add your `GROQ_API_KEY`. **(Do not commit this file to GitHub!)**

*Note: If you prefer not to use a `.env` file, you can simply run the app and enter your API key directly into the secure sidebar input in the UI.*

### 3. Build the Vector Store (Optional)
The pre-built vector database is already included in this repository so you can skip this step and go straight to Step 4! 

If you want to ingest the SEC filings from scratch yourself:
```bash
# Run the ingestion script from the src/ directory
cd src && uv run python -m ingest.ingestion && cd ..
```

### 4. Run the Agent UI
Launch the Streamlit interface to interact with the agent:
```bash
uv run streamlit run src/app.py
```
Once the UI is live on your localhost, it will prompt you to securely enter your Groq API key in the sidebar. Once entered, you can start interacting with the UI immediately!

---

## 📂 Codebase Structure

All tunable parameters (LLM configs, retries, etc.) are centralized in `src/config/config.py`.

```text
agentic-rag/
├── .env                        # Your local secrets (API keys)
├── pyproject.toml              # Project dependencies (uv)
├── Agentic_RAG_PPT.pptx        # Presentation Deck
├── data/                       # Pre-built FAISS and DuckDB vector store
├── src/
│   ├── app.py            # Streamlit UI (Frontend)
│   ├── config/
│   │   └── config.py     # Single source of truth for all parameters
│   ├── agent/
│   │   ├── graph.py      # The LangGraph state machine definition
│   │   ├── llm.py        # Groq API wrapper (Structured & Text)
│   │   ├── nodes.py      # The 5 agent nodes
│   │   └── state.py      # The Pydantic state schema
│   ├── retrieval/
│   │   ├── database.py   # DuckDB catalog for metadata filtering
│   │   └── hybrid.py     # FAISS (Dense) + BM25 (Sparse) retrieval logic
│   └── ingest/
│       └── ingestion.py  # SEC 10-K parsing (edgartools) and chunking logic
```

---

## 🧠 Architecture Highlights

The system is built without heavy abstractions (like LangChain agents), focusing on simple, readable Python functions and explicit control flow.

* **Agentic Routing:** A 5-node LangGraph state machine (`plan` -> `clarify` -> `retrieve` -> `verify` -> `answer`).
* **The Clarify Gate:** The agent explicitly queries the DuckDB catalog to detect vague questions. If a query matches multiple filings, it pauses execution and asks the user to clarify.
* **Reflexive Self-Correction:** The `verify_node` actively grades retrieved chunks. If they lack the raw facts needed, it forces the agent into a retry loop to fetch better context.
* **LLM Tiering:** Uses a small, lightning-fast model (`llama-3.1-8b`) for structured planning/verifying to save costs, and routes to a massive 70B parameter model (`llama-3.3-70b`) only for the final answer synthesis.

---

*Note: AI assistance was used during development to speed up boilerplate coding and generate comments to improve overall readability.*
