# Agentic RAG over SEC 10-K Filings

An interactive, agentic Retrieval-Augmented Generation (RAG) system. Unlike a standard linear pipeline, this system uses a stateful agent reasoning loop powered by LangGraph to dynamically break down questions, detect ambiguity, and verify its own retrievals.

---

## 🌐 Live Demo

The agent is fully deployed and accessible on Streamlit Community Cloud. **API keys are pre-configured in the cloud secrets**, so you can click the link and start interacting with the agent immediately!

👉 **[Try the Live Demo Here](https://sec-filling-chat.streamlit.app/)**

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
Open the `.env` file and add your `GROQ_API_KEY`. **(Do not commit this file to GitHub!)** The `SEC_USER_AGENT` value is only needed if you plan to re-run ingestion in step 3.

*Note: If you prefer not to use a `.env` file, you can simply run the app and enter your API key directly into the secure sidebar input in the UI.*

### 3. Build the Vector Store (Optional)
The pre-built vector database is already included in this repository so you can skip this step and go straight to Step 4! 

If you want to ingest the SEC filings from scratch yourself:
```bash
# Run the ingestion script from the src/ directory
cd src && uv run python -m ingest.run_ingestion && cd ..

# Or ingest a single company for a faster test run
cd src && uv run python -m ingest.run_ingestion --ticker AAPL && cd ..
```

### 4. Run the Agent UI
Launch the Streamlit interface to interact with the agent:
```bash
uv run streamlit run src/app.py
```
Once the UI is live on your localhost, it will prompt you to securely enter your Groq API key in the sidebar. Once entered, you can start interacting with the UI immediately!

### 5. Run the Evaluation (Optional)
```bash
PYTHONPATH=src uv run python eval/planner_eval.py
```

---

## 🧪 Sample Prompts

* *Compare Apple and Tesla risk factors in 2024*
* *What were the main revenue drivers?*
* *What was Microsoft's CEO's exact favorite color in 2024?*

---

## 📂 Codebase Structure

All tunable parameters (LLM configs, retries, etc.) are centralized in `src/config/config.py`.

```text
agentic-rag/
├── .env                        # Your local secrets (API keys)
├── .env.example                # Template for the above
├── pyproject.toml              # Project dependencies (uv)
├── requirements.txt            # Auto-generated requirements for Streamlit Cloud
├── DESIGN.md                   # Architecture, key decisions and trade-offs
├── LIMITATIONS.md              # Known limitations and the reasoning behind them
├── WALKTHROUGH.md              # Line-by-line code tour and five dry runs
├── data/                       # Pre-built FAISS and DuckDB vector store
├── eval/
│   └── planner_eval.py   # 20 labelled questions: extraction + clarify gate
├── src/
│   ├── app.py            # Streamlit UI (Frontend)
│   ├── config/
│   │   └── config.py     # Single source of truth for all parameters
│   ├── agent/
│   │   ├── graph.py      # The LangGraph state machine definition
│   │   ├── llm.py        # Groq API wrapper (Structured & Text)
│   │   ├── nodes.py      # The 5 agent nodes
│   │   └── state.py      # The Pydantic state and decision schemas
│   ├── retrieval/
│   │   ├── database.py   # DuckDB catalog for metadata filtering
│   │   └── hybrid.py     # FAISS (Dense) + BM25 (Sparse) retrieval logic
│   └── ingest/
│       └── ingestion.py  # SEC 10-K parsing (edgartools) and chunking logic
```

---

## 📖 Further Reading

* **[DESIGN.md](DESIGN.md)** — the architecture, why the agent loop is shaped this way, and the decisions worth defending.
* **[LIMITATIONS.md](LIMITATIONS.md)** — what the system does not do, and the trade-off behind each gap.
* **[WALKTHROUGH.md](WALKTHROUGH.md)** — a code tour with five dry runs showing how different questions flow through the graph.

---

*Note: AI assistance was used during development to speed up boilerplate coding and generate comments to improve overall readability.*
