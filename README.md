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

---

## 📊 Evaluation

Twenty labelled questions covering entity extraction and the clarify gate. The gate is a deterministic DuckDB lookup, so whether it stops to ask is decided entirely by the ticker and fiscal year the planner extracted — one run therefore measures planner accuracy *and* interactivity together, with no labelled chunks and no LLM judge.

```bash
PYTHONPATH=src uv run python eval/planner_eval.py
```

```text
20/20 cases passed
  over-asking:  0 of 10 clear questions were needlessly clarified
  under-asking: 0 of 10 under-specified questions were searched anyway
```

The two failure directions are reported separately because they cost different things: clarifying a question that was already clear is an annoyance, while searching an under-specified one produces a confident wrong answer. Ten of the twenty cases exist only to catch over-asking — an agent that clarifies reflexively would pass a "does it clarify?" spot-check and fail here.

The harness runs the real nodes rather than reimplementing the gate's rules, so it cannot drift from the code it measures.

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

## ⚖️ Known Limitations

* **No conversational memory.** State lives for one question and the clarification that resolves it, so a question gets the same answer whether it is the first of the session or the hundredth — but follow-ups like *"what about Microsoft?"* are planned against those words alone.
* **Passages are replaced on each retry, not accumulated.** A chunk found on the first attempt can be lost if the reworded query no longer surfaces it.
* **The retry budget can cut off a hard question.** After `MAX_RETRIES` the agent must answer with what it holds — this is what guarantees termination.
* **Narrative text only.** Questions turning on figures in financial tables are answered from surrounding commentary rather than the tables themselves.
* **Small corpus.** 5 tickers × 2 fiscal years, so most of EDGAR is out of scope and refusing cleanly is a common correct outcome.
* **Grounding is unmeasured.** Routing is evaluated above; whether every claim traces to a cited passage was checked by hand, not scored.

---

*Note: AI assistance was used during development to speed up boilerplate coding and generate comments to improve overall readability.*
