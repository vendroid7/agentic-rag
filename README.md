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

## 🧪 Sample Prompts

These three prompts each exercise a different capability. `Evidence.pdf` contains
screenshots of all three running end to end, including the expanded reasoning trace.

| Prompt | What it demonstrates |
| --- | --- |
| *Compare Apple and Tesla risk factors in 2024* | Multi-hop planning — the query is split into two independently filtered retrievals and synthesised into one grounded comparison. |
| *What were the main revenue drivers?* | The clarify gate — no company or year was given, the catalog matches every filing, so the agent stops and asks before searching. |
| *What was Microsoft's CEO's exact favorite color in 2024?* | Dynamic action selection — the agent chooses `refine`, rewrites its own search, retries, then honestly reports that the filings do not contain the answer. |

---

## 📂 Codebase Structure

All tunable parameters (LLM configs, retries, etc.) are centralized in `src/config/config.py`.

```text
agentic-rag/
├── .env                        # Your local secrets (API keys)
├── .env.example                # Template for the above
├── pyproject.toml              # Project dependencies (uv)
├── requirements.txt            # Auto-generated requirements for Streamlit Cloud
├── Agentic_RAG_PPT.pptx        # Presentation Deck
├── Evidence.pdf                # Screenshots of the three demo prompts, run end to end
├── data/                       # Pre-built FAISS and DuckDB vector store
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

## 🧠 Architecture Highlights

The system is built without heavy abstractions (like LangChain agents), focusing on simple, readable Python functions and explicit control flow.

* **Agentic Routing:** A 5-node LangGraph state machine (`plan` -> `clarify` -> `retrieve` -> `decide` -> `answer`).
* **The Clarify Gate:** The agent explicitly queries the DuckDB catalog to detect vague questions. If a query matches multiple filings, it pauses execution and asks the user to clarify.
* **Dynamic Action Selection:** After retrieving, the `decide_node` chooses one of four courses of action rather than following a fixed retry edge — `answer`, `refine` (rewrite the search wording and try again), `broaden` (drop an over-narrow company/year filter), or `ask_user` (hand the question back). The retry budget is enforced in code, so the model chooses *what* to do while the graph guarantees termination.
* **Constrained by design:** The action space is a closed enum validated by Pydantic rather than open-ended tool use. On a small planning model this buys reliability and a hard ceiling on cost per query, at the cost of extensibility. This is closer to Corrective RAG (CRAG) than to a ReAct agent.
* **LLM Tiering:** Uses a small, lightning-fast model (`llama-3.1-8b`) for structured planning/verifying to save costs, and routes to a massive 70B parameter model (`llama-3.3-70b`) only for the final answer synthesis.

---

*Note: AI assistance was used during development to speed up boilerplate coding and generate comments to improve overall readability.*
