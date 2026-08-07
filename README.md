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

## ⚙️ How it works

Five nodes, wired as a LangGraph state machine. Two of them make a decision; the other three are fixed steps.

* **plan** — splits the question into up to four sub-questions, each with a ticker and a fiscal year. A missing one stays `null` on purpose.
* **clarify** — asks DuckDB what matches. No rows, or more than one with a `null` filter, and it stops and asks the user. The run is checkpointed, so the reply resumes it rather than starting over.
* **retrieve** — DuckDB narrows to the right filing, then FAISS and BM25 each fetch 20 candidates and a cross-encoder keeps the best four per sub-question.
* **decide** — the only node the model steers. It picks `answer`, `refine`, `broaden` or `ask_user`. The retry budget is checked in code before the model is called, so the loop always ends.
* **answer** — writes from the passages only and cites each claim.

The UI remembers the last five questions and the filing each resolved to, so follow-ups work; a sidebar button clears it.

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

### Answer grounding

A second harness scores the answers themselves. Citation validity and context relevance cost nothing — the cited chunk ids are checked against what retrieval actually returned, and the reranker already scored the passages. Only faithfulness needs a judge.

```bash
PYTHONPATH=src uv run python eval/answer_eval.py
```

```text
8 questions scored
  citation validity: 8/8 answers cited only retrieved passages
  faithfulness:      45/48 claims supported by the passages
```

No answer invented a citation. Two of the three unsupported claims came from one question about Tesla's market risk, where the answer drifted into key-personnel risk that the retrieved passages did not contain — the kind of drift a citation check alone would miss, since the citations themselves were valid.

### Retrieval

The reranker's own scores cost nothing to read back, so retrieval can be checked without any model call. Across the same eight questions, **three in four of the passages kept were found by both the dense and the sparse arm** — the remaining quarter came from one arm alone, which is what the second arm is there for.

Mean reranker confidence in the passages it kept is **41%**, and that figure follows what the agent does next: 85–92% on questions it answers straight away, and near zero on the two it either rewrites its search for or declines to answer. Low confidence is the signal the loop acts on, not a defect.

Recall is not measured: that needs a labelled question-to-chunk set built by reading the filings, whereas everything above needs no ground truth. So these numbers show that answers are grounded in what was retrieved — not that retrieval found everything it should have.

---

## 🧪 Sample Prompts

* *Compare Apple and Tesla risk factors in 2024*
* *What were the main revenue drivers?*
* *What was Microsoft's CEO's exact favorite color in 2024?*
* *What are Apple's risk factors in 2024?* then *what about Microsoft?*
* *What are Apple's risk factors in 2024?* then *and 2025?*

The last two are follow-ups. Clear the memory from the sidebar and ask *what about Microsoft?* on its own to see the difference — with nothing to follow up on, the agent has no year and stops to ask.

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
│   ├── planner_eval.py   # 20 labelled questions: extraction + clarify gate
│   └── answer_eval.py    # citations, faithfulness and reranker confidence
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

*Note: AI assistance was used during development to speed up boilerplate coding and generate comments to improve overall readability.*
