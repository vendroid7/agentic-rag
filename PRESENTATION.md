# Presentation Script — 9 Minutes

This script is designed to be delivered alongside a live demo. Timings are deliberate, ensuring half the time is spent showing the agent in action (planning and clarifying) as requested by the take-home brief.

---

## 0:00 — 1:00 · The Problem and Our Core Commitment

A traditional RAG pipeline always tries to answer. If you ask a standard pipeline "What were the main revenue drivers?" against a corpus of ten filings, it retrieves *something* and answers confidently about a company you never named. It has no mechanism to notice that the question is underspecified.

So, the core design commitment of this system: **The agent must ask clarifying questions when a query does not resolve to exactly one filing, and that decision must be driven by hard data, not LLM heuristics.**

Everything else—the hybrid search, the state graph—exists to support that one commitment.

## 1:00 — 2:30 · Architecture overview

*Show the high-level architecture diagram or code structure.*

- **Ingestion:** EDGAR → Parse on Item boundaries → Chunk → Embed. We use three stores: FAISS for dense vectors, BM25 for sparse keywords, and DuckDB for metadata and text.
- **The Single ID Invariant:** All three stores share the exact same `chunk_id`. This is the load-bearing invariant that makes citations trustworthy. A dense hit and a sparse hit for the same ID are provably the same passage.
- **The Catalog (`database.py`):** The set of answerable targets is just a `SELECT DISTINCT` query. It's tiny by construction.
- **The Agent Loop:** A LangGraph state machine with distinct nodes for planning, clarifying, retrieving, verifying, and answering.

## 2:30 — 4:00 · Demo 1: The Agent Plans

*Live demo: `Compare the risk factors Apple and Tesla reported in fiscal 2024.`*

*Narrate the trace while it runs:*
- The planner breaks this into two sub-questions, each with its own strict metadata filter (Company, Year, Item).
- The Clarify Gate resolves both against the DuckDB catalog. Both match exactly one filing. It proceeds silently.
- **Crucially:** We do two *separate* hybrid searches against two *different* filings. We don't do one muddled search over both and hope the embedding model pulls equally from both companies.
- The answer synthesizes the results, and every claim carries a traceable chunk ID, company, and year.

## 4:00 — 6:00 · Demo 2: The Agent Asks

*Live demo: `What were the main revenue drivers?`*

- The planner creates one sub-question with an empty filter. It matches all ten filings in our corpus.
- **The agent stops.** It retrieves nothing. It cannot guess.
- The Clarify Gate looks at the multiple matches and composes a question offering the real filings from the catalog. It isn't generating a hallucinated list of options.

*Reply: `NVIDIA, fiscal 2025`.*
- The agent re-plans over the original question *and* the reply. It resolves to one filing, does one retrieval pass, and answers. (Takes ~10-12s).
- *Honest limitation:* The gate catches *metadata* ambiguity (which company/year). It cannot see *semantic* ambiguity ("performance"—financial or environmental?).

## 6:00 — 7:30 · Retrieval Design: Lean and Honest

Let's look at `hybrid.py`. The retrieval is built to be fast, understandable, and exact.

1. **Filter Inside the Search:** We use DuckDB to find the allowed chunk IDs first, then pass them to FAISS via `IDSelectorBatch`. If we searched globally and filtered after, a top-20 search might leave us with only 1 usable hit from the target filing.
2. **Dense + Sparse Union:** FAISS catches semantic similarity; BM25 catches exact tickers and jargon. We run both.
3. **Cross-Encoder Reranking:** We take the top 20 hits from both the dense and sparse searches and union them. We then pass this combined pool of ~40 chunks through a `ms-marco-MiniLM` Cross-Encoder. This neural model scores the exact semantic relationship between the query and the chunk text, allowing us to drop the irrelevant hits and return only the absolute top 4 best chunks to the LLM.

## 7:30 — 8:30 · Trade-offs and Deliberate Cuts

Every non-obvious choice has a cost:
- **No LLM in the Clarify Gate:** LLMs are bad at knowing when they don't know something. Our gate is a deterministic SQL count. It's falsifiable.
- **Re-planning instead of `interrupt()`:** When the user clarifies, we re-plan from scratch rather than resuming a suspended graph thread. This costs one extra small-model call, but avoids the need for complex checkpointers and state management.
- **No Answer Cache:** We cut this to focus on the agentic behavior. But the design for it is solid: the cache key must include the *resolved filters*, not just the query, so a cached answer for Apple is never served for Microsoft.

## 8:30 — 9:00 · Scaling to a Million Filings

If we scale this from 10 filings to 1,000,000, the infrastructure changes, but the reasoning layer doesn't.

- FAISS becomes Qdrant; BM25 becomes OpenSearch; DuckDB becomes Postgres.
- But the **Catalog stays tiny** because it grows with distinct filings, not chunks.
- The resolved metadata filters from the Clarify Gate become **shard routing keys**. Once the agent knows the company and year, it routes the search to one specific shard instead of searching a million documents.

The exact same metadata that makes the agent interactive today is what makes it scale tomorrow.

---

## Anticipated Follow-up Questions

**Q: Why not use an LLM to decide when a query is ambiguous?**
A: It's unfalsifiable and inconsistent. An LLM might think "revenue" is specific enough and try to answer. A SQL count against our catalog guarantees we ask if there are multiple targets, and guarantees we *don't* ask if the corpus only has one company anyway.

**Q: Why union the hybrid results instead of fusing the scores?**
A: BM25 and cosine similarity are on completely different scales. Any formula to combine them requires a magic weighting number that can't be justified from first principles. Taking the union ensures we don't drop a perfect keyword match just because its dense score was average.

**Q: How do you handle the notoriously difficult SEC HTML formatting?**
A: Instead of writing fragile, custom regex scrapers that break every time the SEC updates their layout, we rely on `edgartools`. It's a specialized, open-source library that parses 10-K sections cleanly. As a senior engineer, I allocate my complexity budget to the agent's logic, not to reinventing the wheel on HTML parsing.
