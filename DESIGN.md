# Design & Architecture

The design decisions behind the agent, and the reasoning that justifies each one.
Written to be presentable: each section is roughly one slide.

---

## The problem with a pipeline

A standard RAG pipeline runs a fixed sequence — retrieve, then generate. It cannot
notice that a question was ambiguous, cannot tell that its search missed, and cannot
do anything about either. On SEC filings both failures are common: *"what were the
main revenue drivers?"* names no company and no year, and 10-K prose rarely uses the
words a user would.

This system is built around the two decisions a pipeline cannot make: **should I ask
before searching**, and **is what I found good enough**.

---

## The loop

```
                    ┌──────── refine / broaden ────────┐
                    ▼                                  │
  plan ──> clarify ──> retrieve ──> decide ────────────┤
             │                        │                │
             │                        ├─> answer ──> END
             └────────────────────────┴─> ask user (suspend, then re-plan)
```

Five nodes, built without heavy abstractions — no LangChain agents, just readable
Python functions and explicit control flow.

| Node | Job | LLM? |
|---|---|---|
| `plan` | Split the question into sub-questions, extract ticker and fiscal year | Yes |
| `clarify` | Check the extracted targets against the DuckDB catalog | **No** |
| `retrieve` | Hybrid FAISS + BM25 search per sub-question, cross-encoder rerank | No |
| `decide` | Choose what to do next | Yes |
| `answer` | Grounded synthesis with citations | Yes |

---

## Decision 1 — one node holds all the agency

`decide` is the only place a choice is made. `plan`, `retrieve` and `answer` are
deterministic steps, and both routing functions are pure translators: the node has
already decided, the router only turns that into an edge.

This is what makes the agency auditable. A reviewer can read one function and see
every path the system can take, instead of hunting for `if` statements spread across
a pipeline.

**Trade-off:** less flexible than letting any node redirect the flow. Worth it for
being able to state, in one sentence, where the agent's judgement lives.

---

## Decision 2 — a closed action set, not open tool use

After retrieving, `decide` picks exactly one of four actions:

| Action | Meaning |
|---|---|
| `answer` | The passages contain the facts needed |
| `refine` | Right filing, wrong wording — rewrite the query and retry |
| `broaden` | Wrong company or year filter — drop or correct it and retry |
| `ask_user` | No amount of searching will help; hand it back to the person |

A closed enum validated by Pydantic, rather than open-ended tool calling. Free-form
tool use demos better and defends worse: with four actions, every possible path is
enumerable and every failure mode is describable.

This is closer to Corrective RAG (CRAG) than to a ReAct agent.

**The dynamism is real but bounded.** The same graph produces a three-node path for
a clear question and a seven-node path for a hard one, because the model chooses
differently at each step — but it can never choose something unanticipated.

**Termination is enforced in code, not in the prompt.** After `MAX_RETRIES` the
router forces `answer`. A prompt instruction is advisory; a router is not.

---

## Decision 3 — the clarify gate is deterministic

The gate makes no LLM call. It takes the ticker and year the planner extracted and
asks DuckDB what matches:

* **0 matches** → the corpus cannot answer this; say what it does hold
* **>1 match with a missing filter** → genuinely ambiguous; ask which filing
* **exactly 1 match** → proceed

Ambiguity is therefore not a judgement call. It is a catalog lookup, which makes it
fast, free, and reproducible — the same question always gets the same decision.

The heuristic behind it: **ask only when the ambiguity changes what would be
retrieved.** Company, fiscal year, section. Ambiguity that does not move the
retrieval target is settled by retrieving. That line is what separates useful
interactivity from an agent that interrogates the user before every answer.

---

## Decision 4 — clarifying suspends the run, it does not end it

Both places that need the user call LangGraph's `interrupt()`, and the graph is
compiled with a checkpointer. The reply resumes the *same* run: it is recorded in
`clarifications`, and the agent returns to `plan` to re-plan against the original
question plus what it just learned.

The earlier design ended the graph and restarted it with the reply pasted onto the
original question. That re-planned from scratch, discarded any retrieval already
paid for, and split one exchange across two runs with two separate traces.

The user now sees one continuous thread:

```
Plan: "main revenue drivers" -> any / any year
Clarify: 10 matches — asking user
Clarify: User replied — Apple 2024
Plan: "main revenue drivers" -> AAPL / 2024
Clarify: All targets resolved — searching
```

`MAX_CLARIFICATIONS` bounds the new `plan → clarify → plan` cycle, so the agent
cannot interrogate the user indefinitely.

---

## Decision 5 — one model, arrived at by testing

Every call runs on `llama-3.3-70b`.

The original design tiered: `llama-3.1-8b` for planning and deciding, the 70B for
answer synthesis. That is the right split for a *pipeline*, where generation is the
hard part. For an agent it is inverted — the hard part is the decisions, and writing
from four reranked passages is the easy job.

Two findings killed the tier:

* The 8B **echoes a nested schema back instead of filling it in**. The `Decision`
  schema had been flattened to bare strings to work around this, which cost the
  agent the ability to correct the company or year on a retry.
* Asked to write the final answer, the 8B **repeats itself** — the same risk factor
  listed four times, and a letter sign-off.

With planning and deciding moved up and synthesis staying, nothing was left for a
second tier to do. `LLMClient` takes one model.

**Trade-off:** higher latency and more rate-limit exposure than a tiered setup, in
exchange for correct schemas and non-repetitive answers.

---

## Decision 6 — measure the routing, not just the answer

`eval/planner_eval.py` — 20 labelled questions, each tagged with the tickers and
years the planner should extract and whether the gate should fire.

Because the gate is deterministic, whether it stops is decided entirely by what the
planner extracted. **One run therefore measures planner accuracy and "does it ask at
the right moment" together** — no labelled chunks, no LLM judge, seconds to run.

```
20/20 cases passed
  over-asking:  0 of 10 clear questions were needlessly clarified
  under-asking: 0 of 10 under-specified questions were searched anyway
```

Ten of the twenty cases exist only to catch **over-asking** — an agent that
clarifies reflexively would score perfectly on a "does it clarify?" check and fail
here. The two directions are reported separately because they cost different things:
asking when the question was already clear is an annoyance, while searching an
under-specified one produces a confident wrong answer.

The harness runs the real nodes through a two-node graph rather than reimplementing
the gate's rules, so it cannot drift from the code it measures. It was
mutation-tested — inverted gate expectations, wrong ticker, wrong sub-question count
— and catches each, so the score reflects behaviour rather than a vacuous assertion.

---

## Retrieval

* **Hybrid.** FAISS (dense, `bge-small-en-v1.5`) for meaning, BM25 (sparse) for the
  exact statutory phrasing filings use. Each arm fetches 20 candidates.
* **Cross-encoder rerank.** The union is judged by `ms-marco-MiniLM-L-6-v2`, and the
  top 4 reach the answer. Precision matters more than recall here, because the
  answer node cites what it is given.
* **Metadata filtering first.** DuckDB narrows to the right filing *before* the
  vector search, so a question about Apple FY2024 cannot retrieve Tesla passages.
* **Chunking in embedding-model tokens**, not characters, so no chunk silently
  overflows the 512-token window and gets truncated at encode time.

---

## What was deliberately left out

* **Open-ended tool calling** — four actions cover the problem and stay defensible.
* **Multi-agent decomposition** — nothing in the task requires it; it multiplies
  failure modes.
* **A separate reflection/critique node** — `decide` already rejects weak evidence
  by looping back. A critic would be a second thing to explain for the same
  behaviour.
* **Conversational memory** — see [LIMITATIONS.md](LIMITATIONS.md); the statelessness
  is what keeps the clarify gate predictable.
