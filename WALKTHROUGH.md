# Code Walkthrough & Demo Guide

This document breaks down the agent logic and includes five "dry runs" showing exactly how different questions flow through the graph. This is your study guide for defending the architecture.

---

## 1. The State (`agent/state.py`)
This file defines what data is passed between your nodes. It uses Pydantic for strict type checking.

```python
# Defines a single search query + its metadata filters
class SubQuestion(BaseModel):
    text: str
    company: Optional[str] = None
    fiscal_year: Optional[int] = None

# A wrapper around a list of sub-questions
class Plan(BaseModel):
    sub_questions: list[SubQuestion]

# What the decide node returns — the agent's choice of what to do next
class Decision(BaseModel):
    action: Literal["answer", "refine", "broaden", "ask_user"]
    reason: str = ""
    # Full sub-questions, not bare strings, so a retry can correct the
    # company or year it searched — not just the wording.
    revised_queries: list[SubQuestion] = Field(default_factory=list)
    question_for_user: str = ""

# The main State object passed to every node
class AgentState(BaseModel):
    user_query: str  # The original question
    plan: Optional[Plan] = None  # Populated by plan_node
    contexts: list[Chunk] = Field(default_factory=list)  # Populated by retrieve_node

    # Replies the user gave to questions the agent stopped to ask.
    # operator.add so a second clarification adds to the first.
    clarifications: Annotated[list[str], operator.add] = []
    clarify_rounds: int = 0    # Bounded by MAX_CLARIFICATIONS
    next_action: Optional[str] = None  # Set by clarify/decide, read by the routers
    retry_count: int = 0       # Increments on refine/broaden
    final_answer: Optional[str] = None # Populated by answer_node

    # Annotated with operator.add tells LangGraph to CONCATENATE
    # this list across nodes instead of overwriting it.
    trace: Annotated[list[str], operator.add] = []
```

## 2. The Nodes (`agent/nodes.py`)
These are the pure Python functions that take the `AgentState` and return a dictionary of what they changed.

1. **`plan_node`**: Uses Groq's JSON mode to force the LLM to return data matching our `Plan` schema. Runs again after every clarification, planning against the original question *plus* whatever the user has since told us.
2. **`clarify_node`**: The Gatekeeper, and pure Python — it queries DuckDB using the extracted company and year. No LLM call.
    - 0 matches: suspends and tells the user what the corpus actually holds.
    - \>1 match with a missing filter: suspends and asks which filing was meant.
    - Exactly 1 match: proceeds to retrieve.
3. **`retrieve_node`**: Runs the hybrid search (FAISS + BM25) per sub-question, widening `k` on each retry.
4. **`decide_node`**: The only node with agency. Reads the passages and picks one of four actions — `answer`, `refine`, `broaden`, or `ask_user`. The retry budget is enforced here in code, so the model chooses *what* to do while the graph guarantees termination.
5. **`answer_node`**: Grounded synthesis over the retrieved passages, citing every claim.

**Suspending vs. ending.** When `clarify_node` or `decide_node` needs the user, it calls LangGraph's `interrupt()`. This pauses the run with its state intact rather than ending it. The graph is compiled with a checkpointer, so the user's reply resumes the same run — a question and its clarification stay one exchange with one reasoning trace.

## 3. The Graph (`agent/graph.py`)
This file wires the nodes together using traffic cops (routing functions). Note that neither router makes a decision — the node already made it, and the router only translates it into an edge.

```python
# TRAFFIC COP 1: What happens after the Clarify Gate?
def route_after_clarify(state: AgentState) -> str:
    # The gate sets "plan" only after the user answered a question,
    # so the new information gets planned against.
    return "plan" if state.next_action == "plan" else "retrieve"

# TRAFFIC COP 2: What happens after the agent decides?
def route_after_decide(state: AgentState) -> str:
    # "retrieve" is refine/broaden; "plan" is a fresh clarification.
    if state.next_action in ("retrieve", "plan"):
        return state.next_action
    return "answer"
```

---

## 4. Dry Runs (How questions flow)

### Scenario A: The Happy Path
**User:** *"What are Apple's risk factors in 2024?"*

1. **Plan Node:** Extracts `company="AAPL"`, `fiscal_year=2024`.
2. **Clarify Node:** Checks DuckDB. Finds exactly 1 match (Apple Inc. FY2024). Sets `next_action="retrieve"`.
3. **Graph Router:** `route_after_clarify` routes to `retrieve`.
4. **Retrieve Node:** Runs hybrid search, finds chunks.
5. **Decide Node:** Reads the passages, chooses `answer` with a one-line reason.
6. **Answer Node:** Synthesizes the final text with citations.
7. **END.**

### Scenario B: Complex Planning
**User:** *"Compare Apple and Tesla risk factors in 2024."*

1. **Plan Node:** Splits this into TWO sub-questions:
   - SQ1: `company="AAPL"`, `fiscal_year=2024`
   - SQ2: `company="TSLA"`, `fiscal_year=2024`
2. **Clarify Node:** Loops through both. Both return exactly 1 match in DuckDB.
3. **Retrieve Node:** Runs two separate searches, combining all chunks into `contexts`.
4. **Decide Node:** Chooses `answer` — the facts for *both* companies being present is enough; it does not expect a pre-written comparison in the passages.
5. **Answer Node:** Writes the comparison.

### Scenario C: Vague (The Clarify Gate)
**User:** *"What were the main revenue drivers?"*

1. **Plan Node:** Extracts the text, but `company=None` and `fiscal_year=None`.
2. **Clarify Node:** Checks DuckDB with no filters. DuckDB returns ALL filings. Because matches > 1, the node calls `interrupt("Which filing did you mean?")`.
3. **Agent Suspends.** The run pauses with its state saved. Streamlit shows the question.
4. **User Replies:** *"Apple 2024"*
5. **Graph Resumes:** The reply is recorded in `clarifications` and the router sends the agent back to `plan` — *inside the same run*. The planner now sees the original question plus the clarification and extracts `AAPL`/`2024`. The gate gets 1 match and proceeds to retrieve.

The trace the user sees is continuous:

```
Plan: "main revenue drivers" -> any / any year
Clarify: 10 matches — asking user
Clarify: User replied — Apple 2024
Plan: "main revenue drivers" -> AAPL / 2024
Clarify: All targets resolved — searching
```

### Scenario D: Out of Corpus
**User:** *"What are Microsoft's risk factors in 1999?"*

1. **Plan Node:** Extracts `company="MSFT"`, `fiscal_year=1999`.
2. **Clarify Node:** Checks DuckDB. Finds 0 matches (we only indexed 2024/2025). Interrupts with what the corpus *does* hold.
3. **Agent Suspends** instead of searching blindly. Refusing cleanly is a common correct outcome on a 5-company corpus, not an edge case.

### Scenario E: The Self-Correcting Loop
**User:** *"What were Apple's main revenue drivers?"* (after clarifying to FY2024)

1. **Retrieve Node:** Searches `"main revenue drivers"` filtered to AAPL/2024. Returns 4 chunks about revenue *recognition* — related wording, wrong content.
2. **Decide Node:** Chooses `refine`, reasoning that the passages cover revenue policy rather than drivers, and returns a rewritten sub-question: `"major revenue streams"`, keeping the AAPL/2024 filters.
3. **Graph Router:** `route_after_decide` routes back to `retrieve`. `retry_count` increments, and `k` widens.
4. **Loop repeats** until the agent chooses `answer` or the retry budget is spent — at which point the code, not the model, forces the answer.

This is the difference between a pipeline and an agent: the same graph produced a 3-node path in Scenario A and a 7-node path here, because the model chose differently at each step.

---

## 5. Evaluation (`eval/planner_eval.py`)

Twenty labelled questions covering entity extraction and the clarify gate. Because the gate is a deterministic catalog lookup, one run measures planner accuracy *and* "does it ask at the right moment" together — no labelled chunks, no LLM judge, seconds to run.

```bash
PYTHONPATH=src python eval/planner_eval.py
```

The two failure directions are reported separately, because they cost different things: clarifying a question that was already clear is an annoyance, while searching an under-specified one produces a confident wrong answer. Currently 20/20, with 0/10 over-asking.
