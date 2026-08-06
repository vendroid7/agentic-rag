"""Measures what the planner extracts and whether the clarify gate fires correctly.

The gate itself is deterministic — it is a DuckDB catalog lookup — so whether it
stops to ask is decided entirely by the company and fiscal year the planner pulled
out of the question. That makes one cheap run cover both planner accuracy and the
"does it ask at the right moment" behaviour, with no labelled chunks and no LLM
judge: every assertion below is something we already know the answer to.

Retrieval is deliberately not exercised, so the whole set runs in seconds.

Usage:
    PYTHONPATH=src python eval/planner_eval.py
"""

import uuid
from dataclasses import dataclass

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import StateGraph, END

from agent.llm import LLMClient
from agent.nodes import make_nodes
from agent.state import AgentState
from config import config
from retrieval.database import Database


@dataclass
class Case:
    """
    One question and what the agent is expected to make of it.

    The three extraction fields default to None, meaning "not checked". That is
    used for questions aimed at filings we never ingested, where the only thing
    worth asserting is that the agent stops rather than answers.

    Attributes:
        question (str): The question as a user would type it.
        clarifies (bool): Whether the clarify gate is expected to stop and ask.
        companies (list[str] | None): Tickers the planner should extract, sorted.
        years (list[int] | None): Fiscal years the planner should extract, sorted.
        sub_questions (int | None): How many sub-questions the plan should hold.
    """

    question: str
    clarifies: bool
    companies: list[str] | None = None
    years: list[int] | None = None
    sub_questions: int | None = None


# Ten questions that name a filing we hold. These are the over-asking guard: an
# agent that clarifies reflexively fails here, and no spot-check would catch it.
ANSWERABLE = [
    Case("What are Apple's risk factors in 2024?", False, ["AAPL"], [2024], 1),
    Case("Compare Apple and Tesla risk factors in 2024", False, ["AAPL", "TSLA"], [2024], 2),
    Case("What did Microsoft say about competition in FY2025?", False, ["MSFT"], [2025], 1),
    Case("appl risk factors fy24", False, ["AAPL"], [2024], 1),
    Case("nvidia 2025 legal proceedings", False, ["NVDA"], [2025], 1),
    Case("Amazon's business description for fiscal 2024", False, ["AMZN"], [2024], 1),
    Case("Tesla FY2025 management discussion", False, ["TSLA"], [2025], 1),
    Case("How did NVIDIA and AMZN describe their risks in 2024?", False, ["AMZN", "NVDA"], [2024], 2),
    Case("Microsoft 2024 vs Microsoft 2025 risk factors", False, ["MSFT"], [2024, 2025], 2),
    Case("What is Apple's market risk disclosure in 2024?", False, ["AAPL"], [2024], 1),
]

# Under-specified questions. The corpus can answer them, but not until the user
# says which filing they meant.
AMBIGUOUS = [
    Case("What were the main revenue drivers?", True, [], [], 1),
    Case("What are the risk factors?", True, [], [], 1),
    Case("Tell me about legal proceedings", True, [], [], 1),
    Case("What did they say about competition?", True, [], [], 1),
    Case("Apple's risk factors", True, ["AAPL"], [], 1),
    Case("What were the biggest risks in 2024?", True, [], [2024], 1),
]

# Filings we never ingested. Extraction is not asserted, because there is no right
# answer for a ticker outside the catalog — only stopping is correct.
OUT_OF_CORPUS = [
    Case("What are Netflix's risk factors in 2024?", True),
    Case("What were Apple's risk factors in 2019?", True),
    Case("Google 2024 revenue drivers", True),
    Case("Tesla's 2010 risk factors", True),
]

CASES = ANSWERABLE + AMBIGUOUS + OUT_OF_CORPUS


def build_planner():
    """
    Compiles a two-node graph of just the planner and the clarify gate.

    Running the real nodes rather than re-implementing the gate's rules keeps the
    eval honest: if the gate changes, this changes with it. The retriever is never
    reached by either node, so it is not constructed.

    Returns:
        CompiledGraph: A graph that plans, checks the catalog, and stops.
    """
    database = Database(config.DUCKDB_PATH)
    llm = LLMClient(
        api_key=config.GROQ_API_KEY,
        model=config.MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )
    plan_node, clarify_node = make_nodes(database, None, llm)[:2]

    workflow = StateGraph(AgentState)
    workflow.add_node("plan", plan_node)
    workflow.add_node("clarify", clarify_node)
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "clarify")
    workflow.add_edge("clarify", END)

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("agent.state", "Plan"), ("agent.state", "SubQuestion")]
    )
    return workflow.compile(checkpointer=MemorySaver(serde=serde))


def run_case(planner, case: Case) -> tuple[bool, list[str]]:
    """
    Runs one question through the planner and gate and grades the result.

    Args:
        planner (CompiledGraph): The graph returned by `build_planner`.
        case (Case): The question and its expectations.

    Returns:
        tuple[bool, list[str]]: Whether every expectation held, and one line
            describing each that did not.
    """
    thread = {"configurable": {"thread_id": uuid.uuid4().hex}}
    result = planner.invoke(AgentState(user_query=case.question), thread)

    clarified = bool(result.get("__interrupt__"))
    sub_questions = result["plan"].sub_questions
    companies = sorted({sq.company for sq in sub_questions if sq.company})
    years = sorted({sq.fiscal_year for sq in sub_questions if sq.fiscal_year})

    failures = []
    if clarified != case.clarifies:
        failures.append("asked when it should have searched" if clarified else "searched when it should have asked")
    if case.companies is not None and companies != sorted(case.companies):
        failures.append(f"companies {companies} != {sorted(case.companies)}")
    if case.years is not None and years != sorted(case.years):
        failures.append(f"years {years} != {sorted(case.years)}")
    if case.sub_questions is not None and len(sub_questions) != case.sub_questions:
        failures.append(f"{len(sub_questions)} sub-question(s) != {case.sub_questions}")

    return not failures, failures


def main():
    """
    Runs every case and prints per-case results followed by a summary.

    The summary separates the two ways the gate can be wrong, because they carry
    different costs: asking when the question was already clear is an annoyance,
    while searching an under-specified question produces a confident wrong answer.
    """
    planner = build_planner()

    passed = 0
    asked_when_clear = 0
    searched_when_vague = 0

    for case in CASES:
        ok, failures = run_case(planner, case)
        passed += ok
        asked_when_clear += "asked when it should have searched" in failures
        searched_when_vague += "searched when it should have asked" in failures

        print(f"{'PASS' if ok else 'FAIL'}  {case.question}")
        for failure in failures:
            print(f"        {failure}")

    total = len(CASES)
    print(f"\n{passed}/{total} cases passed")
    print(f"  asked when the question was already clear: {asked_when_clear}/{len(ANSWERABLE)}")
    print(f"  searched when it should have asked:        {searched_when_vague}/{len(AMBIGUOUS) + len(OUT_OF_CORPUS)}")


if __name__ == "__main__":
    main()
