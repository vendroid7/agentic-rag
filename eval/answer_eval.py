"""Measures whether answers are grounded in the passages that were retrieved.

Three scores per question, chosen so that two of them cost nothing:

* Citation validity — every chunk the answer cites must be one the retriever
  actually returned. Pure string matching against the state, so a fabricated
  citation is caught deterministically.
* Context relevance — the mean cross-encoder score of the passages that reached
  the answer. The reranker already computed this during retrieval, so reading it
  back is free.
* Faithfulness — an LLM judge counts how many claims in the answer are supported
  by the passages. This is the only part that costs tokens: one call per question,
  on top of the agent run itself.

Recall is deliberately not measured. That needs a labelled question-to-chunk set
built by reading the filings, which is the expensive half; these three need no
ground truth at all.

Usage:
    PYTHONPATH=src python eval/answer_eval.py
"""

import re
import uuid

from pydantic import BaseModel

from agent.graph import build_agent
from agent.llm import LLMClient
from agent.state import AgentState
from config import config

# Every question names one filing, so the clarify gate stays out of the way and
# each run reaches an answer in a single turn.
QUESTIONS = [
    "What are Apple's risk factors in 2024?",
    "What did Microsoft say about competition in FY2025?",
    "What were Apple's main revenue drivers in 2024?",
    "What legal proceedings did NVIDIA disclose in 2025?",
    "How does Amazon describe its business in 2024?",
    "What market risks did Tesla disclose in FY2025?",
    "What was Microsoft's CEO's exact favorite color in 2024?",
    "What does NVIDIA say about supply chain risk in 2024?",
]


class Faithfulness(BaseModel):
    """
    The judge's verdict on how much of an answer the passages actually support.

    Attributes:
        total_claims (int): How many factual claims the answer makes.
        supported_claims (int): How many of those the passages back up.
        unsupported (list[str]): The claims that are not backed up, quoted briefly.
    """

    total_claims: int
    supported_claims: int
    unsupported: list[str] = []


JUDGE_INSTRUCTIONS = """
You are checking whether an answer is grounded in the passages it was given.

Break the answer into its factual claims, ignoring hedges and pleasantries. For
each, decide whether the passages support it. A claim that the passages do not
contain the information is itself supported — refusing honestly is grounded.

Report total_claims, supported_claims, and quote any unsupported claim briefly.
""".strip()


def cited_chunk_ids(answer: str) -> set[int]:
    """
    Pulls the chunk ids an answer cites out of its citation markers.

    Args:
        answer (str): The generated answer, citing passages as "[chunk 157, ...]".

    Returns:
        set[int]: Every chunk id the answer refers to.
    """
    return {int(n) for n in re.findall(r"\[chunk (\d+)", answer)}


def main():
    """
    Runs every question through the agent and scores the answer it produced.

    Prints a line per question and then the three aggregate scores.
    """
    app = build_agent()
    judge = LLMClient(
        api_key=config.GROQ_API_KEY,
        model=config.MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )

    valid_citations = 0
    scored = 0
    relevance_total = 0.0
    supported_total = 0
    claims_total = 0

    for question in QUESTIONS:
        thread = {"configurable": {"thread_id": uuid.uuid4().hex}}
        state = AgentState.model_validate(app.invoke(AgentState(user_query=question), thread))

        answer = state.final_answer or ""
        retrieved = {c.chunk_id for c in state.contexts}
        cited = cited_chunk_ids(answer)

        invented = cited - retrieved
        valid_citations += not invented

        relevance = sum(c.score for c in state.contexts) / len(state.contexts) if state.contexts else 0.0
        relevance_total += relevance

        passages = "\n\n".join(f"[chunk {c.chunk_id}]\n{c.text}" for c in state.contexts)
        verdict = judge.structured(
            JUDGE_INSTRUCTIONS, f"Answer:\n{answer}\n\nPassages:\n{passages}", Faithfulness
        )
        supported_total += verdict.supported_claims
        claims_total += verdict.total_claims
        scored += 1

        faithful = verdict.supported_claims / verdict.total_claims if verdict.total_claims else 1.0
        print(f"{'ok  ' if not invented else 'BAD '} {question[:52]:<54} "
              f"faithful {faithful:.0%}  relevance {relevance:+.2f}  cites {len(cited)}")
        for claim in verdict.unsupported:
            print(f"       unsupported: {claim[:90]}")

    print(f"\n{scored} questions scored")
    print(f"  citation validity: {valid_citations}/{scored} answers cited only retrieved passages")
    print(f"  faithfulness:      {supported_total}/{claims_total} claims supported by the passages")
    print(f"  context relevance: {relevance_total / scored:+.2f} mean cross-encoder score")


if __name__ == "__main__":
    main()
