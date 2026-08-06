import operator
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field
from retrieval.hybrid import Chunk


class SubQuestion(BaseModel):
    """
    Represents a targeted sub-query extracted by the planner.

    Attributes:
        text (str): The standalone search query.
        company (Optional[str]): The target ticker (e.g., 'AAPL'), or None if open-ended.
        fiscal_year (Optional[int]): The target fiscal year (e.g., 2024), or None if open-ended.
    """
    text: str
    company: Optional[str] = None
    fiscal_year: Optional[int] = None


class Plan(BaseModel):
    """
    The reasoning plan containing the decomposed queries.

    Attributes:
        sub_questions (list[SubQuestion]): A list of queries to execute sequentially.
    """
    sub_questions: list[SubQuestion]


class Decision(BaseModel):
    """
    The agent's choice of what to do next, made after seeing retrieved passages.

    Attributes:
        action (str): One of 'answer', 'refine', 'broaden', or 'ask_user'.
        reason (str): One sentence explaining the choice, shown in the trace.
        revised_queries (list[SubQuestion]): Rewritten sub-questions, one per
            sub-question in the original order. Carrying the full sub-question
            rather than bare text lets a retry correct the company or year it
            searched, not just the wording. Used by 'refine' and 'broaden'.
        question_for_user (str): The question to put to the user, used by 'ask_user'.
    """
    action: Literal["answer", "refine", "broaden", "ask_user"]
    reason: str = ""
    revised_queries: list[SubQuestion] = Field(default_factory=list)
    question_for_user: str = ""


class AgentState(BaseModel):
    """
    The central state object bridging the LangGraph nodes.

    Attributes:
        user_query (str): The initial query from the user.
        plan (Optional[Plan]): The decomposed execution plan.
        contexts (list[Chunk]): The aggregated context retrieved from the vector store.
        clarifications (Annotated[list[str], operator.add]): The user's replies to
            questions the agent stopped to ask, oldest first. The planner reads
            these alongside the original query.
        clarify_rounds (int): How many times the agent has stopped to ask the user.
        next_action (Optional[str]): The node the plan or decide step chose, read by the router.
        retry_count (int): Integer tracking verification loop iterations.
        final_answer (Optional[str]): The synthesized generation output.
        trace (Annotated[list[str], operator.add]): The concatenated reasoning trace log.
    """
    user_query: str
    plan: Optional[Plan] = None
    contexts: list[Chunk] = Field(default_factory=list)
    clarifications: Annotated[list[str], operator.add] = []
    clarify_rounds: int = 0
    next_action: Optional[str] = None
    retry_count: int = 0
    final_answer: Optional[str] = None
    trace: Annotated[list[str], operator.add] = []