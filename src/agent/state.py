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

    Every field is a primitive on purpose: given a schema with a nested model
    reference, the small model tends to echo the schema back instead of filling
    it in.

    Attributes:
        action (str): One of 'answer', 'refine', 'broaden', or 'ask_user'.
        reason (str): One sentence explaining the choice, shown in the trace.
        revised_queries (list[str]): Rewritten search text, one per sub-question
            in the original order. Used by 'refine' and 'broaden'.
        question_for_user (str): The question to put to the user, used by 'ask_user'.
    """
    action: Literal["answer", "refine", "broaden", "ask_user"]
    reason: str = ""
    revised_queries: list[str] = Field(default_factory=list)
    question_for_user: str = ""


class AgentState(BaseModel):
    """
    The central state object bridging the LangGraph nodes.

    Attributes:
        user_query (str): The initial query from the user.
        plan (Optional[Plan]): The decomposed execution plan.
        contexts (list[Chunk]): The aggregated context retrieved from the vector store.
        clarification_message (Optional[str]): System prompt pausing execution for user input.
        next_action (Optional[str]): The action the decide node chose, read by the router.
        retry_count (int): Integer tracking verification loop iterations.
        final_answer (Optional[str]): The synthesized generation output.
        trace (Annotated[list[str], operator.add]): The concatenated reasoning trace log.
    """
    user_query: str
    plan: Optional[Plan] = None
    contexts: list[Chunk] = Field(default_factory=list)
    clarification_message: Optional[str] = None
    next_action: Optional[str] = None
    retry_count: int = 0
    final_answer: Optional[str] = None
    trace: Annotated[list[str], operator.add] = []