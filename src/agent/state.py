import operator
from typing import Annotated, Optional
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


class Coverage(BaseModel):
    """
    The verifier's binary judgement on passage relevance.

    Attributes:
        covered (bool): Indicates if the context sufficiently answers the query.
        gap (str): A description of missing information if covered is False.
    """
    covered: bool
    gap: str = ""


class AgentState(BaseModel):
    """
    The central state object bridging the LangGraph nodes.

    Attributes:
        user_query (str): The initial query from the user.
        plan (Optional[Plan]): The decomposed execution plan.
        contexts (list[Chunk]): The aggregated context retrieved from the vector store.
        clarification_message (Optional[str]): System prompt pausing execution for user input.
        coverage_ok (bool): Boolean flag denoting successful retrieval validation.
        retry_count (int): Integer tracking verification loop iterations.
        final_answer (Optional[str]): The synthesized generation output.
        trace (Annotated[list[str], operator.add]): The concatenated reasoning trace log.
    """
    user_query: str
    plan: Optional[Plan] = None
    contexts: list[Chunk] = Field(default_factory=list)
    clarification_message: Optional[str] = None
    coverage_ok: bool = False
    retry_count: int = 0
    final_answer: Optional[str] = None
    trace: Annotated[list[str], operator.add] = []