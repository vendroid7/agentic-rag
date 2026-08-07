"""Fakes for the three things the nodes talk to.

The nodes are the part worth testing and the only part that is slow to run for
real, so the catalog, the retriever and the model are all stood in for here.
Every test then runs offline, in milliseconds, with no API key.
"""

from unittest.mock import MagicMock

import pytest

from agent.nodes import make_nodes
from agent.state import Plan, SubQuestion
from retrieval.database import Target
from retrieval.hybrid import Chunk

CATALOG = [
    Target(company="Apple Inc.", ticker="AAPL", fiscal_year=2024),
    Target(company="Apple Inc.", ticker="AAPL", fiscal_year=2025),
    Target(company="Tesla, Inc.", ticker="TSLA", fiscal_year=2024),
]


def chunk(chunk_id=1, text="Apple faces supplier concentration risk.", company="Apple Inc."):
    """
    Builds a retrieved passage.

    Args:
        chunk_id (int): The id the answer will cite.
        text (str): The passage body.
        company (str): The company the passage belongs to.

    Returns:
        Chunk: A passage shaped like one the retriever returns.
    """
    return Chunk(
        chunk_id=chunk_id, text=text, company=company, ticker="AAPL",
        fiscal_year=2024, item_section="Item 1A",
        source_url="https://sec.gov/x", score=1.0, found_by=["dense"],
    )


@pytest.fixture
def database():
    """
    A catalog that resolves to whatever a test tells it to.

    Returns:
        MagicMock: Stands in for `Database`. `resolve.return_value` is what the
            clarify gate will see, so a test sets it to [] for "not in corpus",
            one Target for "resolved", or several for "ambiguous".
    """
    db = MagicMock()
    db.get_all.return_value = CATALOG
    db.resolve.return_value = [CATALOG[0]]
    return db


@pytest.fixture
def retriever():
    """
    A retriever that always returns one passage per call.

    Returns:
        MagicMock: Stands in for `HybridRetriever`.
    """
    r = MagicMock()
    r.search.return_value = [chunk()]
    return r


@pytest.fixture
def llm():
    """
    A model whose replies each test sets explicitly.

    Returns:
        MagicMock: Stands in for `LLMClient`. `structured` covers plan and
            decide, `text` covers the answer.
    """
    m = MagicMock()
    m.structured.return_value = Plan(sub_questions=[SubQuestion(text="risk factors", company="AAPL", fiscal_year=2024)])
    m.text.return_value = "Apple faces supplier risk [chunk 1, Apple Inc. FY2024, Item 1A]."
    return m


@pytest.fixture
def nodes(database, retriever, llm):
    """
    The five nodes, wired to the fakes above.

    Returns:
        tuple: (plan_node, clarify_node, retrieve_node, decide_node, answer_node)
    """
    return make_nodes(database, retriever, llm)
