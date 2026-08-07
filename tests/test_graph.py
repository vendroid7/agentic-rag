"""The wiring: that each node's chosen action becomes the edge it should.

`build_agent` normally loads two transformer models and three indices, so all of
that is patched out and stub nodes are put in their place. What is left under
test is the part that matters here — the routing.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from agent.graph import build_agent
from agent.state import AgentState, Plan, SubQuestion

PLAN = Plan(sub_questions=[SubQuestion(text="risk factors", company="AAPL", fiscal_year=2024)])


@contextmanager
def graph_of(plan, clarify, retrieve, decide, answer):
    """
    Compiles the real graph around stub nodes.

    Args:
        plan, clarify, retrieve, decide, answer (callable): Stand-ins that return
            the state updates a test wants, so a chosen path can be forced.

    Yields:
        CompiledGraph: The real wiring, with nothing heavy behind it.
    """
    with patch("agent.graph.SentenceTransformer"), patch("agent.graph.CrossEncoder"), \
         patch("agent.graph.HybridRetriever"), patch("agent.graph.Database"), \
         patch("agent.graph.LLMClient"), \
         patch("agent.graph.make_nodes", return_value=(plan, clarify, retrieve, decide, answer)):
        yield build_agent()


def run(app, query="q"):
    """
    Invokes a compiled graph on its own thread.

    Args:
        app (CompiledGraph): The graph to run.
        query (str): The question to start from.

    Returns:
        dict: The final state.
    """
    return app.invoke(AgentState(user_query=query), {"configurable": {"thread_id": "t"}})


def node(**update):
    """
    Builds a stub node that always returns the same update.

    Args:
        **update: The state update the node should return.

    Returns:
        callable: A node function.
    """
    return lambda s: dict(update)


class TestHappyPath:
    def test_a_resolved_question_runs_straight_through(self):
        with graph_of(
            node(plan=PLAN, trace=["plan"]),
            node(next_action="retrieve", trace=["clarify"]),
            node(contexts=[], trace=["retrieve"]),
            node(next_action="answer", trace=["decide"]),
            node(final_answer="done", trace=["answer"]),
        ) as app:
            out = run(app)
        assert out["final_answer"] == "done"
        assert out["trace"] == ["plan", "clarify", "retrieve", "decide", "answer"]

    def test_the_clean_path_visits_five_nodes(self):
        with graph_of(
            node(plan=PLAN, trace=["plan"]),
            node(next_action="retrieve", trace=["clarify"]),
            node(contexts=[], trace=["retrieve"]),
            node(next_action="answer", trace=["decide"]),
            node(final_answer="done", trace=["answer"]),
        ) as app:
            out = run(app)
        assert len(out["trace"]) == 5


class TestRouteAfterDecide:
    def test_retrieve_sends_it_round_again(self):
        calls = {"n": 0}

        def decide(s):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"next_action": "retrieve", "trace": ["decide:retry"]}
            return {"next_action": "answer", "trace": ["decide:answer"]}

        with graph_of(
            node(plan=PLAN, trace=["plan"]),
            node(next_action="retrieve", trace=["clarify"]),
            node(contexts=[], trace=["retrieve"]),
            decide,
            node(final_answer="done", trace=["answer"]),
        ) as app:
            out = run(app)
        assert calls["n"] == 2
        assert out["trace"].count("retrieve") == 2

    def test_plan_sends_it_back_to_the_planner(self):
        calls = {"n": 0}

        def decide(s):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"next_action": "plan", "trace": ["decide:ask"]}
            return {"next_action": "answer", "trace": ["decide:answer"]}

        with graph_of(
            node(plan=PLAN, trace=["plan"]),
            node(next_action="retrieve", trace=["clarify"]),
            node(contexts=[], trace=["retrieve"]),
            decide,
            node(final_answer="done", trace=["answer"]),
        ) as app:
            out = run(app)
        assert out["trace"].count("plan") == 2

    def test_anything_else_falls_through_to_answer(self):
        with graph_of(
            node(plan=PLAN, trace=["plan"]),
            node(next_action="retrieve", trace=["clarify"]),
            node(contexts=[], trace=["retrieve"]),
            node(next_action=None, trace=["decide"]),
            node(final_answer="done", trace=["answer"]),
        ) as app:
            out = run(app)
        assert out["final_answer"] == "done"


class TestRouteAfterClarify:
    def test_plan_loops_back_before_retrieving(self):
        calls = {"n": 0}

        def clarify(s):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"next_action": "plan", "clarifications": ["Apple 2024"], "trace": ["clarify:ask"]}
            return {"next_action": "retrieve", "trace": ["clarify:ok"]}

        with graph_of(
            node(plan=PLAN, trace=["plan"]),
            clarify,
            node(contexts=[], trace=["retrieve"]),
            node(next_action="answer", trace=["decide"]),
            node(final_answer="done", trace=["answer"]),
        ) as app:
            out = run(app)
        assert out["trace"].count("plan") == 2
        assert out["clarifications"] == ["Apple 2024"]
        assert out["trace"].count("retrieve") == 1


class TestCheckpointing:
    def test_the_graph_is_compiled_with_a_checkpointer(self):
        with graph_of(
            node(plan=PLAN, trace=["plan"]),
            node(next_action="retrieve", trace=["clarify"]),
            node(contexts=[], trace=["retrieve"]),
            node(next_action="answer", trace=["decide"]),
            node(final_answer="done", trace=["answer"]),
        ) as app:
            assert app.checkpointer is not None
            thread = {"configurable": {"thread_id": "t"}}
            app.invoke(AgentState(user_query="q"), thread)
            assert app.get_state(thread).values["final_answer"] == "done"

    def test_a_thread_id_is_required(self):
        with graph_of(
            node(plan=PLAN, trace=["plan"]),
            node(next_action="retrieve", trace=["clarify"]),
            node(contexts=[], trace=["retrieve"]),
            node(next_action="answer", trace=["decide"]),
            node(final_answer="done", trace=["answer"]),
        ) as app:
            with pytest.raises(Exception):
                app.invoke(AgentState(user_query="q"))
