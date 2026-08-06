"""Wires the five nodes into the agent loop and provides a CLI."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer, CrossEncoder

from agent.state import AgentState
from agent.llm import LLMClient
from agent.nodes import make_nodes
from config import config
from retrieval.database import Database
from retrieval.hybrid import HybridRetriever


def build_agent():
    """
    Constructs and compiles the LangGraph state machine for the RAG agent.

    Initializes all external dependencies (embedding model, retrieval indices,
    and LLM client) and injects them into the node factories. Defines the
    control flow edges and conditional routing logic.

    The graph is compiled with a checkpointer because clarifying suspends the run
    rather than ending it: the saved state is what a reply resumes into, so a
    question and its clarification stay one run with one reasoning trace.

    Returns:
        CompiledGraph: The executable LangGraph application. Callers must pass a
            config carrying a `thread_id`, which is the key the state is saved under.
    """
    embedder = SentenceTransformer(config.DEFAULT_EMBEDDING_MODEL)
    reranker = CrossEncoder(config.CROSS_ENCODER_MODEL)
    retriever = HybridRetriever(
        faiss_path=config.FAISS_INDEX_PATH,
        bm25_path=config.BM25_PATH,
        duckdb_path=config.DUCKDB_PATH,
        embedder=embedder,
        reranker=reranker,
    )
    database = Database(config.DUCKDB_PATH)
    llm = LLMClient(
        api_key=config.GROQ_API_KEY,
        model=config.MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )

    plan_node, clarify_node, retrieve_node, decide_node, answer_node = make_nodes(
        database, retriever, llm
    )

    def route_after_clarify(state: AgentState) -> str:
        """
        Determines the transition path following the clarify node.

        Args:
            state (AgentState): The active state post-clarification.

        Returns:
            str: The node identifier to transition to ('plan' when the user has
                just answered a question, otherwise 'retrieve').
        """
        return "plan" if state.next_action == "plan" else "retrieve"

    def route_after_decide(state: AgentState) -> str:
        """
        Dispatches on the action the decide node chose.

        The decision itself is made by the model inside decide_node; this router
        only translates that choice into a graph transition.

        Args:
            state (AgentState): The active state post-decision.

        Returns:
            str: The node identifier to transition to ('answer', 'retrieve' or 'plan').
        """
        if state.next_action in ("retrieve", "plan"):
            return state.next_action
        return "answer"

    workflow = StateGraph(AgentState)

    workflow.add_node("plan", plan_node)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("decide", decide_node)
    workflow.add_node("answer", answer_node)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "clarify")
    workflow.add_conditional_edges(
        "clarify", route_after_clarify, {"plan": "plan", "retrieve": "retrieve"}
    )
    workflow.add_edge("retrieve", "decide")
    workflow.add_conditional_edges(
        "decide",
        route_after_decide,
        {"answer": "answer", "retrieve": "retrieve", "plan": "plan"},
    )
    workflow.add_edge("answer", END)

    return workflow.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    import sys
    import uuid

    from langgraph.types import Command

    if len(sys.argv) < 2:
        print('Usage: python -m agent.graph "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    print(f"Question: {question}\n")

    app = build_agent()
    thread = {"configurable": {"thread_id": uuid.uuid4().hex}}

    # The first turn starts the run; each later one resumes the suspended graph
    # with the user's answer, so the whole exchange is a single run.
    turn = AgentState(user_query=question)
    while True:
        result = app.invoke(turn, thread)
        pending = result.get("__interrupt__")
        if not pending:
            break
        turn = Command(resume=input(f"\n{pending[0].value}\n\nYour reply: ").strip())

    state = AgentState.model_validate(result)

    print("\nReasoning:")
    for line in state.trace:
        print(f"  {line}")
    print(f"\n{state.final_answer}")