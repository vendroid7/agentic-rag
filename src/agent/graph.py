"""Wires the five nodes into the agent loop and provides a CLI."""

from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer

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

    Returns:
        CompiledGraph: The executable LangGraph application.
    """
    embedder = SentenceTransformer(config.DEFAULT_EMBEDDING_MODEL)
    retriever = HybridRetriever(
        faiss_path=config.FAISS_INDEX_PATH,
        bm25_path=config.BM25_PATH,
        duckdb_path=config.DUCKDB_PATH,
        embedder=embedder,
    )
    database = Database(config.DUCKDB_PATH)
    llm = LLMClient(
        api_key=config.GROQ_API_KEY,
        small_model=config.SMALL_MODEL,
        large_model=config.LARGE_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )

    plan_node, clarify_node, retrieve_node, verify_node, answer_node = make_nodes(
        database, retriever, llm
    )

    def route_after_clarify(state: AgentState) -> str:
        """
        Determines the transition path following the clarify node.

        Args:
            state (AgentState): The active state post-clarification.

        Returns:
            str: The node identifier to transition to ('END' or 'retrieve').
        """
        if state.clarification_message:
            return END
        return "retrieve"

    def route_after_verify(state: AgentState) -> str:
        """
        Determines the transition path following the verify node.

        Evaluates the verification coverage flag and retry threshold to decide
        whether to proceed to answer synthesis or trigger a retrieval retry.

        Args:
            state (AgentState): The active state post-verification.

        Returns:
            str: The node identifier to transition to ('answer' or 'retrieve').
        """
        if state.coverage_ok or state.retry_count >= config.MAX_RETRIES:
            return "answer"
        return "retrieve"

    workflow = StateGraph(AgentState)

    workflow.add_node("plan", plan_node)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("answer", answer_node)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "clarify")
    workflow.add_conditional_edges(
        "clarify", route_after_clarify, {END: END, "retrieve": "retrieve"}
    )
    workflow.add_edge("retrieve", "verify")
    workflow.add_conditional_edges(
        "verify", route_after_verify, {"answer": "answer", "retrieve": "retrieve"}
    )
    workflow.add_edge("answer", END)

    return workflow.compile()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python -m agent.graph "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    print(f"Question: {question}\n")

    app = build_agent()
    result = app.invoke(AgentState(user_query=question))
    state = AgentState.model_validate(result)

    print("Reasoning:")
    for line in state.trace:
        print(f"  {line}")

    if state.clarification_message:
        print(f"\n{state.clarification_message}")
        reply = input("\nYour reply: ").strip()
        if reply:
            combined = f"{question}\nThe user clarified: {reply}"
            result = app.invoke(AgentState(user_query=combined))
            state = AgentState.model_validate(result)
            print("\nReasoning:")
            for line in state.trace:
                print(f"  {line}")
            print(f"\n{state.final_answer}")
    else:
        print(f"\n{state.final_answer}")