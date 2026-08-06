"""Streamlit chat interface for the Agentic RAG system."""

import uuid

import streamlit as st
from langgraph.types import Command

from agent.graph import build_agent
from agent.state import AgentState
from config import config
from retrieval.database import Database


@st.cache_resource
def load_app(api_key: str):
    """Build the agent and database once per server process."""
    original_key = config.GROQ_API_KEY
    config.GROQ_API_KEY = api_key
    app = build_agent()
    config.GROQ_API_KEY = original_key
    database = Database(config.DUCKDB_PATH)
    return app, database


def render_trace(trace_list: list) -> str:
    """Format the reasoning trace for display."""
    return "\n".join(trace_list) if trace_list else "Thinking..."


def main():
    """
    Executes the main Streamlit application loop.

    This function initializes the web interface, manages conversation state,
    and streams the agent's reasoning trace dynamically to the UI.
    """
    st.set_page_config(page_title="Agentic RAG | SEC Insights", page_icon="🏦", layout="wide")
    
    st.markdown("""
        <style>
        /* Premium UI Aesthetics */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1 {
            background: -webkit-linear-gradient(45deg, #00B4D8, #4CAF50);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800 !important;
        }
        div[data-testid="stChatMessage"] {
            border-radius: 12px !important;
            padding: 10px !important;
            margin-bottom: 15px !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        div[data-testid="stSidebar"] {
            background-color: #11151c !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🏦 Agentic RAG | SEC Insights")
    st.markdown("*An intelligent, self-correcting retrieval agent powered by LangGraph.*")
    st.divider()

    api_key = config.GROQ_API_KEY

    with st.sidebar:
        if not api_key:
            api_key = st.text_input("🔑 Enter Groq API Key", type="password", help="Get your free key at console.groq.com")
            if not api_key:
                st.warning("Please enter your Groq API Key to continue.")
                st.stop()
        else:
            st.success("✅ API Key loaded securely")
            
        st.divider()

    app, database = load_app(api_key)
    catalog = database.get_all()

    if len(catalog) == 0:
        st.error(
            "Nothing has been ingested yet. Build the corpus first using the ingest script."
        )
        return

    if "history" not in st.session_state:
        welcome_text = "Welcome to the SEC 10-K RAG Agent! 👋\n\n**I currently have access to the following filings:**\n"
        for filing in catalog:
            welcome_text += f"- {filing.company} ({filing.ticker}) FY{filing.fiscal_year}\n"
        
        welcome_text += "\n**Here are a few things you can ask me:**\n"
        welcome_text += "- *Compare Apple and Tesla risk factors in 2024*\n"
        welcome_text += "- *What were the main revenue drivers?*\n"
        welcome_text += "- *What was Microsoft's CEO's exact favorite color in 2024?*\n"
        welcome_text += "- *What are Apple's risk factors in 2024?* then *what about Microsoft?*\n"
        welcome_text += "- *What are Apple's risk factors in 2024?* then *and 2025?*\n"
        
        st.session_state["history"] = [{"role": "assistant", "content": welcome_text}]

    # Holds the thread of a run that stopped to ask something, so the next message
    # typed resumes it instead of starting a new one. None means no run is waiting.
    st.session_state.setdefault("suspended_thread", None)

    # One entry per answered question, oldest first, capped at MAX_HISTORY_TURNS.
    # This is what lets "what about Microsoft?" know what it is following up on.
    st.session_state.setdefault("turns", [])

    with st.sidebar:
        st.subheader("Ingested Corpus")
        st.caption(f"{len(catalog)} filings indexed")
        for filing in catalog:
            st.text(f"{filing.company} ({filing.ticker}) FY{filing.fiscal_year}")
        
        st.subheader("🧠 Memory")
        st.caption(
            f"Remembering {len(st.session_state.turns)} of the last "
            f"{config.MAX_HISTORY_TURNS} questions, so follow-ups know what they refer to."
        )
        if st.button("🧹 Clear memory", use_container_width=True):
            st.session_state.turns = []
            # A run waiting on a clarification belongs to the conversation being
            # forgotten, so drop it too rather than resume into it later.
            st.session_state.suspended_thread = None
            st.rerun()

        st.subheader("💡 Suggested Prompts")
        st.caption("Compare Apple and Tesla risk factors in 2024")
        st.caption("What were the main revenue drivers?")
        st.caption("What was Microsoft's CEO's exact favorite color in 2024?")
        st.caption("What are Apple's risk factors in 2024? → what about Microsoft?")
        st.caption("What are Apple's risk factors in 2024? → and 2025?")

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("trace"):
                with st.expander("Reasoning"):
                    st.code(turn["trace"], language=None)

    question = st.chat_input("Ask about the SEC filings...")
    if not question:
        return

    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.suspended_thread:
        thread = st.session_state.suspended_thread
        turn = Command(resume=question)
    else:
        thread = {"configurable": {"thread_id": uuid.uuid4().hex}}
        turn = AgentState(user_query=question, history=st.session_state.turns)

    with st.chat_message("assistant"):
        panel = st.empty()
        try:
            with st.spinner("Working..."):
                for snapshot in app.stream(turn, thread, stream_mode="values"):
                    final_snapshot = snapshot
                    panel.code(render_trace(final_snapshot.get("trace", [])), language=None)
        except Exception as exc:
            # Usually a Groq rate limit. The thread keeps its saved state, so a
            # question the agent had already paused on can still be resumed.
            panel.empty()
            st.error(f"The agent could not finish this question: {exc}")
            st.stop()

        panel.empty()

        # A run that stopped to ask has its question waiting on the saved state.
        pending = app.get_state(thread).interrupts
        st.session_state.suspended_thread = thread if pending else None
        reply = pending[0].value if pending else final_snapshot.get("final_answer")

        if not pending:
            # Only a question that reached an answer becomes a turn. A reply to a
            # clarification belongs to the question that prompted it, and the
            # original wording is on the state whichever way we got here.
            answered = AgentState.model_validate(final_snapshot)
            targets = ", ".join(
                f"{sq.company or 'any company'}/{sq.fiscal_year or 'any year'}"
                for sq in (answered.plan.sub_questions if answered.plan else [])
            )
            st.session_state.turns = (
                st.session_state.turns + [f'"{answered.user_query}" ({targets})']
            )[-config.MAX_HISTORY_TURNS:]

        st.markdown(reply)
        with st.expander("Reasoning", expanded=bool(pending)):
            st.code(render_trace(final_snapshot.get("trace", [])), language=None)

    st.session_state.history.append(
        {"role": "assistant", "content": reply, "trace": render_trace(final_snapshot.get("trace", []))}
    )


if __name__ == "__main__":
    main()
