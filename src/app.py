"""Streamlit chat interface for the Agentic RAG system."""

import streamlit as st

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
        welcome_text += "- *Compare Apple and Tesla risk factors in 2024* (Demonstrates Multi-Hop Planning)\n"
        welcome_text += "- *What were the main revenue drivers?* (Demonstrates the Clarify Gate)\n"
        welcome_text += "- *What was Microsoft's CEO's exact favorite color in 2024?* (Demonstrates Context Verification)\n"
        
        st.session_state["history"] = [{"role": "assistant", "content": welcome_text}]

    st.session_state.setdefault("answering", None)

    with st.sidebar:
        st.subheader("Ingested Corpus")
        st.caption(f"{len(catalog)} filings indexed")
        for filing in catalog:
            st.text(f"{filing.company} ({filing.ticker}) FY{filing.fiscal_year}")
        
        st.subheader("💡 Suggested Prompts")
        st.caption("Compare Apple and Tesla risk factors in 2024")
        st.caption("What were the main revenue drivers?")
        st.caption("What was Microsoft's CEO's exact favorite color in 2024?")

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

    if st.session_state.answering:
        combined_query = f"{st.session_state.answering}\nThe user clarified: {question}"
        initial_state = AgentState(user_query=combined_query)
    else:
        initial_state = AgentState(user_query=question)

    with st.chat_message("assistant"):
        panel = st.empty()
        with st.spinner("Working..."):
            for snapshot in app.stream(initial_state, stream_mode="values"):
                final_snapshot = snapshot
                panel.code(render_trace(final_snapshot.get("trace", [])), language=None)
        
        panel.empty()

        reply = final_snapshot.get("clarification_message") or final_snapshot.get("final_answer")
        
        st.session_state.answering = initial_state.user_query if final_snapshot.get("clarification_message") else None

        st.markdown(reply)
        with st.expander("Reasoning", expanded=bool(final_snapshot.get("clarification_message"))):
            st.code(render_trace(final_snapshot.get("trace", [])), language=None)

    st.session_state.history.append(
        {"role": "assistant", "content": reply, "trace": render_trace(final_snapshot.get("trace", []))}
    )


if __name__ == "__main__":
    main()
