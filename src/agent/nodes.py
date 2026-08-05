"""The five agent nodes: plan, clarify, retrieve, decide, answer.

Each node takes the AgentState and returns a dict of only the fields it changed.
LangGraph merges those updates into the state automatically.
"""

import textwrap

from agent.state import AgentState, Plan, SubQuestion, Decision
from agent.llm import LLMClient
from config import config
from retrieval.database import Database
from retrieval.hybrid import HybridRetriever


def rewrite_sub_questions(
    old: list[SubQuestion], texts: list[str], drop_filters: bool
) -> list[SubQuestion]:
    """
    Applies the model's rewritten query text to the existing retrieval targets.

    One sub-question is kept per original target, so a comparison across two
    companies cannot collapse into a single search.

    Args:
        old (list[SubQuestion]): The sub-questions from the current plan.
        texts (list[str]): Replacement query text, positionally matched to `old`.
            Positions the model left out keep their original text.
        drop_filters (bool): Whether to clear the company and fiscal year
            constraints, which is what widens an over-narrow search.

    Returns:
        list[SubQuestion]: The rewritten sub-questions to retry with.
    """
    rewritten = []
    for i, sq in enumerate(old):
        text = texts[i] if i < len(texts) else sq.text
        if drop_filters:
            rewritten.append(SubQuestion(text=text))
        else:
            rewritten.append(
                SubQuestion(text=text, company=sq.company, fiscal_year=sq.fiscal_year)
            )
    return rewritten


def make_nodes(database: Database, retriever: HybridRetriever, llm: LLMClient):
    """
    Initializes and constructs the fundamental LangGraph processing nodes.

    Args:
        database (Database): The SQL metadata catalog for the vector store.
        retriever (HybridRetriever): The Dense/Sparse hybrid retrieval engine.
        llm (LLMClient): The interface to the underlying Large Language Model.

    Returns:
        tuple: A 5-tuple containing the constructed node callables:
               (plan_node, clarify_node, retrieve_node, decide_node, answer_node)
    """

    catalog = database.get_all()
    corpus_list = "\n".join(
        f"- {t.company} ({t.ticker}) FY{t.fiscal_year}" for t in catalog
    )

    ticker_years: dict[str, list[int]] = {}
    for t in catalog:
        ticker_years.setdefault(t.ticker, []).append(t.fiscal_year)
    years_list = "\n".join(
        f"- {ticker}: available years {sorted(years)}" for ticker, years in ticker_years.items()
    )

    plan_instructions = textwrap.dedent(f"""
        You plan how to answer questions about SEC 10-K filings.

        The corpus contains exactly these filings:
        {corpus_list}

        IMPORTANT — always use the TICKER as the company value. Map common names like this:
        - Apple, APPL, apple inc → AAPL
        - Tesla, tsla, tesla inc → TSLA
        - Amazon, AMZN, amazon.com → AMZN
        - Microsoft, MSFT, msft → MSFT
        - Nvidia, NVDA, nvidia corp → NVDA
        NOTE: Users often misspell tickers or use informal names. Always normalize to the ticker.

        Available fiscal years per company:
        {years_list}

        Split the question into sub-questions (max {config.MAX_SUBQUESTIONS}). For each, set:
        - text: a standalone search query
        - company: the TICKER from the list above (e.g. "AAPL" not "Apple"). Extract this if ANY part of the input (including clarifications) mentions a company.
        - fiscal_year: the year as an integer (e.g. 2024). Extract this if ANY part of the input (including clarifications) mentions a year.

        If the input completely lacks a company or year, set those to null.
    """).strip()

    def plan_node(state: AgentState) -> dict:
        """
        Decomposes the primary query into structured execution steps.

        Args:
            state (AgentState): The current execution state encapsulating the user query.

        Returns:
            dict: An updated state dictionary containing the generated 'plan' 
                  and the concatenated 'trace' for observability.
        """
        try:
            plan = llm.structured(plan_instructions, state.user_query, Plan)
        except ValueError:
            plan = Plan(sub_questions=[SubQuestion(text=state.user_query)])

        trace = [f"Plan: {len(plan.sub_questions)} sub-question(s)"]
        for i, sq in enumerate(plan.sub_questions, 1):
            trace.append(f'  {i}. "{sq.text}" -> {sq.company or "any"} / {sq.fiscal_year or "any year"}')

        return {"plan": plan, "trace": trace}

    def clarify_node(state: AgentState) -> dict:
        """
        Validates planned sub-queries against the active database index.

        Identifies ambiguous requests or out-of-domain queries by comparing 
        the extracted entities against the SQL metadata catalog.

        Args:
            state (AgentState): The execution state containing the active plan.

        Returns:
            dict: A state update potentially containing a 'clarification_message' 
                  if human-in-the-loop validation is required, alongside the 'trace'.
        """
        for sq in state.plan.sub_questions:
            matches = database.resolve(company=sq.company, fiscal_year=sq.fiscal_year)

            if len(matches) == 0:
                options = ", ".join(
                    f"{t.company} ({t.ticker}) FY{t.fiscal_year}" for t in catalog
                )
                return {
                    "clarification_message": (
                        f"I don't have filings matching that. "
                        f"I have: {options}. Which would you like?"
                    ),
                    "trace": [f"Clarify: No match — asking user"],
                }

            if len(matches) > 1 and (sq.company is None or sq.fiscal_year is None):
                options = ", ".join(
                    f"{t.company} ({t.ticker}) FY{t.fiscal_year}" for t in matches
                )
                return {
                    "clarification_message": (
                        f'Which filing did you mean for "{sq.text}"? '
                        f"Options: {options}"
                    ),
                    "trace": [f"Clarify: {len(matches)} matches — asking user"],
                }

        return {"trace": ["Clarify: All targets resolved — searching"]}

    def retrieve_node(state: AgentState) -> dict:
        """
        Executes hybrid vector/sparse retrieval over the decomposed queries.

        Args:
            state (AgentState): The active state containing validated sub-queries.

        Returns:
            dict: The state update appending retrieved 'contexts' and 'trace' logs.
        """
        all_chunks = []
        trace = []
        
        # Dynamically expand the search net on retries
        current_k = config.FINAL_K * (state.retry_count + 1)
        
        for sq in state.plan.sub_questions:
            chunks = retriever.search(
                query=sq.text, company=sq.company, fiscal_year=sq.fiscal_year, k=current_k
            )
            all_chunks.extend(chunks)
            trace.append(f'Retrieved {len(chunks)} chunks for "{sq.text}" (Attempt {state.retry_count + 1})')

        return {"contexts": all_chunks, "trace": trace}

    decide_instructions = textwrap.dedent("""
        You are deciding what the agent should do next, given the passages retrieved so far.

        Choose exactly one action:
        - "answer": the passages contain the raw facts needed. If the user asked to
          compare or synthesize across companies, the facts for ALL of them being
          present is enough — do not expect a pre-written comparison in the passages.
        - "refine": the right filing was searched, but the wording missed the content.
          Return revised_queries using the language a 10-K actually uses
          (e.g. "supplier concentration" rather than "vendor risk").
        - "broaden": the search was restricted to the wrong company or year and so
          returned little of use. Return revised_queries; the filters are dropped
          for you.
        - "ask_user": the corpus cannot answer this without more input from the person.
          Return question_for_user.

        revised_queries must contain one query per sub-question, in the same order
        as the sub-questions you were given.

        Prefer "answer" when the facts are present. Only choose "ask_user" when no
        amount of further searching would help. Explain your choice in one sentence
        in `reason`.
    """).strip()

    def decide_node(state: AgentState) -> dict:
        """
        Chooses the agent's next action after inspecting the retrieved passages.

        This is the dynamic step in the loop: the model picks the action, while
        the retry budget is enforced here in code so it cannot loop indefinitely.

        Args:
            state (AgentState): The state containing the retrieved contexts.

        Returns:
            dict: An update setting 'next_action' for the router, plus the rewritten
                  'plan' and incremented 'retry_count' when a retry was chosen, or
                  'clarification_message' when the agent chose to ask the user.
        """
        if state.retry_count >= config.MAX_RETRIES:
            return {
                "next_action": "answer",
                "trace": ["Decide: retry budget spent — answering with what we have"],
            }

        # Nothing came back at all, so there is nothing to grade. The filters are
        # the only thing that could have excluded everything, so drop them.
        if not state.contexts:
            widened = rewrite_sub_questions(state.plan.sub_questions, [], drop_filters=True)
            return {
                "next_action": "retrieve",
                "plan": Plan(sub_questions=widened),
                "retry_count": state.retry_count + 1,
                "trace": ["Decide: broaden — filters matched nothing, dropping them"],
            }

        passages = "\n\n".join(
            f"[chunk {c.chunk_id}, {c.company} FY{c.fiscal_year}, {c.item_section}]\n{c.text[:500]}"
            for c in state.contexts
        )
        searched = "\n".join(
            f"{i}. {sq.text} ({sq.company or 'any'} / {sq.fiscal_year or 'any year'})"
            for i, sq in enumerate(state.plan.sub_questions, 1)
        )
        question = (
            f"Question: {state.user_query}\n\n"
            f"Sub-questions searched:\n{searched}\n\nPassages:\n{passages}"
        )

        try:
            decision = llm.structured(decide_instructions, question, Decision)
        except ValueError:
            return {"next_action": "answer", "trace": ["Decide: could not grade, answering"]}

        if decision.action == "answer":
            return {"next_action": "answer", "trace": [f"Decide: answer — {decision.reason}"]}

        if decision.action == "ask_user":
            return {
                "next_action": "ask_user",
                "clarification_message": decision.question_for_user,
                "trace": [f"Decide: ask_user — {decision.reason}"],
            }

        new_plan = Plan(
            sub_questions=rewrite_sub_questions(
                state.plan.sub_questions,
                decision.revised_queries,
                drop_filters=decision.action == "broaden",
            )
        )
        trace = [f"Decide: {decision.action} — {decision.reason}"]
        trace += [
            f'  retry: "{sq.text}" -> {sq.company or "any"} / {sq.fiscal_year or "any year"}'
            for sq in new_plan.sub_questions
        ]
        return {
            "next_action": "retrieve",
            "plan": new_plan,
            "retry_count": state.retry_count + 1,
            "trace": trace,
        }

    answer_instructions = textwrap.dedent("""
        Answer the question using only the passages provided.
        - Use only the passages. Never add facts from your own knowledge.
        - Cite every claim with its label, e.g. [chunk 42, NVIDIA CORP FY2025, Item 1A].
        - If you cannot find the answer in the passages, say so honestly.
        - Write directly to the person who asked.
    """).strip()

    def answer_node(state: AgentState) -> dict:
        """
        Synthesizes the final output utilizing the validated context chunks.

        Args:
            state (AgentState): The terminal state with verified contexts.

        Returns:
            dict: The final state update containing the synthesized 'final_answer'.
        """
        if not state.contexts:
            return {
                "final_answer": "I searched the SEC filings but could not find relevant passages.",
                "trace": ["Answer: No passages to work with"],
            }

        passages = "\n\n".join(
            f"[chunk {c.chunk_id}, {c.company} FY{c.fiscal_year}, {c.item_section}]\n{c.text}"
            for c in state.contexts
        )
        question = f"Question: {state.user_query}\n\nPassages:\n{passages}"

        answer = llm.text(answer_instructions, question, model="large")
        return {"final_answer": answer, "trace": ["Answer: Synthesized response"]}

    return plan_node, clarify_node, retrieve_node, decide_node, answer_node