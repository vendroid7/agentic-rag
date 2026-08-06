"""The five agent nodes: plan, clarify, retrieve, decide, answer.

Each node takes the AgentState and returns a dict of only the fields it changed.
LangGraph merges those updates into the state automatically.
"""

import textwrap

from langgraph.types import interrupt

from agent.state import AgentState, Plan, SubQuestion, Decision
from agent.llm import LLMClient
from config import config
from retrieval.database import Database
from retrieval.hybrid import HybridRetriever


def rewrite_sub_questions(
    old: list[SubQuestion], revised: list[SubQuestion], drop_filters: bool
) -> list[SubQuestion]:
    """
    Applies the model's rewritten sub-questions to the existing retrieval targets.

    One sub-question is kept per original target, so a comparison across two
    companies cannot collapse into a single search.

    Args:
        old (list[SubQuestion]): The sub-questions from the current plan.
        revised (list[SubQuestion]): Replacements, positionally matched to `old`.
            Positions the model left out keep their original sub-question, and a
            filter the model left empty falls back to the one already in place.
        drop_filters (bool): Whether to clear the company and fiscal year
            constraints, which is what widens an over-narrow search.

    Returns:
        list[SubQuestion]: The rewritten sub-questions to retry with.
    """
    rewritten = []
    for i, sq in enumerate(old):
        new = revised[i] if i < len(revised) else sq
        if drop_filters:
            rewritten.append(SubQuestion(text=new.text))
        else:
            rewritten.append(
                SubQuestion(
                    text=new.text,
                    company=new.company or sq.company,
                    fiscal_year=new.fiscal_year or sq.fiscal_year,
                )
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

        Earlier turns may be listed above the current question. Use them only to
        resolve what the current question leaves implicit — "it", "that", "what
        about X", "and 2025". When the current question names its own company and
        year, ignore the earlier turns entirely.
    """).strip()

    def plan_node(state: AgentState) -> dict:
        """
        Decomposes the primary query into structured execution steps.

        Runs again after every clarification, so anything the user has since told
        us is planned against together with their original question.

        Args:
            state (AgentState): The current execution state encapsulating the user query.

        Returns:
            dict: An updated state dictionary containing the generated 'plan'
                  and the concatenated 'trace' for observability.
        """
        question = state.user_query
        for reply in state.clarifications:
            question += f"\nThe user clarified: {reply}"

        if state.history:
            earlier = "\n".join(f"- {turn}" for turn in state.history)
            question = f"Earlier turns:\n{earlier}\n\nCurrent question: {question}"

        try:
            plan = llm.structured(plan_instructions, question, Plan)
        except ValueError:
            plan = Plan(sub_questions=[SubQuestion(text=state.user_query)])

        # The limit in the prompt is advice the model can ignore; this is what makes
        # it hold, and each extra sub-question is another full retrieval and rerank.
        dropped = len(plan.sub_questions) - config.MAX_SUBQUESTIONS
        if dropped > 0:
            plan = Plan(sub_questions=plan.sub_questions[: config.MAX_SUBQUESTIONS])

        trace = [f"Plan: {len(plan.sub_questions)} sub-question(s)"]
        for i, sq in enumerate(plan.sub_questions, 1):
            trace.append(f'  {i}. "{sq.text}" -> {sq.company or "any"} / {sq.fiscal_year or "any year"}')
        if dropped > 0:
            trace.append(f"  ({dropped} more dropped — the plan is capped at {config.MAX_SUBQUESTIONS})")

        return {"plan": plan, "trace": trace}

    def clarify_node(state: AgentState) -> dict:
        """
        Validates planned sub-queries against the active database index.

        Identifies ambiguous requests or out-of-domain queries by comparing
        the extracted entities against the SQL metadata catalog. When one is
        found the graph is suspended until the user answers, and the reply sends
        the agent back through planning with the target it was missing.

        Args:
            state (AgentState): The execution state containing the active plan.

        Returns:
            dict: A state update routing to 'plan' with the user's reply recorded
                  in 'clarifications', or to 'retrieve' when every target resolves.
        """
        for sq in state.plan.sub_questions:
            matches = database.resolve(company=sq.company, fiscal_year=sq.fiscal_year)

            if len(matches) == 0:
                options = ", ".join(
                    f"{t.company} ({t.ticker}) FY{t.fiscal_year}" for t in catalog
                )
                question = (
                    f"I don't have filings matching that. "
                    f"I have: {options}. Which would you like?"
                )
                note = "Clarify: No match — asking user"
            elif len(matches) > 1 and (sq.company is None or sq.fiscal_year is None):
                options = ", ".join(
                    f"{t.company} ({t.ticker}) FY{t.fiscal_year}" for t in matches
                )
                question = (
                    f'Which filing did you mean for "{sq.text}"? Options: {options}'
                )
                note = f"Clarify: {len(matches)} matches — asking user"
            else:
                continue

            if state.clarify_rounds >= config.MAX_CLARIFICATIONS:
                return {
                    "next_action": "retrieve",
                    "trace": ["Clarify: Still ambiguous, but already asked — searching anyway"],
                }

            reply = interrupt(question)
            return {
                "next_action": "plan",
                "clarifications": [reply],
                "clarify_rounds": state.clarify_rounds + 1,
                "trace": [note, f"Clarify: User replied — {reply}"],
            }

        return {
            "next_action": "retrieve",
            "trace": ["Clarify: All targets resolved — searching"],
        }

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
        - "ask_user": the person can tell you something that would narrow the search --
          which company, which fiscal year, which section. Return question_for_user.
          Do not choose this when the filings simply do not discuss the subject:
          nothing the user could say would help, so "answer" and say so plainly.

        revised_queries must contain one entry per sub-question, in the same order
        as the sub-questions you were given. Each entry has the same shape as a
        sub-question: text, company (a ticker or null) and fiscal_year (an integer
        or null). Set company or fiscal_year when the search went to the wrong
        filing and you know which one it should have gone to; leave them null to
        keep the filter already in place.

        Prefer "answer" when the facts are present, and equally when the filings
        clearly do not contain them -- reporting honestly that something is not in
        the filings is a real answer, not a failure. Explain your choice in one
        sentence in `reason`.
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
                  the user's reply in 'clarifications' when the agent chose to ask.
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
            if state.clarify_rounds >= config.MAX_CLARIFICATIONS:
                return {
                    "next_action": "answer",
                    "trace": ["Decide: ask_user, but already asked once — answering instead"],
                }
            reply = interrupt(decision.question_for_user)
            return {
                "next_action": "plan",
                "clarifications": [reply],
                "clarify_rounds": state.clarify_rounds + 1,
                "trace": [
                    f"Decide: ask_user — {decision.reason}",
                    f"Decide: User replied — {reply}",
                ],
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

        answer = llm.text(answer_instructions, question)
        return {"final_answer": answer, "trace": ["Answer: Synthesized response"]}

    return plan_node, clarify_node, retrieve_node, decide_node, answer_node