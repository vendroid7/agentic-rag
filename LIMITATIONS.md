# Limitations & Trade-offs

Working notes on what this system does not do and why. Kept out of the README so
it can stay blunt, and so the presentation has one place to draw from.

---

## Slide version

| Limitation | The trade |
|---|---|
| No conversational memory | Same answer the 1st time and the 100th; no follow-ups |
| Passages replaced on retry, not accumulated | Simple state; a chunk found early can be lost later |
| Hard retry budget | Guaranteed termination and cost ceiling; hard questions get cut off |
| One clarification per question | Cannot interrogate the user; a vague reply is searched anyway |
| Narrative text only | Clean chunking; weak on figures in financial tables |
| 5 tickers × 2 years | Fast reproducible ingest; most of EDGAR is out of corpus |
| Planning and clarifying measured, grounding not | 20/20 on routing; answer quality still checked by hand |

---

## No conversational memory

State lives for exactly one question and the clarification that resolves it. Each
new question starts a fresh thread, so nothing carries over.

**A question gets the same answer whether it is the first of the session or the
hundredth.** Nothing an earlier turn said can leak into a later one, drift across a
long session is impossible, and any answer can be reproduced by asking that one
question on its own.

The cost is that follow-ups do not work. *"What about Microsoft?"* is planned
against those four words alone, so the agent keeps the company and loses the topic.

Adding history is a small change — a plain `history` field on the state, filled by
the UI and read by the planner, roughly fifteen lines. It was left out because it
would let an earlier turn silently resolve an ambiguity the clarify gate should
have stopped on. Ask about Apple FY2024, then ask *"what were the main revenue
drivers?"*, and the planner would answer from history instead of stopping to ask.
The statelessness is what makes the gate's behaviour predictable, and the gate is
the thing worth demonstrating.

## Retrieved passages are replaced on each retry, not accumulated

`contexts` is a plain state field with no reducer, so a `refine` overwrites the
previous passages rather than adding to them. The loop does widen the net each
attempt (`k` scales with the retry count), but a chunk found on attempt 1 can still
be lost if the reworded query no longer surfaces it.

Accumulating instead would let a partial hit early compose with a partial hit later.
That is the single change most likely to improve answer quality.

## The retry budget can cut off a genuinely hard question

After `MAX_RETRIES` the agent must answer with whatever it holds. This is what
guarantees termination and a fixed cost ceiling per query, and it is enforced in the
router rather than the prompt so the model cannot talk its way past it. The cost is
that a question needing a fourth angle gets answered from the third.

## One clarification per question

`MAX_CLARIFICATIONS = 1` bounds the `plan → clarify → plan` cycle. If the user's
reply is still ambiguous, the agent searches anyway rather than asking again. This
stops the agent interrogating the user, at the cost of proceeding on a guess when
one round was not enough.

## Narrative text only

Chunks are prose from the target Items, so questions turning on figures in financial
tables are answered from the surrounding commentary rather than the tables
themselves. Numeric comparisons across years are the weakest case.

## Small corpus

5 tickers × 2 fiscal years, so most of EDGAR is out of scope. This is why the "no
match" branch of the clarify gate matters as much as the retrieval path — refusing
cleanly is a common correct outcome here, not an edge case.

## Grounding is not measured

`eval/planner_eval.py` covers the half of the system that decides things. Twenty
questions, each labelled with the tickers and years the planner should extract and
whether the clarify gate should fire. Because the gate is a deterministic catalog
lookup, one run measures planner accuracy and "does it ask at the right moment"
together — no labelled chunks, no LLM judge, seconds to run.

Current result: **20/20**, with both failure directions separated, because they cost
different things — asking when the question was already clear is an annoyance, while
searching an under-specified question yields a confident wrong answer.

    asked when the question was already clear: 0/10
    searched when it should have asked:        0/10

The harness was mutation-checked: feeding it deliberately wrong expectations
(inverted gate, wrong ticker, wrong sub-question count) makes it fail on each, so
the score reflects behaviour rather than a vacuous assertion.

What is still unmeasured is everything after retrieval. There is no labelled set of
questions to expected chunks, so recall@k is unknown, and no check that every claim
in an answer traces to a cited passage. Both need ground truth built by reading the
filings, which is the expensive half. Answer quality was checked by hand against the
sample prompts.
