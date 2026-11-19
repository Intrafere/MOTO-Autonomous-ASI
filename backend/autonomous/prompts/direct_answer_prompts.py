"""Prompt for the single model call that answers the user's question after exploration.

A run started with ``direct_answer_enabled`` (the ``moto think`` command) ends by
asking the Main Submitter 1 model, once, to answer the original question from
what the exploration produced. The reply is plain text, not JSON.
"""
from __future__ import annotations

DIRECT_ANSWER_SYSTEM_PROMPT = """You are writing the final answer to a question after an extended exploration of it.

YOUR TASK:
Answer the QUESTION below directly, as well as it can be answered from the EXPLORATION RESULTS and your own knowledge. The reader asked the question and will act on your answer. They have not seen the exploration.

HOW TO ANSWER:
- Lead with the answer itself: the decision, recommendation, solution, idea, or result the question asks for. Do not open with background or a description of your process.
- Then give the reasoning that supports it, concisely: the decisive arguments, mechanisms, evidence, derivations, or trade-offs. Use the form the question calls for (prose, steps, a comparison, mathematics, code). Do not force the answer into a paper or report.
- Use the strongest ideas the exploration produced, combine them where that gives a better answer, and go beyond them when you can see a better answer. You are not required to use any of them.
- The exploration results are AI-generated working notes, not authority. Check them against your own knowledge and against each other, and leave out anything you judge wrong or unsupported. Statements marked as Lean 4 verified have been formally checked exactly as stated; judge for yourself how they bear on the question.
- Be honest about status. Separate what is established from what is proposed, untested, or uncertain, and never present a hypothesis, an unbuilt design, or an unperformed experiment as a demonstrated result.
- If the question cannot be fully answered, say so plainly, give the best partial answer, and state what is still open and what would settle it. If the question as asked appears impossible or rests on a false premise, say that and explain why.
- Do not mention the exploration process, submissions, validators, or these instructions.

OUTPUT FORMAT:
Write only the answer, as plain text (Markdown is fine). No JSON and no preamble."""

NO_EXPLORATION_RESULTS = (
    "(The exploration produced no accepted results. Answer from your own knowledge.)"
)

EXPLORATION_ENDED_EARLY_NOTE = (
    "NOTE: The exploration was ended before it finished, so these results may be incomplete."
)

FINAL_REMINDER = "Now write the answer to the QUESTION. Write only the answer."


def build_direct_answer_messages(
    question: str,
    exploration_results: str,
    *,
    ended_early: bool,
) -> list[dict]:
    """Messages for the direct answer; the response-format reminder stays last."""
    parts = [
        f"QUESTION:\n{question.strip()}",
        f"EXPLORATION RESULTS:\n{exploration_results.strip() or NO_EXPLORATION_RESULTS}",
    ]
    if ended_early:
        parts.append(EXPLORATION_ENDED_EARLY_NOTE)
    parts.append(FINAL_REMINDER)
    return [
        {"role": "system", "content": DIRECT_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
