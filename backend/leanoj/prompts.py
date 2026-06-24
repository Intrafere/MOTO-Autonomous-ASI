"""Prompt builders for the LeanOJ proof-solver mode."""
from __future__ import annotations

import re
from typing import Any, Iterable


JSON_RULES = (
    "Respond with ONLY valid JSON. Do not use markdown fences. "
    "Escape Lean backslashes and newlines correctly for JSON strings."
)

LEANOJ_FORMALIZATION_GUARDRAILS = """LEANOJ FORMALIZATION GUARDRAILS:
- Treat the LeanOJ template as the source of truth for formal semantics. Do not silently reinterpret template operations to match informal olympiad intuition. For example, in a template over `Nat`, `a - b` is truncated natural subtraction, not signed integer subtraction.
- Before committing to a closed-form `answer`, test proposed formulas and constructions against the exact Lean predicate on small cases when feasible. Counterexamples to the exact template override informal expectations.
- Lean acceptance is necessary but not sufficient for final success. A Lean-verified file proves the formal statement it encodes; it does not automatically prove the user's informal problem statement if the template or chosen definitions exploit or mismatch the natural-language task.
- If the template semantics and informal statement appear to conflict, make the mismatch explicit in reasoning and do not claim that a Lean-verified template proof settles the informal statement unless that correspondence has also been justified."""

CREATIVITY_EMPHASIS_BOOST_PROMPT = """CREATIVITY EMPHASIS BOOST:
This is the special creativity-emphasized submitter turn. Follow the same JSON schema and proof rigor requirements as normal.

Only where it is apparent, appearing true, and potentially very helpful, you may use extreme creativity to propose a near-solution or adjacent solution that solves toward the user's prompt and could advance this brainstorm further in future submissions.

Do not force creativity. If the creative route is not apparent or would weaken Lean-template rigor, submit the strongest normal direct-progress contribution instead."""


def _format_items(items: Iterable[Any], *, empty: str = "[none]") -> str:
    values = [str(item).strip() for item in (items or []) if str(item).strip()]
    if not values:
        return empty
    return "\n".join(f"{index}. {value}" for index, value in enumerate(values, start=1))


def _format_brainstorm(ideas: list[str], limit: int = 80) -> str:
    if not ideas:
        return "[No accepted brainstorm ideas yet.]"
    visible = ideas[-limit:]
    prefix = "" if len(visible) == len(ideas) else f"[Showing most recent {len(visible)} of {len(ideas)} accepted ideas.]\n"
    return prefix + "\n".join(f"{index}. {idea}" for index, idea in enumerate(visible, start=1))


def _final_mode_text(value: Any) -> str:
    text = str(value or "")
    cleaned = (
        text.replace("need_more_brainstorming", "additional proof context")
        .replace("Brainstorm", "Proof memory")
        .replace("brainstorm", "proof memory")
        .replace("BRAINSTORM", "PROOF MEMORY")
    )
    return _remove_attempt_count_language(cleaned)


def _remove_attempt_count_language(value: Any) -> str:
    text = str(value or "")
    replacements = (
        (
            r"\bfailed\s+\d+\s+consecutive\s+verification/edit\s+attempts?\b",
            "encountered repeated verification/edit failures",
        ),
        (r"\bfailed\s+\d+\s+consecutive\s+attempts?\b", "encountered repeated failures"),
        (r"\bfailed\s+\d+\s+attempts?\b", "encountered repeated failures"),
        (r"\bfailed\s+\d+\s+times\b", "encountered repeated failures"),
        (r"\bafter\s+failed\s+attempts\b", "after recent proof-check failures"),
        (r"\bfailed\s+attempts\b", "proof-check failures"),
        (r"\battempts\s+\d+\s*-\s*\d+\b", "recent final-loop feedback"),
        (r"\bwith\s+exactly\s+\d+\s+failed\s+attempts?\b", "with recent proof-check failures"),
        (r"\bUse this exact failed-attempt count[^.]*\.", ""),
        (r"\bfailed-attempt count\b", "failure context"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return re.sub(r" {2,}", " ", text).strip()


def _format_proof_memory_notes(ideas: list[str], limit: int = 80) -> str:
    if not ideas:
        return "[No accepted proof memory notes yet.]"
    visible = ideas[-limit:]
    prefix = "" if len(visible) == len(ideas) else f"[Showing most recent {len(visible)} accepted proof memory notes.]\n"
    return prefix + "\n".join(f"{index}. {_final_mode_text(idea)}" for index, idea in enumerate(visible, start=1))


def _format_verified_subproofs(subproofs: list[dict[str, Any]]) -> str:
    if not subproofs:
        return "[No verified subproofs yet.]"
    blocks = []
    for index, subproof in enumerate(subproofs, start=1):
        lean_feedback = str(subproof.get("lean_feedback") or "").strip()
        feedback_lines = ["Lean verifier feedback:", lean_feedback] if lean_feedback else []
        blocks.append(
            "\n".join(
                [
                    f"SUBPROOF {index}: {subproof.get('request', '')}",
                    f"Role: {subproof.get('role', '')}",
                    f"Theorem/Lemma: {subproof.get('theorem_or_lemma', '')}",
                    *feedback_lines,
                    "Verified Lean 4 code:",
                    subproof.get("lean_code", ""),
                    "---",
                ]
            )
        )
    return "\n".join(blocks)


def _format_verified_subproofs_for_final(subproofs: list[dict[str, Any]]) -> str:
    if not subproofs:
        return "[No verified subproofs yet.]"
    blocks = []
    for index, subproof in enumerate(subproofs, start=1):
        lean_feedback = _final_mode_text(subproof.get("lean_feedback") or "").strip()
        feedback_lines = ["Lean verifier feedback:", lean_feedback] if lean_feedback else []
        blocks.append(
            "\n".join(
                [
                    f"SUBPROOF {index}: {_final_mode_text(subproof.get('request', ''))}",
                    f"Theorem/Lemma: {_final_mode_text(subproof.get('theorem_or_lemma', ''))}",
                    *feedback_lines,
                    "Verified Lean 4 code:",
                    subproof.get("lean_code", ""),
                    "---",
                ]
            )
        )
    return "\n".join(blocks)


def _format_partial_proofs(partial_proofs: list[dict[str, Any]], limit: int = 8) -> str:
    if not partial_proofs:
        return "[No accepted partial proof scaffolds yet.]"
    blocks = []
    for index, proof in enumerate(partial_proofs[-limit:], start=1):
        placeholders = ", ".join(proof.get("placeholder_tokens") or []) or "unknown"
        blocks.append(
            "\n".join(
                [
                    f"PARTIAL PROOF {index}: {proof.get('request', '')}",
                    f"Target: {proof.get('target', '')}; placeholders: {placeholders}",
                    f"Summary: {proof.get('summary', '')}",
                    "Lean-accepted incomplete scaffold:",
                    proof.get("lean_code", ""),
                    "---",
                ]
            )
        )
    return "\n".join(blocks)


def _format_partial_proofs_for_final(partial_proofs: list[dict[str, Any]], limit: int = 8) -> str:
    if not partial_proofs:
        return "[No accepted partial proof scaffolds yet.]"
    blocks = []
    for index, proof in enumerate(partial_proofs[-limit:], start=1):
        placeholders = ", ".join(proof.get("placeholder_tokens") or []) or "unknown"
        blocks.append(
            "\n".join(
                [
                    f"PARTIAL PROOF {index}: {_final_mode_text(proof.get('request', ''))}",
                    f"Placeholders: {placeholders}",
                    f"Summary: {_final_mode_text(proof.get('summary', ''))}",
                    "Lean-accepted incomplete scaffold:",
                    proof.get("lean_code", ""),
                    "---",
                ]
            )
        )
    return "\n".join(blocks)


def _format_failures(failures: list[dict[str, Any]], limit: int = 10) -> str:
    if not failures:
        return "[No useful failed proof feedback yet.]"
    visible = failures[-limit:]
    blocks = []
    for index, failure in enumerate(visible, start=1):
        block = (
            f"{index}. {_remove_attempt_count_language(failure.get('request', 'final proof'))} :: "
            f"{_remove_attempt_count_language(failure.get('error_summary', ''))}"
        )
        lean_feedback = str(failure.get("lean_feedback") or "").strip()
        if lean_feedback:
            block += f"\n   Lean feedback: {_remove_attempt_count_language(lean_feedback)}"
        blocks.append(block)
    return "\n".join(blocks)


def _format_feedback_notes(failures: list[dict[str, Any]], limit: int = 10) -> str:
    if not failures:
        return "[No recent proof feedback available.]"
    visible = failures[-limit:]
    blocks = []
    for failure in visible:
        request = str(failure.get("request") or "").strip()
        error_summary = str(failure.get("error_summary") or failure.get("error_output") or "").strip()
        lean_feedback = str(failure.get("lean_feedback") or "").strip()
        combined = "\n".join(part for part in [request, error_summary, lean_feedback] if part).lower()
        phase_noise = "need_more_brainstorming" in combined or "stuck_needs_brainstorm" in combined
        if phase_noise and not _has_concrete_execution_feedback(combined):
            continue
        pieces = [
            part
            for part in [
                _final_mode_text(error_summary),
                f"Lean feedback: {_final_mode_text(lean_feedback)}" if lean_feedback else "",
            ]
            if part
        ]
        if pieces:
            blocks.append("\n".join(pieces))
    return "\n\n---\n\n".join(blocks) if blocks else "[No recent proof feedback available.]"


def _has_concrete_execution_feedback(text: str) -> bool:
    concrete_terms = (
        "old_string",
        "unexpected token",
        "missing cases",
        "unsolved goals",
        "error:",
        "rejected",
        "invalid",
        "json",
        "max_tokens",
        "lean",
        "verification",
        "watchdog",
    )
    lowered = str(text or "").lower()
    return any(term in lowered for term in concrete_terms)


def _clip_prompt_field(value: Any, limit: int = 1200) -> str:
    text = _final_mode_text(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20].rstrip() + " ... [truncated]"


def _format_recent_final_attempts(attempts: list[dict[str, Any]], limit: int = 5) -> str:
    visible = [record for record in (attempts or [])[-limit:] if isinstance(record, dict)]
    if not visible:
        return "[No recent final feedback recorded.]"
    blocks = []
    for index, record in enumerate(visible, start=1):
        request = _clip_prompt_field(record.get("request") or "final proof feedback", limit=300)
        error_summary = _clip_prompt_field(
            record.get("error_summary") or record.get("error_output") or "",
            limit=1400,
        )
        lean_feedback = _clip_prompt_field(record.get("lean_feedback") or "", limit=1000)
        reasoning = _clip_prompt_field(record.get("reasoning") or "", limit=800)
        lines = [f"FEEDBACK ITEM {index}: {request}"]
        if error_summary:
            lines.append(f"Result/error: {error_summary}")
        if lean_feedback:
            lines.append(f"Lean feedback: {lean_feedback}")
        if reasoning:
            lines.append(f"Prior solver reasoning: {reasoning}")
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)


def _format_context_blocks(context_blocks: dict[str, str] | None, fallback: str) -> str:
    if not context_blocks:
        return fallback
    sections = []
    working_proof = (context_blocks.get("current_working_proof_attempt") or "").strip()
    current_packet = (context_blocks.get("current_final_cycle_packet") or "").strip()
    direct_context = (context_blocks.get("direct_proof_context") or "").strip()
    rag_context = (context_blocks.get("rag_evidence_context") or "").strip()
    proof_search_context = (context_blocks.get("proof_search_context") or "").strip()
    refuted_warnings = (context_blocks.get("refuted_construction_warnings") or "").strip()
    capped_feedback = (context_blocks.get("capped_rejection_feedback") or "").strip()
    if working_proof:
        sections.append(working_proof)
    if current_packet:
        sections.append(current_packet)
    if direct_context:
        sections.append(f"DIRECT PROOF CONTEXT:\n{direct_context}")
    if rag_context:
        sections.append(f"RETRIEVED LEANOJ RAG EVIDENCE:\n{rag_context}")
    if proof_search_context:
        sections.append(
            "SYNTHETIC / LOCAL VERIFIED PROOF SEARCH RESULTS:\n"
            f"{proof_search_context}"
        )
    if refuted_warnings:
        sections.append(
            "REFUTED CONSTRUCTIONS - DO NOT USE AS PROOF EVIDENCE:\n"
            f"{refuted_warnings}"
        )
    if capped_feedback:
        sections.append(f"CAPPED REJECTION FEEDBACK:\n{capped_feedback}")
    return "\n\n".join(sections) if sections else fallback


def build_topic_candidate_prompt(
    user_prompt: str,
    lean_template: str,
    prior_topics: list[str],
    creativity_emphasized: bool = False,
) -> str:
    creativity_section = f"\n{CREATIVITY_EMPHASIS_BOOST_PROMPT}\n" if creativity_emphasized else ""
    return f"""You are generating one candidate root foundation question for a LeanOJ proof-solving run.

The system must solve the user's Lean 4 template completely. Propose a broad initial foundation question that can guide the entire session before recursive brainstorms add details. This is not a local sublemma target: it should set the durable direction for finding the complete solution.

The topic must address ALL major solution obligations:
- Determine an explicit formula/value for `answer n`.
- Find or verify the extremal lower-bound construction.
- Prove the matching upper bound.
- Respect the exact LeanOJ template semantics, including Lean/Nat behavior.
- Identify a Mathlib-compatible Lean 4 formalization route for `IsGreatest (S n) (answer n)`.

Reject narrow framing in your own generation. Do not return a topic that is only about one lemma, one tactic, one bound, one construction, small-case testing alone, or repairing a current proof attempt.

USER PROBLEM:
{user_prompt}

LEANOJ TEMPLATE:
{lean_template}

{LEANOJ_FORMALIZATION_GUARDRAILS}

PRIOR VALIDATED TOPICS:
{_format_items(prior_topics)}
{creativity_section}

Return a new non-duplicative broad foundation topic. It should read like a general question that addresses the whole problem and can remain locked as the initial session foundation. If prior topics already cover the same root framing, choose a distinct foundation angle that still covers all obligations, such as exact-template semantics first, extremal-combinatorics first, or Lean-formalization architecture first.

Correct topic style:
{{"topic": "Determine a complete Lean 4 solution strategy for the exact LeanOJ template, including the explicit answer formula, extremal construction, upper-bound proof, template-semantics checks, and Mathlib formalization route.", "reasoning": "This covers every obligation needed for the final LeanOJ proof."}}

Wrong topic style:
{{"topic": "Find a useful divisibility lemma for complex numbers.", "reasoning": "This is too narrow because it targets only one possible lemma and does not address the full solution foundation."}}

{JSON_RULES}
JSON format:
{{"topic": "broad foundation topic", "reasoning": "why this topic sets the best foundation for solving the whole Lean template"}}
"""


def build_topic_validation_prompt(user_prompt: str, lean_template: str, topic: str, accepted_topics: list[str]) -> str:
    return f"""You are validating a proposed LeanOJ initial foundation topic.

Accept only if the topic is relevant to solving the user's exact Lean 4 template, non-duplicative, and broad enough to serve as the locked initial session foundation.

The topic must address ALL major solution obligations:
- An explicit formula/value for `answer n`.
- A lower-bound construction.
- A matching upper-bound proof.
- Exact LeanOJ template semantics, including Lean/Nat behavior.
- A Lean 4 / Mathlib formalization route for `IsGreatest (S n) (answer n)`.

Reject topics that are narrow, partial, or local: one sublemma, one tactic, one bound, one construction, small-case testing alone, or current-proof repair. Those belong in recursive brainstorms after the foundation exists, not in initial topic selection.

USER PROBLEM:
{user_prompt}

LEANOJ TEMPLATE:
{lean_template}

{LEANOJ_FORMALIZATION_GUARDRAILS}

ACCEPTED TOPICS:
{_format_items(accepted_topics)}

PROPOSED TOPIC:
{topic}

Correct acceptance target:
{{"decision": "accept", "reasoning": "The topic covers answer formula, construction, upper bound, template semantics, and Lean formalization.", "summary": "Broad foundation topic."}}

Required rejection target for narrow topics:
{{"decision": "reject", "reasoning": "The topic asks for one divisibility lemma.", "summary": "Invalid because this is a narrow sublemma topic, not a whole-problem foundation."}}

{JSON_RULES}
JSON format:
{{"decision": "accept or reject", "reasoning": "brief validation reasoning", "summary": "short feedback if rejected"}}
"""


def build_topic_batch_validation_prompt(
    user_prompt: str,
    lean_template: str,
    topics: list[str],
    accepted_topics: list[str],
) -> str:
    formatted_topics = "\n\n---\n\n".join(
        f"TOPIC {index}:\n{topic}"
        for index, topic in enumerate(topics, start=1)
    )
    return f"""You are the single validator for cumulative LeanOJ initial foundation topics.

Evaluate EACH proposed topic independently against the current accepted topic context, then check accepted topics for intra-batch redundancy. Accept only topics that are relevant to solving the user's exact Lean 4 template, non-duplicative, and broad enough to serve as the locked initial session foundation.

Each accepted topic must address ALL major solution obligations:
- An explicit formula/value for `answer n`.
- A lower-bound construction.
- A matching upper-bound proof.
- Exact LeanOJ template semantics, including Lean/Nat behavior.
- A Lean 4 / Mathlib formalization route for `IsGreatest (S n) (answer n)`.

CRITICAL:
- Judge each topic against CURRENT ACCEPTED TOPICS first, not against the other topics in this batch.
- Only after independent decisions, compare independently accepted topics against each other.
- If two accepted topics are redundant with each other, keep the stronger/more concrete one and reject the weaker one with an intra-batch redundancy summary.
- Reject narrow or partial initial topics even if they would be useful later: one sublemma, one tactic, one bound, one construction, small-case testing alone, or current-proof repair.
- Return exactly one decision object per topic, in the same order.

USER PROBLEM:
{user_prompt}

LEANOJ TEMPLATE:
{lean_template}

{LEANOJ_FORMALIZATION_GUARDRAILS}

CURRENT ACCEPTED TOPICS:
{_format_items(accepted_topics)}

TOPICS TO VALIDATE:
{formatted_topics}

Correct acceptance target:
{{"topic_number": 1, "decision": "accept", "reasoning": "The topic covers the full answer formula, construction, upper bound, exact template semantics, and Lean formalization route.", "summary": "Broad foundation topic."}}

Required rejection target for narrow topics:
{{"topic_number": 1, "decision": "reject", "reasoning": "The topic asks for one useful tactic or helper lemma.", "summary": "Invalid because this is too narrow for initial topic selection."}}

{JSON_RULES}
JSON format:
{{"decisions": [{{"topic_number": 1, "decision": "accept or reject", "reasoning": "validation reasoning", "summary": "short rejection or acceptance summary"}}]}}
"""


def build_topic_selection_prompt(user_prompt: str, lean_template: str, topics: list[str]) -> str:
    return f"""You are selecting the locked initial foundation topic for a LeanOJ proof-solving run.

Choose exactly one of the validated topics below, or propose a clearly better replacement topic. The chosen topic must maximize the chance of solving the Lean 4 template by setting a broad root direction for the whole session.

The selected topic will be treated as the initial frozen foundation that recursive brainstorms build on. It must not be a narrow sublemma, tactic-only investigation, one-bound-only question, one-construction-only question, small-case-only check, or current-proof repair target.

The selected topic must address ALL major solution obligations:
- Determine an explicit formula/value for `answer n`.
- Establish the extremal lower-bound construction.
- Prove the matching upper bound.
- Respect exact LeanOJ template semantics, including Lean/Nat behavior.
- Set a Mathlib-compatible Lean 4 formalization route for `IsGreatest (S n) (answer n)`.

USER PROBLEM:
{user_prompt}

LEANOJ TEMPLATE:
{lean_template}

{LEANOJ_FORMALIZATION_GUARDRAILS}

VALIDATED TOPICS:
{_format_items(topics)}

Correct selected-topic style:
{{"topic": "Determine a complete Lean 4 solution strategy for the exact LeanOJ template, including the explicit answer formula, extremal construction, upper-bound proof, template-semantics checks, and Mathlib formalization route.", "reasoning": "This is broad enough to anchor the session and leaves recursive brainstorms to fill in details."}}

Wrong selected-topic style:
{{"topic": "Prove one helper divisibility lemma.", "reasoning": "This is too narrow because it cannot serve as the locked foundation for the whole problem."}}

{JSON_RULES}
JSON format:
{{"topic": "selected or improved broad foundation topic", "reasoning": "why this is the best locked initial foundation for solving the whole Lean template"}}
"""


def build_brainstorm_prompt(
    user_prompt: str,
    lean_template: str,
    active_topic: str,
    accepted_ideas: list[str],
    verified_subproofs: list[dict[str, Any]],
    failed_feedback: list[dict[str, Any]],
    context_blocks: dict[str, str] | None = None,
    creativity_emphasized: bool = False,
) -> str:
    fallback_context = f"""ACCEPTED BRAINSTORM CONTEXT:
{_format_brainstorm(accepted_ideas)}

VERIFIED SUBPROOFS:
{_format_verified_subproofs(verified_subproofs)}

USEFUL FAILED PROOF FEEDBACK:
{_format_failures(failed_feedback)}"""
    creativity_section = f"\n{CREATIVITY_EMPHASIS_BOOST_PROMPT}\n" if creativity_emphasized else ""
    return f"""You are a LeanOJ proof brainstorm submitter.

YOUR TASK:
Generate a novel mathematical insight that advances the user's goal.
Generate one concrete idea that helps solve the user's Lean 4 template.

Focus on exact Lean tactics, Mathlib lemmas, theorem-shaping, induction/cases structure, or mathematical transformations. If a current working proof attempt is provided, treat ACTIVE TOPIC as that exact proof-repair target. Brainstorm only information that directly helps complete or repair it; if a direct solution is unavailable, give the nearest concrete step that works toward solving that exact proof.

If you can produce a complete Lean 4 proof for a useful sublemma or proof fragment, you may choose `submission_type: "lean_proof"`. Use that route only when the proved statement directly discharges, splits, or repairs a current obligation in the LeanOJ template and is either public/citable novelty absent from standard references or Mathlib, or a template-specific proof artifact whose novelty rationale clearly explains why it is not merely a standard known fact. Do not use it to build a generic known-knowledge base of routine Mathlib facts, standard textbook lemmas, proof-engineering glue, or program-local firsts. The system will require novelty/prompt-rationale fields, run Lean 4 first, give you up to 5 repair attempts with Lean feedback, and only then send the Lean-verified proof to the normal brainstorm validator. Do not use `sorry`, `admit`, or fake `axiom`/`constant`/`opaque` devices.

Do not write a whole final proof unless the idea is directly useful as context. Final template solving still happens in the final loop.

USER PROBLEM:
{user_prompt}

LEANOJ TEMPLATE:
{lean_template}

{LEANOJ_FORMALIZATION_GUARDRAILS}

ACTIVE TOPIC:
{active_topic}

ALLOCATED LEANOJ PROOF MEMORY:
{_format_context_blocks(context_blocks, fallback_context)}
{creativity_section}

{JSON_RULES}
JSON format for a normal idea:
{{"submission_type": "idea", "submission": "one concrete proof-solving idea", "reasoning": "why it advances the LeanOJ solution"}}

JSON format for a Lean proof candidate:
{{"submission_type": "lean_proof", "theorem_statement": "natural-language statement proved", "formal_sketch": "why this proof fragment helps the LeanOJ template", "expected_novelty_tier": "major_mathematical_discovery | mathematical_discovery | novel_variant | novel_formulation", "prompt_relevance_rationale": "which exact LeanOJ template obligation this proof discharges, splits, or repairs", "novelty_rationale": "why this proof fragment is absent from standard references or Mathlib / citable for this template route rather than a generic known fact or program-local first", "why_not_standard_known_result": "why this is not merely a standard Mathlib/textbook/routine helper lemma", "theorem_name": "optional Lean declaration name", "lean_code": "complete Lean 4 code", "reasoning": "why this verified proof would help"}}
"""


def build_brainstorm_validation_prompt(
    user_prompt: str,
    lean_template: str,
    submission: str,
    accepted_ideas: list[str],
    context_blocks: dict[str, str] | None = None,
) -> str:
    fallback_context = f"CURRENT ACCEPTED IDEAS:\n{_format_brainstorm(accepted_ideas)}"
    return f"""You are the single validator for a cumulative LeanOJ proof-solving brainstorm.

Accept the submission only if it adds useful, non-redundant information for solving the exact Lean template. Reject vague encouragement, duplicate ideas, or claims unrelated to Lean verification.

If the submission contains [LEAN 4 VERIFIED BRAINSTORM PROOF], Lean 4 and MOTO integrity checks already accepted the code. Your job is still to decide whether the verified proof is useful, relevant, and non-redundant for this LeanOJ brainstorm. Do not re-prove Lean correctness, and do not accept irrelevant, trivial, routine, or generic known-knowledge proofs merely because Lean verified them. Accept such a proof only when it directly discharges, splits, or repairs an exact LeanOJ template obligation.

Classify accepted submissions for later final-proof context:
- active_plan: a concrete current proof route, decomposition plan, or next obligation that should guide `master_proof.lean`.
- verified_hint: a reusable verified lemma or exact Lean tactic fact.
- refuted_construction: a failed construction/counterexample/route warning. This is useful only as "do not use" feedback and must not be treated as proof evidence.
- scratch: useful exploratory context that should not be direct final-proof context.

Use `scratch` unless the submission clearly fits one of the narrower roles. Do not default to `active_plan`.

USER PROBLEM:
{user_prompt}

LEANOJ TEMPLATE:
{lean_template}

{LEANOJ_FORMALIZATION_GUARDRAILS}

ALLOCATED LEANOJ PROOF MEMORY:
{_format_context_blocks(context_blocks, fallback_context)}

SUBMISSION:
{submission}

{JSON_RULES}
JSON format:
{{"decision": "accept", "context_role": "scratch", "reasoning": "validation reasoning", "summary": "short rejection or acceptance summary"}}
"""


def build_brainstorm_batch_validation_prompt(
    user_prompt: str,
    lean_template: str,
    submissions: list[str],
    accepted_ideas: list[str],
    context_blocks: dict[str, str] | None = None,
) -> str:
    formatted_submissions = "\n\n---\n\n".join(
        f"SUBMISSION {index}:\n{submission}"
        for index, submission in enumerate(submissions, start=1)
    )
    fallback_context = f"CURRENT ACCEPTED IDEAS:\n{_format_brainstorm(accepted_ideas)}"
    return f"""You are the single validator for a cumulative LeanOJ proof-solving brainstorm.

Evaluate EACH submission independently against the current accepted brainstorm context, then check accepted submissions for intra-batch redundancy. Accept only submissions that add useful, non-redundant information for solving the exact Lean template. Reject vague encouragement, duplicate ideas, or claims unrelated to Lean verification.

If a submission contains [LEAN 4 VERIFIED BRAINSTORM PROOF], Lean 4 and MOTO integrity checks already accepted the code. Still decide whether that verified proof is useful, relevant, and non-redundant for this LeanOJ brainstorm. Reject generic known-knowledge proofs, routine helpers, or standard Mathlib/textbook facts unless the submission explains how the verified statement directly discharges, splits, or repairs an exact LeanOJ template obligation.

For each accepted submission, classify how it may be used later:
- active_plan: a concrete current proof route, decomposition plan, or next obligation that should guide `master_proof.lean`.
- verified_hint: a reusable verified lemma or exact Lean tactic fact.
- refuted_construction: a failed construction/counterexample/route warning. This is useful only as "do not use" feedback and must not be treated as proof evidence.
- scratch: useful exploratory context that should not be direct final-proof context.

Use `scratch` unless the submission clearly fits one of the narrower roles. Do not default to `active_plan`.

CRITICAL:
- Judge each submission against CURRENT ACCEPTED IDEAS first, not against the other submissions in this batch.
- Only after independent decisions, compare independently accepted submissions against each other.
- If two accepted submissions are redundant with each other, keep the stronger/more concrete one and reject the weaker one with an intra-batch redundancy summary.
- Return exactly one decision object per submission, in the same order.

USER PROBLEM:
{user_prompt}

LEANOJ TEMPLATE:
{lean_template}

{LEANOJ_FORMALIZATION_GUARDRAILS}

ALLOCATED LEANOJ PROOF MEMORY:
{_format_context_blocks(context_blocks, fallback_context)}

SUBMISSIONS TO VALIDATE:
{formatted_submissions}

{JSON_RULES}
JSON format:
{{"decisions": [{{"submission_number": 1, "decision": "accept", "context_role": "scratch", "reasoning": "validation reasoning", "summary": "short rejection or acceptance summary"}}]}}
"""


def build_brainstorm_prune_review_prompt(
    user_prompt: str,
    lean_template: str,
    active_topic: str,
    accepted_ideas: list[str],
    context_blocks: dict[str, str] | None = None,
) -> str:
    fallback_context = f"CURRENT ACCEPTED IDEAS:\n{_format_brainstorm(accepted_ideas)}"
    return f"""You are checking whether any LeanOJ brainstorm memory should be removed or updated because it is outdated, redundant, wrong, harmful, superseded, or missing proof-solving information.

You may propose AT MOST ONE operation. Do not force a removal: choose "none" unless one operation clearly improves the proof-solving database.

Allowed actions:
- "none": no change is needed.
- "delete": remove one accepted idea that is outdated, wrong, harmful, redundant with stronger retained context, or now superseded.
- "edit": replace one accepted idea with a more accurate version, especially when it removes outdated or redundant content while preserving unique proof-solving value.
- "add": add one compact corrective insight that is now clearly needed.

Do not prune merely for style. Keep any idea that still provides unique proof-solving value. The question is whether any single idea should be removed or updated due to being outdated, redundant, wrong, harmful, or superseded; if not, return "none".

USER PROBLEM:
{user_prompt}

LEANOJ TEMPLATE:
{lean_template}

{LEANOJ_FORMALIZATION_GUARDRAILS}

ACTIVE TOPIC:
{active_topic}

ALLOCATED LEANOJ PROOF MEMORY:
{_format_context_blocks(context_blocks, fallback_context)}

ACCEPTED BRAINSTORM IDEAS TO REVIEW:
{_format_brainstorm(accepted_ideas)}

{JSON_RULES}
JSON format:
{{"action": "none", "idea_index": null, "new_content": "", "reasoning": "why no prune is needed or why this one operation improves the database"}}
"""


def build_brainstorm_prune_validation_prompt(
    user_prompt: str,
    lean_template: str,
    active_topic: str,
    accepted_ideas: list[str],
    operation: dict[str, Any],
    context_blocks: dict[str, str] | None = None,
) -> str:
    fallback_context = f"CURRENT ACCEPTED IDEAS:\n{_format_brainstorm(accepted_ideas)}"
    return f"""You are the single validator for a proposed LeanOJ brainstorm prune operation.

Validate ONLY whether this operation improves the proof-solving brainstorm database for the exact Lean template and active topic. Use a conservative default: reject if uncertain.

ACCEPT delete only if the selected idea is outdated, wrong, harmful, redundant with stronger retained context, or superseded by stronger retained context.
ACCEPT edit only if the replacement is materially more accurate and still useful, including when it removes outdated or redundant content while preserving unique proof-solving value.
ACCEPT add only if the new content is concrete, non-redundant, and directly useful for the proof.
REJECT vague, stylistic, speculative, or risky changes.

USER PROBLEM:
{user_prompt}

LEANOJ TEMPLATE:
{lean_template}

{LEANOJ_FORMALIZATION_GUARDRAILS}

ACTIVE TOPIC:
{active_topic}

ALLOCATED LEANOJ PROOF MEMORY:
{_format_context_blocks(context_blocks, fallback_context)}

CURRENT ACCEPTED IDEAS:
{_format_brainstorm(accepted_ideas)}

PROPOSED OPERATION:
{operation}

{JSON_RULES}
JSON format:
{{"decision": "reject", "reasoning": "why this prune operation should be accepted or rejected"}}
"""


def build_sufficiency_prompt(
    user_prompt: str,
    lean_template: str,
    accepted_ideas: list[str],
    verified_subproofs: list[dict[str, Any]],
    context_blocks: dict[str, str] | None = None,
) -> str:
    fallback_context = f"""ACCEPTED BRAINSTORM CONTEXT:
{_format_brainstorm(accepted_ideas)}

VERIFIED SUBPROOFS:
{_format_verified_subproofs(verified_subproofs)}"""
    return f"""You are deciding whether there is enough context to attempt solving the user's LeanOJ template now.

This is not final proof validation. Lean 4 will validate the actual proof. Decide whether the accumulated context is likely sufficient to start the final proof loop.

USER PROBLEM:
{user_prompt}

LEANOJ TEMPLATE:
{lean_template}

{LEANOJ_FORMALIZATION_GUARDRAILS}

ALLOCATED LEANOJ PROOF MEMORY:
{_format_context_blocks(context_blocks, fallback_context)}

{JSON_RULES}
JSON format:
{{"enough": true, "reasoning": "why the final loop should or should not start"}}
"""


def build_path_decision_prompt(
    user_prompt: str,
    lean_template: str,
    accepted_ideas: list[str],
    verified_subproofs: list[dict[str, Any]],
    failed_feedback: list[dict[str, Any]],
    context_blocks: dict[str, str] | None = None,
) -> str:
    fallback_context = f"""ACCEPTED BRAINSTORM CONTEXT:
{_format_brainstorm(accepted_ideas)}

VERIFIED SUBPROOFS:
{_format_verified_subproofs(verified_subproofs)}

USEFUL FAILED PROOF FEEDBACK:
{_format_failures(failed_feedback)}"""
    return f"""You are choosing the next path in a LeanOJ proof-solving state machine.

There is no give-up state. Choose one:
- solve_final_now: the system should attempt the final full Lean 4 solution.
- need_more_brainstorming: more cumulative brainstorm context is needed.

When solve_final_now is available, make this decision from the final proof solver's perspective: decide whether the dominant next move toward a solution is to enter the final Lean proof loop now. Since Lean-verified subproofs can now be generated during any brainstorm, defer only to more brainstorming when the final proof path is not yet the strongest next move.

If the current proof memory includes a recent final-cycle packet or working-proof attempt caused by repeated stale `old_string` edits, no-progress watchdog feedback, placeholder/comment churn, or an unresolved missing lemma, choose `need_more_brainstorming` unless the allocated memory already contains fresh concrete proof content that directly resolves that blocker.

USER PROBLEM:
{user_prompt}

LEANOJ TEMPLATE:
{lean_template}

{LEANOJ_FORMALIZATION_GUARDRAILS}

ALLOCATED LEANOJ PROOF MEMORY:
{_format_context_blocks(context_blocks, fallback_context)}

{JSON_RULES}
JSON format:
{{"path": "solve_final_now", "reasoning": "why this path is required", "remaining_questions": ["optional missing questions"]}}
"""


def build_path_validation_prompt(
    user_prompt: str,
    lean_template: str,
    proposed_path: str,
    proposed_reasoning: str,
    accepted_ideas: list[str],
    verified_subproofs: list[dict[str, Any]],
    context_blocks: dict[str, str] | None = None,
) -> str:
    fallback_context = f"""ACCEPTED BRAINSTORM CONTEXT:
{_format_brainstorm(accepted_ideas)}

VERIFIED SUBPROOFS:
{_format_verified_subproofs(verified_subproofs)}"""
    return f"""You are validating a LeanOJ path decision.

Accept only if the proposed path is justified by the current proof-solving context. Reject decisions that try the final proof too early or request more brainstorming when the next proof action is already clear.

VALID PATHS:
- solve_final_now
- need_more_brainstorming

USER PROBLEM:
{user_prompt}

LEANOJ TEMPLATE:
{lean_template}

{LEANOJ_FORMALIZATION_GUARDRAILS}

ALLOCATED LEANOJ PROOF MEMORY:
{_format_context_blocks(context_blocks, fallback_context)}

PROPOSED PATH:
{proposed_path}

PROPOSED REASONING:
{proposed_reasoning}

{JSON_RULES}
JSON format:
{{"decision": "accept", "reasoning": "validation reasoning", "summary": "short rejection feedback if rejected", "corrected_path": "solve_final_now or need_more_brainstorming if rejected"}}
"""

def build_final_solver_prompt(
    user_prompt: str,
    lean_template: str,
    current_master_proof: str,
    master_proof_metadata: dict[str, Any],
    accepted_ideas: list[str],
    verified_subproofs: list[dict[str, Any]],
    partial_proofs: list[dict[str, Any]],
    failed_feedback: list[dict[str, Any]],
    final_attempts: list[dict[str, Any]],
    context_blocks: dict[str, str] | None = None,
) -> str:
    metadata_lines = "\n".join(
        f"- {key}: {value}"
        for key, value in (master_proof_metadata or {}).items()
        if value not in (None, "")
    ) or "[No master proof metadata available.]"
    recent_final_feedback = _format_recent_final_attempts(final_attempts, limit=5)
    fallback_context = f"""ACTIVE PROOF-PLAN NOTES:
{_format_proof_memory_notes(accepted_ideas)}

VERIFIED SUBPROOFS:
{_format_verified_subproofs_for_final(verified_subproofs)}

RECENT EXECUTION FEEDBACK - USE TO CHOOSE THE NEXT EDIT; DO NOT TREAT FAILED CODE AS PROVEN:
{_format_feedback_notes(failed_feedback)}"""
    return f"""You are in the final LeanOJ master-proof editing loop.

Your task is to edit the durable master Lean 4 proof like a paper draft. Preserve the original imports and declarations unless changing them is necessary and allowed by the problem template. Replace required `sorry` holes with real Lean proofs over as many edit prompts as needed.

Master proof route discipline:
- `master_proof.lean` must contain the current chosen proof route only.
- Do not append multiple competing constructions or abandoned approaches into the master proof.
- If a route is refuted or superseded, replace it with the chosen route instead of keeping both.
- Failed constructions may appear only as compact comments when they directly explain an active invariant; otherwise keep them out of the Lean file.
- Use verified standalone lemmas and active proof-plan notes as positive context. Treat refuted-construction warnings only as "do not use" constraints, never as evidence for a proof route.

Correction priority:
- Required corrections take priority over new additions. Treat recent final feedback, Lean errors, exact-string edit rejections, edit-validator feedback, and semantic-review continuation feedback as the next correction targets.
- If any correction is pending, your next edit must address that correction before attempting unrelated new lemmas, fresh proof routes, or speculative additions.
- New additions are allowed only when they directly implement the required correction or provide helper code needed for that correction. Do not expand `master_proof.lean` into a general known-knowledge base of routine helper lemmas or standard Mathlib facts; use standard facts inline when they solve the current obligation.
- In your reasoning, name the correction you addressed. If no correction is pending, state which next unsolved proof obligation your edit advances.

You must choose exactly one action: edit_proof.
This final mode cannot request phase transitions, cannot delegate to planning, and cannot stop early. If the proof is incomplete, make the best concrete edit available and set "needs_more_time": true.

Binary verification gate:
- The system runs Lean after every proposed master proof edit before accepting it into the durable master proof.
- If your edit is useful but the proof still needs more editing time, set "needs_more_time": true. Lean will check the edited file with placeholders allowed, and syntax/type errors will reject the edit.
- If your edit should make the current master proof final-ready, set "needs_more_time": false. Lean will check the edited file with no placeholders allowed, then final integrity/review checks will run.
- A master proof edit is not accepted merely because the string edit applies; it must pass the appropriate Lean gate first.
- A Lean-accepted loophole may be useful intermediate progress, but it is not final-ready. For LeanOJ `answer` definitions, do not terminate with `answer` defined as `sSup`, `csSup`, `Nat.sSup`, `Sup`, or an equivalent maximum over the same feasible set. Final readiness requires an explicit formula/value for `answer n` and a proof that this formula is greatest.
- A continuing edit must change non-comment Lean proof content in a way that discharges, splits, or materially advances an obligation. Do not spend an edit only rewriting comments, TODOs, placeholders, or "prepare for next edit" wording.

Exact-string editing rules:
- Use operation "full_content" only when replacing the whole master proof.
- Use operation "replace", "insert_after", or "delete" for targeted edits.
- For targeted edits, old_string must be copied verbatim from the CURRENT FULL MASTER PROOF and must appear exactly once in that full master proof.
- Include enough surrounding Lean lines in old_string to make the match unique.
- new_string must contain the replacement/insertion Lean code, except delete uses an empty new_string.
- Never introduce fake `axiom`, `constant`, or `opaque` proof devices.
- Final verification requires no `sorry`/`admit`, but intermediate master proof edits may preserve placeholders while you continue working.

USER PROBLEM:
{user_prompt}

LEANOJ TEMPLATE TO SOLVE:
{lean_template}

{LEANOJ_FORMALIZATION_GUARDRAILS}

CURRENT MASTER PROOF METADATA:
{metadata_lines}

RECENT FINAL FEEDBACK (USE TO AVOID REPEATING FAILED EDITS):
{recent_final_feedback}

CURRENT FULL MASTER PROOF TO EDIT (MANDATORY DIRECT-INJECT CONTEXT; NEVER TRUNCATED):
{current_master_proof or lean_template}

ALLOCATED LEANOJ PROOF MEMORY:
{_format_context_blocks(context_blocks, fallback_context)}

{JSON_RULES}
JSON format for continuing edits:
{{"action": "edit_proof", "needs_more_time": true, "operation": "replace", "old_string": "exact unique text from CURRENT MASTER PROOF", "new_string": "updated Lean code", "reasoning": "why this edit advances the proof and what remains"}}

JSON format for final verification after this edit:
{{"action": "edit_proof", "needs_more_time": false, "operation": "replace", "old_string": "exact unique text from CURRENT MASTER PROOF", "new_string": "updated Lean code expected to verify", "reasoning": "why the edited master proof should now pass Lean"}}
"""


def build_master_proof_edit_validation_prompt(
    user_prompt: str,
    lean_template: str,
    current_master_proof: str,
    proposed_master_proof: str,
    edit: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    metrics_lines = "\n".join(
        f"- {key}: {value}"
        for key, value in (metrics or {}).items()
        if value not in (None, "")
    ) or "[No shortening metrics available.]"
    return f"""You are the independent LeanOJ master-proof edit validator.

The final Proof Solver proposed an edit that shortens the durable master proof. Your job is to decide whether this shortening is real proof progress or whether it deletes useful work because the solver is stuck, frustrated, restarting, or giving up.

Accept only if the proposed shorter proof is genuinely progressive for solving the exact LeanOJ template:
- It preserves or strengthens useful solved Lean content, definitions, lemmas, and proof structure.
- It replaces removed material with equivalent or stronger proof content, or removes only clearly redundant/noisy material.
- It still moves toward a complete Lean 4 proof of the original template.

Reject if the edit goes backward:
- It deletes useful proof progress, helper lemmas, explicit formulas, or developed argument structure without a stronger replacement.
- It replaces concrete work with `sorry`, `admit`, comments, vague plans, or a reset toward the original template.
- It looks like abandonment, frustration, a restart, or an attempt to make the file shorter by discarding hard obligations.
- It bloats the master proof by accumulating multiple competing/refuted proof routes instead of maintaining one current chosen route.
- It ignores a required correction in the proof and instead prioritizes unrelated new additions, fresh routes, or speculative helper material.

If you reject, give precise feedback to the proof submitter. Name the content that must be restored or the exact kind of progressive replacement required.
If corrections are required, your feedback must say that those corrections must be fixed before any new addition attempts. New additions are acceptable only when they directly implement the required correction.

If you accept, give a clear justification that can be shown later alongside the old longer proof. This justification must explain:
- WHY the validator allowed the shortening instead of requiring the longer attempt to be restored.
- What the apparent issue was with the old longer attempt, such as redundant code, noisy scaffolding, a weaker route, or content replaced by stronger proof structure.

USER PROBLEM:
{user_prompt}

LEANOJ TEMPLATE:
{lean_template}

{LEANOJ_FORMALIZATION_GUARDRAILS}

PROPOSED EDIT:
- operation: {edit.get("operation", "")}
- needs_more_time: {edit.get("needs_more_time", "")}
- solver_reasoning: {edit.get("reasoning", "")}
- old_string:
{edit.get("old_string", "")}

- new_string:
{edit.get("new_string", "")}

SHORTENING METRICS:
{metrics_lines}

CURRENT MASTER PROOF BEFORE EDIT:
{current_master_proof}

PROPOSED MASTER PROOF AFTER EDIT:
{proposed_master_proof}

{JSON_RULES}
JSON format if this shortening is progressive:
{{"decision": "accept", "reasoning": "why this shorter edit preserves or improves proof progress", "shortening_approval_justification": "clear reason the validator allowed this shortening", "apparent_issue_with_old_attempt": "what was apparently wrong, redundant, noisy, or superseded in the old longer attempt", "feedback_to_submitter": ""}}

JSON format if this shortening goes backward:
{{"decision": "reject", "reasoning": "why this deletes progress or gives up", "feedback_to_submitter": "precise correction for the final solver"}}
"""


def build_final_solution_review_prompt(
    user_prompt: str,
    lean_template: str,
    lean_code: str,
    final_solver_reasoning: str,
    lean_feedback: str,
) -> str:
    return f"""You are the final LeanOJ proof checker for a Lean-accepted submission.

Lean 4 has already checked the code. Your job is NOT to re-run Lean or act as a planning validator. Your job is to decide whether this Lean-accepted file actually solves the user's LeanOJ problem prompt and template in the intended sense.

Lean acceptance is necessary but not sufficient. Reject loopholes that satisfy the weak formal theorem while evading the natural-language task, such as defining an answer by taking a maximum/supremum over the same feasible set instead of determining the requested value in terms of n.

Accept only if the code:
- Preserves and solves the user's LeanOJ template.
- Fully addresses the actual problem prompt, not merely a different formal statement.
- Uses an answer/formulation that genuinely determines the requested object when the problem asks for an explicit value or formula.
- Contains no placeholder proof devices or semantic shortcuts that should remain continuation context instead of the final stop condition.

USER PROBLEM:
{user_prompt}

LEANOJ TEMPLATE:
{lean_template}

{LEANOJ_FORMALIZATION_GUARDRAILS}

LEAN 4 FEEDBACK FROM THE ACCEPTED RUN:
{lean_feedback or "Lean 4 accepted with no diagnostics."}

FINAL SOLVER REASONING BEFORE LEAN CHECK:
{final_solver_reasoning or "[No final solver reasoning provided.]"}

LEAN-ACCEPTED FINAL CODE:
{lean_code}

{JSON_RULES}
JSON format if this is truly solved:
{{"solved": true, "reasoning": "why this Lean-accepted code completely solves the LeanOJ problem prompt and template"}}

JSON format if this is not done:
{{"solved": false, "continuation_feedback": "specific feedback explaining what is missing and what the next final solver attempt should fix", "reasoning": "why Lean acceptance is not enough here"}}
"""
