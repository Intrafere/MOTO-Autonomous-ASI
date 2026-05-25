"""Shared Lean 4 gate for brainstorm proof candidates."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from backend.autonomous.prompts.proof_prompts import LEAN4_COMMON_PITFALLS
from backend.shared.api_client_manager import api_client_manager
from backend.shared.config import system_config
from backend.shared.json_parser import parse_json
from backend.shared.lean4_client import get_lean4_client
from backend.shared.lean_proof_integrity import validate_full_lean_proof_integrity
from backend.shared.model_error_utils import is_non_retryable_model_error
from backend.shared.models import ProofAttemptFeedback

logger = logging.getLogger(__name__)

BRAINSTORM_LEAN_PROOF_MARKER = "[LEAN 4 VERIFIED BRAINSTORM PROOF]"
NOVEL_PROOF_TIERS = {
    "major_mathematical_discovery",
    "mathematical_discovery",
    "novel_variant",
    "novel_formulation",
}


@dataclass
class BrainstormProofGateResult:
    """Result of checking a proof candidate before normal brainstorm validation."""

    accepted: bool
    submission_content: str = ""
    theorem_statement: str = ""
    theorem_name: str = ""
    formal_sketch: str = ""
    expected_novelty_tier: str = ""
    prompt_relevance_rationale: str = ""
    novelty_rationale: str = ""
    why_not_standard_known_result: str = ""
    lean_code: str = ""
    reasoning: str = ""
    lean_feedback: str = ""
    attempts: list[ProofAttemptFeedback] | None = None
    failure_feedback: str = ""


def is_lean_proof_submission(parsed: dict[str, Any]) -> bool:
    """Return True when a submitter chose the optional Lean proof route."""
    submission_type = str(parsed.get("submission_type") or parsed.get("type") or "").strip().lower()
    if submission_type in {"lean_proof", "proof", "lean4_proof"}:
        return True
    return bool(parsed.get("lean_code")) and bool(parsed.get("theorem_statement") or parsed.get("theorem_or_lemma"))


def _summarize_error(error_output: str, limit: int = 1400) -> str:
    text = " ".join((error_output or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _format_attempts(attempts: list[ProofAttemptFeedback]) -> str:
    if not attempts:
        return "No prior Lean attempts."
    blocks: list[str] = []
    for attempt in attempts[-5:]:
        lean_feedback = (
            attempt.error_output
            or attempt.diagnostic_output
            or attempt.raw_stderr
            or ("Lean accepted this attempt with no diagnostics." if attempt.success else "[none]")
        )
        blocks.extend(
            [
                f"ATTEMPT {attempt.attempt}:",
                f"Reasoning: {attempt.reasoning or '[none]'}",
                "Lean code:",
                attempt.lean_code or "[none]",
                "Lean / integrity feedback:",
                lean_feedback,
                f"Goal states: {attempt.goal_states or '[none]'}",
                "---",
            ]
        )
    return "\n".join(blocks)


def _format_lean_feedback(lean_result: Any) -> str:
    diagnostics = str(getattr(lean_result, "diagnostic_output", "") or "").strip()
    if not diagnostics:
        diagnostics = str(getattr(lean_result, "raw_stderr", "") or "").strip()
    goal_states = str(getattr(lean_result, "goal_states", "") or "").strip()
    parts = []
    if diagnostics:
        parts.append(diagnostics)
    if goal_states:
        parts.append(f"Goal state output:\n{goal_states}")
    return "\n\n".join(parts).strip() or "Lean 4 accepted with no diagnostics."


def _build_retry_prompt(
    *,
    user_prompt: str,
    source_context: str,
    theorem_statement: str,
    formal_sketch: str,
    expected_novelty_tier: str,
    prompt_relevance_rationale: str,
    novelty_rationale: str,
    why_not_standard_known_result: str,
    prior_attempts: list[ProofAttemptFeedback],
) -> str:
    context_excerpt = (source_context or "").strip()
    if len(context_excerpt) > 12000:
        context_excerpt = context_excerpt[:12000] + "\n...[context truncated for proof retry]..."
    return f"""You are repairing a Lean 4 proof candidate for a brainstorm submission.

The previous proof candidate was rejected by Lean 4 or by MOTO's post-Lean integrity gate. Produce a corrected complete Lean 4 proof. Do not use `sorry`, `admit`, or fake `axiom`/`constant`/`opaque` proof devices.

{LEAN4_COMMON_PITFALLS}

USER PROMPT:
{user_prompt}

INTENDED THEOREM STATEMENT:
{theorem_statement}

FORMALIZATION NOTES:
{formal_sketch or "[none]"}

EXPECTED NOVELTY TIER:
{expected_novelty_tier}

PROMPT RELEVANCE RATIONALE:
{prompt_relevance_rationale}

NOVELTY RATIONALE:
{novelty_rationale}

WHY THIS IS NOT A STANDARD KNOWN RESULT:
{why_not_standard_known_result}

BRAINSTORM CONTEXT EXCERPT:
{context_excerpt or "[none]"}

PRIOR ATTEMPTS AND FEEDBACK:
{_format_attempts(prior_attempts)}

Respond with ONLY valid JSON:
{{
  "theorem_name": "Lean declaration name, if named",
  "theorem_statement": "natural-language theorem statement being proved",
  "formal_sketch": "updated formalization notes",
  "expected_novelty_tier": "{expected_novelty_tier}",
  "prompt_relevance_rationale": "{prompt_relevance_rationale}",
  "novelty_rationale": "{novelty_rationale}",
  "why_not_standard_known_result": "{why_not_standard_known_result}",
  "lean_code": "complete Lean 4 code",
  "reasoning": "brief explanation of the repair"
}}
"""


def _build_submission_content(
    *,
    theorem_statement: str,
    formal_sketch: str,
    expected_novelty_tier: str,
    prompt_relevance_rationale: str,
    novelty_rationale: str,
    why_not_standard_known_result: str,
    lean_code: str,
    reasoning: str,
    lean_feedback: str,
    attempts: list[ProofAttemptFeedback],
) -> str:
    attempt_count = len(attempts)
    sections = [
        BRAINSTORM_LEAN_PROOF_MARKER,
        "",
        "Lean 4 has accepted the following proof before this submission reached the brainstorm validator. The validator should still decide whether it is useful, non-redundant brainstorm progress.",
        "",
        f"Theorem statement: {theorem_statement}",
    ]
    if formal_sketch:
        sections.extend(["", f"Formalization notes: {formal_sketch}"])
    if expected_novelty_tier:
        sections.extend(["", f"Expected novelty tier: {expected_novelty_tier}"])
    if prompt_relevance_rationale:
        sections.extend(["", f"Prompt relevance rationale: {prompt_relevance_rationale}"])
    if novelty_rationale:
        sections.extend(["", f"Novelty rationale: {novelty_rationale}"])
    if why_not_standard_known_result:
        sections.extend([
            "",
            f"Why this is not merely standard known mathematics: {why_not_standard_known_result}",
        ])
    if reasoning:
        sections.extend(["", f"Submitter reasoning: {reasoning}"])
    sections.extend(
        [
            "",
            f"Lean verification: accepted after {attempt_count} attempt{'s' if attempt_count != 1 else ''}.",
            f"Lean verifier feedback: {lean_feedback}",
            "",
            "Lean 4 code:",
            "```lean",
            lean_code,
            "```",
        ]
    )
    return "\n".join(sections).strip()


async def verify_brainstorm_proof_candidate(
    *,
    parsed: dict[str, Any],
    user_prompt: str,
    source_context: str,
    model_id: str,
    role_id: str,
    task_id_prefix: str,
    max_tokens: int,
    validator_model: Optional[str],
    validator_context: int,
    validator_max_tokens: int,
    validator_role_id: str,
    allowed_baseline: str = "",
    max_attempts: int = 5,
) -> BrainstormProofGateResult:
    """Lean-check a brainstorm proof candidate before it reaches the validator."""
    theorem_statement = str(parsed.get("theorem_statement") or parsed.get("theorem_or_lemma") or parsed.get("submission") or "").strip()
    formal_sketch = str(parsed.get("formal_sketch") or parsed.get("proof_sketch") or "").strip()
    theorem_name = str(parsed.get("theorem_name") or "").strip()
    expected_novelty_tier = str(parsed.get("expected_novelty_tier") or "").strip().lower()
    prompt_relevance_rationale = str(parsed.get("prompt_relevance_rationale") or "").strip()
    novelty_rationale = str(parsed.get("novelty_rationale") or "").strip()
    why_not_standard_known_result = str(parsed.get("why_not_standard_known_result") or "").strip()
    lean_code = str(parsed.get("lean_code") or "").strip()
    reasoning = str(parsed.get("reasoning") or "").strip()

    if not theorem_statement or not lean_code:
        return BrainstormProofGateResult(
            accepted=False,
            theorem_statement=theorem_statement,
            lean_code=lean_code,
            reasoning=reasoning,
            failure_feedback=(
                "Lean proof candidate was malformed: both `theorem_statement` and `lean_code` "
                "are required. Start the next brainstorm attempt fresh."
            ),
            attempts=[],
        )
    if expected_novelty_tier not in NOVEL_PROOF_TIERS:
        expected_novelty_tier = expected_novelty_tier or "not_novel"

    attempts: list[ProofAttemptFeedback] = []
    current = {
        "theorem_statement": theorem_statement,
        "formal_sketch": formal_sketch,
        "theorem_name": theorem_name,
        "expected_novelty_tier": expected_novelty_tier,
        "prompt_relevance_rationale": prompt_relevance_rationale,
        "novelty_rationale": novelty_rationale,
        "why_not_standard_known_result": why_not_standard_known_result,
        "lean_code": lean_code,
        "reasoning": reasoning,
    }

    for attempt_number in range(1, max(1, max_attempts) + 1):
        theorem_statement = str(current.get("theorem_statement") or theorem_statement).strip()
        formal_sketch = str(current.get("formal_sketch") or formal_sketch).strip()
        theorem_name = str(current.get("theorem_name") or theorem_name).strip()
        expected_novelty_tier = str(current.get("expected_novelty_tier") or expected_novelty_tier).strip()
        prompt_relevance_rationale = str(current.get("prompt_relevance_rationale") or prompt_relevance_rationale).strip()
        novelty_rationale = str(current.get("novelty_rationale") or novelty_rationale).strip()
        why_not_standard_known_result = str(
            current.get("why_not_standard_known_result") or why_not_standard_known_result
        ).strip()
        lean_code = str(current.get("lean_code") or "").strip()
        reasoning = str(current.get("reasoning") or reasoning).strip()

        lean_result = await get_lean4_client().check_proof(
            lean_code,
            timeout=system_config.lean4_proof_timeout,
        )
        feedback = ProofAttemptFeedback(
            attempt=attempt_number,
            theorem_id="brainstorm_inline_proof",
            reasoning=reasoning,
            lean_code=lean_code,
            error_output=lean_result.error_output,
            diagnostic_output=str(getattr(lean_result, "diagnostic_output", "") or ""),
            goal_states=lean_result.goal_states,
            raw_stderr=str(getattr(lean_result, "raw_stderr", "") or ""),
            strategy="full_script",
            success=lean_result.success,
        )

        if lean_result.success:
            lean_feedback = _format_lean_feedback(lean_result)
            integrity = await validate_full_lean_proof_integrity(
                user_prompt=user_prompt,
                theorem_statement=theorem_statement,
                formal_sketch=formal_sketch,
                lean_code=lean_code,
                source_excerpt=source_context or theorem_statement,
                allowed_baseline=allowed_baseline,
                validator_model=validator_model,
                validator_context=validator_context,
                validator_max_tokens=validator_max_tokens,
                task_id=f"{task_id_prefix}_integrity_{attempt_number}",
                role_id=validator_role_id,
                require_statement_alignment=True,
            )
            if integrity.valid:
                stored_theorem_statement = (
                    integrity.actual_theorem_statement.strip()
                    or theorem_statement
                )
                stored_theorem_name = (
                    integrity.actual_theorem_name.strip()
                    or theorem_name
                )
                stored_formal_sketch = formal_sketch
                if integrity.category in {"statement_downshifted", "statement_alignment_uncertain", "statement_alignment_unavailable"}:
                    stored_formal_sketch = (
                        f"{stored_formal_sketch}\n\n"
                        f"Original intended theorem candidate: {theorem_statement}\n"
                        f"Statement-alignment classification: {integrity.category}. "
                        f"{integrity.reason or integrity.downshift_reason}"
                    ).strip()
                    lean_feedback = (
                        f"{lean_feedback}\n\n"
                        "MOTO preservation note: Lean accepted this proof. "
                        f"It is stored under the actual proved statement because {integrity.category}: "
                        f"{integrity.reason or integrity.downshift_reason}"
                    ).strip()
                feedback.success = True
                feedback.error_output = ""
                attempts.append(feedback)
                return BrainstormProofGateResult(
                    accepted=True,
                    submission_content=_build_submission_content(
                        theorem_statement=stored_theorem_statement,
                        formal_sketch=stored_formal_sketch,
                        expected_novelty_tier=expected_novelty_tier,
                        prompt_relevance_rationale=prompt_relevance_rationale,
                        novelty_rationale=novelty_rationale,
                        why_not_standard_known_result=why_not_standard_known_result,
                        lean_code=lean_code,
                        reasoning=reasoning,
                        lean_feedback=lean_feedback,
                        attempts=attempts,
                    ),
                    theorem_statement=stored_theorem_statement,
                    theorem_name=stored_theorem_name,
                    formal_sketch=stored_formal_sketch,
                    expected_novelty_tier=expected_novelty_tier,
                    prompt_relevance_rationale=prompt_relevance_rationale,
                    novelty_rationale=novelty_rationale,
                    why_not_standard_known_result=why_not_standard_known_result,
                    lean_code=lean_code,
                    reasoning=reasoning,
                    lean_feedback=lean_feedback,
                    attempts=attempts,
                )

            feedback.success = False
            feedback.error_output = integrity.reason

        attempts.append(feedback)
        if attempt_number >= max_attempts:
            break

        prompt = _build_retry_prompt(
            user_prompt=user_prompt,
            source_context=source_context,
            theorem_statement=theorem_statement,
            formal_sketch=formal_sketch,
            expected_novelty_tier=expected_novelty_tier,
            prompt_relevance_rationale=prompt_relevance_rationale,
            novelty_rationale=novelty_rationale,
            why_not_standard_known_result=why_not_standard_known_result,
            prior_attempts=attempts,
        )
        try:
            response = await api_client_manager.generate_completion(
                task_id=f"{task_id_prefix}_repair_{attempt_number + 1}",
                role_id=role_id,
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            if not response or not response.get("choices"):
                raise ValueError("Proof repair model returned no choices.")
            message = response["choices"][0].get("message", {})
            content = message.get("content") or message.get("reasoning") or ""
            repaired = parse_json(content)
            if isinstance(repaired, list):
                repaired = repaired[0] if repaired else {}
            if not isinstance(repaired, dict):
                raise ValueError("Proof repair response was not a JSON object.")
            current = {
                "theorem_statement": str(repaired.get("theorem_statement") or theorem_statement).strip(),
                "formal_sketch": str(repaired.get("formal_sketch") or formal_sketch).strip(),
                "theorem_name": str(repaired.get("theorem_name") or theorem_name).strip(),
                "expected_novelty_tier": str(
                    repaired.get("expected_novelty_tier") or expected_novelty_tier
                ).strip(),
                "prompt_relevance_rationale": str(
                    repaired.get("prompt_relevance_rationale") or prompt_relevance_rationale
                ).strip(),
                "novelty_rationale": str(
                    repaired.get("novelty_rationale") or novelty_rationale
                ).strip(),
                "why_not_standard_known_result": str(
                    repaired.get("why_not_standard_known_result") or why_not_standard_known_result
                ).strip(),
                "lean_code": str(repaired.get("lean_code") or "").strip(),
                "reasoning": str(repaired.get("reasoning") or "").strip(),
            }
        except Exception as exc:
            if is_non_retryable_model_error(exc):
                raise
            logger.warning("Brainstorm proof repair attempt setup failed: %s", exc)
            current = {
                "theorem_statement": theorem_statement,
                "formal_sketch": formal_sketch,
                "theorem_name": theorem_name,
                "expected_novelty_tier": expected_novelty_tier,
                "prompt_relevance_rationale": prompt_relevance_rationale,
                "novelty_rationale": novelty_rationale,
                "why_not_standard_known_result": why_not_standard_known_result,
                "lean_code": lean_code,
                "reasoning": f"Prior proof repair call failed before Lean verification: {exc}",
            }

    last_error = attempts[-1].error_output if attempts else "No Lean attempts completed."
    return BrainstormProofGateResult(
        accepted=False,
        theorem_statement=theorem_statement,
        theorem_name=theorem_name,
        formal_sketch=formal_sketch,
        expected_novelty_tier=expected_novelty_tier,
        prompt_relevance_rationale=prompt_relevance_rationale,
        novelty_rationale=novelty_rationale,
        why_not_standard_known_result=why_not_standard_known_result,
        lean_code=lean_code,
        reasoning=reasoning,
        attempts=attempts,
        failure_feedback=(
            "Lean proof candidate failed the 5-attempt brainstorm proof gate. "
            f"Last feedback: {_summarize_error(last_error)}. Start the next brainstorm attempt with a fresh useful question or idea."
        ),
    )
