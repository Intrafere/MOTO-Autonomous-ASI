"""
Shared Lean-4 proof novelty assessment.

The autonomous research `ProofVerificationStage` and the compiler's rigor
submitter both need to classify a freshly verified Lean 4 proof as novel
(first time this system has produced it) or known (duplicates a result
already in the proof database). Both call sites share a single helper
here so the prompt + context-budget behaviour stays identical.
"""
from __future__ import annotations

import logging
from typing import Tuple

from backend.autonomous.prompts.proof_prompts import build_proof_novelty_prompt
from backend.shared.api_client_manager import api_client_manager
from backend.shared.json_parser import parse_json
from backend.shared.utils import count_tokens

logger = logging.getLogger(__name__)


async def assess_proof_novelty(
    *,
    user_prompt: str,
    theorem_statement: str,
    lean_code: str,
    validator_model: str,
    validator_context: int,
    validator_max_tokens: int,
    existing_novel_proofs: str,
    task_id: str,
    role_id: str = "autonomous_proof_novelty",
) -> Tuple[bool, str]:
    """Classify a Lean-4-verified theorem as novel or known.

    Args:
        user_prompt: Top-level research prompt for context.
        theorem_statement: Human-readable statement of the verified theorem.
        lean_code: Full Lean 4 source that compiled cleanly.
        validator_model: Model identifier to drive the novelty judgement.
        validator_context: Validator model's context window.
        validator_max_tokens: Maximum output tokens reserved for the judgement.
        existing_novel_proofs: Pre-formatted block listing already-novel proofs;
            trimmed in-loop if it overflows the validator budget.
        task_id: Caller-chosen task id used for workflow tracking.
        role_id: Role identifier forwarded to the API client manager. Defaults
            to the autonomous role; the compiler rigor caller passes a
            compiler-specific role for correct logging.

    Returns:
        Tuple of (is_novel, reasoning). Falls back to (False, <message>) when
        the novelty validator returns no usable response.
    """
    prompt = build_proof_novelty_prompt(
        user_prompt=user_prompt,
        theorem_statement=theorem_statement,
        lean_code=lean_code,
        existing_novel_proofs=existing_novel_proofs,
    )

    max_input_tokens = validator_context - validator_max_tokens
    while count_tokens(prompt) > max_input_tokens and len(existing_novel_proofs) > 2000:
        existing_novel_proofs = existing_novel_proofs[
            : max(len(existing_novel_proofs) // 2, 2000)
        ]
        prompt = build_proof_novelty_prompt(
            user_prompt=user_prompt,
            theorem_statement=theorem_statement,
            lean_code=lean_code,
            existing_novel_proofs=existing_novel_proofs,
        )

    response = await api_client_manager.generate_completion(
        task_id=task_id,
        role_id=role_id,
        model=validator_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=validator_max_tokens,
        temperature=0.0,
    )
    if not response or not response.get("choices"):
        return False, "Novelty validator returned no response."

    message = response["choices"][0].get("message", {})
    content = message.get("content") or message.get("reasoning") or ""
    if not content:
        return False, "Novelty validator returned empty content."

    try:
        data = parse_json(content)
    except Exception as exc:
        logger.warning("Novelty validator JSON parse failed: %s", exc)
        return False, f"Novelty validator JSON parse error: {exc}"

    if isinstance(data, list):
        data = data[0] if data else {}

    return bool(data.get("is_novel", False)), str(data.get("reasoning", "")).strip()
