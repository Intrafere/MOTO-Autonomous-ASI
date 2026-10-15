"""
Shared Lean-4 proof novelty assessment.

The autonomous research `ProofVerificationStage` and the compiler's rigor
submitter both need to classify a freshly verified Lean 4 proof against the
shared novelty tiers. Duplicate detection is local to the stored proof
database, while tier assignment asks whether the verified theorem is novel
relative to standard references, Mathlib, and the user's prompt. Both call
sites share a single helper here so the prompt + context-budget behaviour
stays identical.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from backend.autonomous.prompts.proof_prompts import build_proof_novelty_prompt
from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.config import rag_config
from backend.shared.json_parser import parse_json, sanitize_model_output_for_retry_context
from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_transient_model_call_error,
)
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.response_extraction import extract_message_text
from backend.shared.utils import count_tokens

logger = logging.getLogger(__name__)


VALID_NOVELTY_TIERS = frozenset(
    {
        "not_novel",
        "novel_formulation",
        "novel_variant",
        "mathematical_discovery",
        "major_mathematical_discovery",
    }
)

VALID_NOVELTY_SCOPES = frozenset({"standalone", "group"})


@dataclass(frozen=True)
class ProofNoveltyAssessment:
    novelty_tier: str
    reasoning: str
    novelty_scope: str = "standalone"
    group_claim: str = ""
    group_proof_ids: tuple[str, ...] = ()
    assessment_origin: str = "validator_assessed"

    def __iter__(self):
        """Preserve historical two-value unpacking for callers and tests."""
        yield self.novelty_tier
        yield self.reasoning


def _extract_novelty_content(response: dict[str, Any], label: str) -> str:
    if not response or not response.get("choices"):
        raise ValueError(f"Novelty validator {label} returned no response.")
    message = response["choices"][0].get("message", {})
    content = extract_message_text(message)
    if not content:
        raise ValueError(f"Novelty validator {label} returned empty content.")
    return content


def _parse_novelty_payload(
    content: str,
    *,
    allowed_group_proof_ids: frozenset[str] = frozenset(),
    require_explicit_scope: bool = False,
) -> ProofNoveltyAssessment:
    data = parse_json(content)
    if isinstance(data, list):
        if not data:
            raise ValueError("Novelty validator returned an empty JSON array.")
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("Novelty validator JSON was not an object.")

    raw_tier = str(data.get("novelty_tier", "")).strip().lower()
    if not raw_tier:
        raise ValueError("Novelty validator JSON omitted novelty_tier.")
    if raw_tier not in VALID_NOVELTY_TIERS:
        raise ValueError(f"Novelty validator returned unrecognised tier {raw_tier!r}.")

    reasoning = str(data.get("reasoning", "")).strip()
    if require_explicit_scope and "novelty_scope" not in data:
        raise ValueError("Novelty validator JSON omitted novelty_scope.")
    novelty_scope = str(data.get("novelty_scope", "standalone")).strip().lower()
    if novelty_scope not in VALID_NOVELTY_SCOPES:
        raise ValueError(f"Novelty validator returned unrecognised scope {novelty_scope!r}.")
    raw_group_ids = data.get("group_proof_ids", [])
    if not isinstance(raw_group_ids, list):
        raise ValueError("Novelty validator group_proof_ids must be an array.")
    group_proof_ids = tuple(
        dict.fromkeys(str(proof_id).strip() for proof_id in raw_group_ids if str(proof_id).strip())
    )
    group_claim = str(data.get("group_claim", "")).strip()
    if novelty_scope == "standalone":
        group_proof_ids = ()
        group_claim = ""
    else:
        if not group_proof_ids:
            raise ValueError("Group-scoped novelty requires at least one supporting proof ID.")
        unknown_ids = sorted(set(group_proof_ids) - allowed_group_proof_ids)
        if unknown_ids:
            raise ValueError(
                "Novelty validator cited proof IDs outside the supplied verified group: "
                + ", ".join(unknown_ids)
            )
        if not group_claim:
            raise ValueError("Group-scoped novelty requires a non-empty group_claim.")

    return ProofNoveltyAssessment(
        novelty_tier=raw_tier,
        reasoning=reasoning,
        novelty_scope=novelty_scope,
        group_claim=group_claim,
        group_proof_ids=group_proof_ids,
    )


async def _retry_novelty_payload(
    *,
    prompt: str,
    task_id: str,
    role_id: str,
    validator_model: str,
    validator_context: int,
    validator_max_tokens: int,
    failed_output: str,
    error: Exception,
    allowed_group_proof_ids: frozenset[str],
    require_explicit_scope: bool = False,
) -> ProofNoveltyAssessment:
    logger.info("Novelty validator output failed; attempting bounded JSON retry: %s", error)

    retry_prompt = (
        "Your previous proof-novelty response could not be used.\n\n"
        f"ERROR: {error}\n\n"
        "Return the same novelty decision in valid JSON only. "
        "Use this exact top-level shape:\n"
        "{\n"
        '  "novelty_tier": "not_novel | novel_formulation | '
        'novel_variant | mathematical_discovery | major_mathematical_discovery",\n'
        '  "reasoning": "brief explanation tied to the selected scope",\n'
        '  "novelty_scope": "standalone | group",\n'
        '  "group_claim": "exact combined contribution, or empty for standalone",\n'
        '  "group_proof_ids": ["verified supporting proof IDs, or empty for standalone"]\n'
        "}\n\n"
        "Respond with ONLY the JSON object, no markdown and no explanation."
    )

    max_input_tokens = rag_config.get_available_input_tokens(validator_context, validator_max_tokens)
    failed_output_preview = sanitize_model_output_for_retry_context(
        failed_output,
        max_chars=2000,
    )
    retry_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": failed_output_preview},
        {"role": "user", "content": retry_prompt},
    ]
    if sum(count_tokens(str(message.get("content") or "")) for message in retry_messages) > max_input_tokens:
        retry_messages[1]["content"] = (
            "[failed output omitted because retry context would exceed the model input budget]"
        )
    if sum(count_tokens(str(message.get("content") or "")) for message in retry_messages) > max_input_tokens:
        prompt_with_retry_instruction = f"{prompt}\n\n---\n{retry_prompt}"
        if count_tokens(prompt_with_retry_instruction) <= max_input_tokens:
            retry_messages = [{"role": "user", "content": prompt_with_retry_instruction}]
        else:
            logger.warning(
                "Novelty validator retry instruction too large; retrying original prompt."
            )
            retry_messages = [{"role": "user", "content": prompt}]

    retry_response = await api_client_manager.generate_completion(
        task_id=f"{task_id}_retry",
        role_id=role_id,
        model=validator_model,
        messages=retry_messages,
        max_tokens=validator_max_tokens,
        temperature=0.0,
    )
    retry_content = _extract_novelty_content(retry_response, "retry")
    return _parse_novelty_payload(
        retry_content,
        allowed_group_proof_ids=allowed_group_proof_ids,
        require_explicit_scope=require_explicit_scope,
    )


async def assess_proof_novelty_details(
    *,
    user_prompt: str,
    theorem_statement: str,
    lean_code: str,
    validator_model: str,
    validator_context: int,
    validator_max_tokens: int,
    existing_novel_proofs: str,
    task_id: str,
    verified_group_candidates: str = "",
    allowed_group_proof_ids: frozenset[str] = frozenset(),
    role_id: str = "autonomous_proof_novelty",
) -> ProofNoveltyAssessment:
    """Classify a Lean-4-verified theorem into one of the shared novelty tiers.

    Args:
        user_prompt: Top-level research prompt for context.
        theorem_statement: Human-readable statement of the verified theorem.
        lean_code: Full Lean 4 source that compiled cleanly.
        validator_model: Model identifier to drive the novelty judgement.
        validator_context: Validator model's context window.
        validator_max_tokens: Maximum output tokens reserved for the judgement.
        existing_novel_proofs: Compatibility parameter. Shared registration
            passes an empty string so private proof history cannot influence
            the independent novelty judgment.
        task_id: Caller-chosen task id used for workflow tracking.
        role_id: Role identifier forwarded to the API client manager. Defaults
            to the autonomous role; the compiler rigor caller passes a
            compiler-specific role for correct logging.

    Returns:
        A scoped novelty assessment. Its tier is one of "not_novel",
        "novel_formulation", "novel_variant", "mathematical_discovery", or
        "major_mathematical_discovery". It falls back to a standalone
        "not_novel" assessment only after bounded JSON retry exhaustion.
    """
    prompt = build_proof_novelty_prompt(
        user_prompt=user_prompt,
        theorem_statement=theorem_statement,
        lean_code=lean_code,
        existing_novel_proofs=existing_novel_proofs,
        verified_group_candidates=verified_group_candidates,
    )

    max_input_tokens = rag_config.get_available_input_tokens(validator_context, validator_max_tokens)
    while count_tokens(prompt) > max_input_tokens and len(existing_novel_proofs) > 2000:
        existing_novel_proofs = existing_novel_proofs[
            : max(len(existing_novel_proofs) // 2, 2000)
        ]
        prompt = build_proof_novelty_prompt(
            user_prompt=user_prompt,
            theorem_statement=theorem_statement,
            lean_code=lean_code,
            existing_novel_proofs=existing_novel_proofs,
            verified_group_candidates=verified_group_candidates,
        )
    if count_tokens(prompt) > max_input_tokens and verified_group_candidates:
        logger.warning(
            "Proof novelty group evidence exceeded the configured validator context; "
            "falling back to standalone-only novelty assessment for this verified artifact."
        )
        verified_group_candidates = ""
        allowed_group_proof_ids = frozenset()
        prompt = build_proof_novelty_prompt(
            user_prompt=user_prompt,
            theorem_statement=theorem_statement,
            lean_code=lean_code,
            existing_novel_proofs=existing_novel_proofs,
            verified_group_candidates="",
        )
    if count_tokens(prompt) > max_input_tokens:
        raise ValueError(
            "Proof novelty classification context overflow: the verified theorem and "
            "mandatory standalone classification context do not fit the configured "
            "validator context window."
        )

    response = await api_client_manager.generate_completion(
        task_id=task_id,
        role_id=role_id,
        model=validator_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=validator_max_tokens,
        temperature=0.0,
    )

    content = ""
    try:
        content = _extract_novelty_content(response, "initial")
        return _parse_novelty_payload(
            content,
            allowed_group_proof_ids=allowed_group_proof_ids,
            require_explicit_scope=bool(verified_group_candidates),
        )
    except Exception as exc:
        try:
            return await _retry_novelty_payload(
                prompt=prompt,
                task_id=task_id,
                role_id=role_id,
                validator_model=validator_model,
                validator_context=validator_context,
                validator_max_tokens=validator_max_tokens,
                failed_output=content,
                error=exc,
                allowed_group_proof_ids=allowed_group_proof_ids,
                require_explicit_scope=bool(verified_group_candidates),
            )
        except FreeModelExhaustedError:
            raise
        except RetryableProviderError:
            raise
        except Exception as retry_exc:
            if is_non_retryable_model_error(retry_exc) or is_transient_model_call_error(retry_exc):
                raise
            logger.warning(
                "Novelty validator JSON retry failed; falling back to not_novel: %s",
                retry_exc,
            )
            return ProofNoveltyAssessment(
                novelty_tier="not_novel",
                reasoning=(
                    "Novelty validator JSON retry failed after retry exhaustion: "
                    f"initial error: {exc}; retry error: {retry_exc}"
                ),
                assessment_origin="conservative_fallback",
            )


async def assess_proof_novelty(**kwargs) -> ProofNoveltyAssessment:
    """Return a scoped assessment while preserving two-value unpacking."""
    return await assess_proof_novelty_details(**kwargs)
