"""Fail-closed proposer and Validator agents for proof live-context pruning."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Awaitable, Callable, Optional, TypeVar

from pydantic import ValidationError

from backend.autonomous.prompts.proof_pruning_prompts import (
    build_proof_pruning_proposer_prompt,
    build_proof_pruning_repair_prompt,
    build_proof_pruning_validator_prompt,
    proposer_contract_text,
    validator_contract_text,
)
from backend.shared.api_client_manager import api_client_manager
from backend.shared.config import rag_config
from backend.shared.json_parser import (
    parse_json,
    sanitize_model_output_for_retry_context,
)
from backend.shared.models import (
    ModelConfig,
    ProofPruneCommitIntent,
    ProofPruneGuardResult,
    ProofPruneProposal,
    ProofPruneReviewResult,
    ProofPruneSnapshot,
    ProofPruneValidation,
    ProofRoleConfigSnapshot,
    ProofRuntimeConfigSnapshot,
)
from backend.shared.response_extraction import extract_message_text
from backend.shared.utils import count_tokens


class ProofPruningError(RuntimeError):
    """Base class for Build 03 pruning-only failures."""


class ProofPruningContractError(ProofPruningError):
    """Both the original and bounded repair response violated the contract."""


class ProofPruningContextError(ProofPruningError):
    """Mandatory snapshot material could not fit the configured role budget."""


class ProofPruningStaleSnapshotError(ProofPruningError):
    """The immutable snapshot no longer matches the canonical proof store."""


T = TypeVar("T")


def proof_run_role_suffix(scope: str, proof_run_id: str) -> str:
    """Return a stable, sanitized, collision-resistant role suffix."""
    normalized_scope = re.sub(r"[^a-z0-9]+", "_", str(scope or "").lower()).strip("_")
    normalized_run = re.sub(
        r"[^a-z0-9]+", "_", str(proof_run_id or "").lower()
    ).strip("_")
    digest = hashlib.sha256(str(proof_run_id or "").encode("utf-8")).hexdigest()[:10]
    readable = normalized_run[:32] or "run"
    return f"{normalized_scope or 'proof'}_{readable}_{digest}"


def model_config_from_proof_role(config: ProofRoleConfigSnapshot) -> ModelConfig:
    """Copy every routed role field without hidden defaults."""
    return ModelConfig(
        provider=config.provider,
        model_id=config.model_id,
        openrouter_model_id=(
            config.model_id if config.provider == "openrouter" else None
        ),
        openrouter_provider=config.openrouter_provider,
        openrouter_reasoning_effort=config.openrouter_reasoning_effort,
        lm_studio_fallback_id=config.lm_studio_fallback_id,
        context_window=config.context_window,
        max_output_tokens=config.max_output_tokens,
        supercharge_enabled=config.supercharge_enabled,
    )


def parse_proof_prune_proposal(content: str) -> ProofPruneProposal:
    data = parse_json(content)
    if not isinstance(data, dict):
        raise ValueError("Proof pruning proposer output must be one JSON object.")
    return ProofPruneProposal.model_validate(data)


def parse_proof_prune_validation(
    content: str,
    *,
    expected_proof_id: str,
) -> ProofPruneValidation:
    data = parse_json(content)
    if not isinstance(data, dict):
        raise ValueError("Proof pruning Validator output must be one JSON object.")
    result = ProofPruneValidation.model_validate(data)
    if result.proof_id != expected_proof_id:
        raise ValueError("Validator proof_id did not match the proposed proof.")
    return result


def validate_proposal_against_snapshot(
    proposal: ProofPruneProposal,
    snapshot: ProofPruneSnapshot,
) -> ProofPruneGuardResult:
    if proposal.action == "no_prune":
        return ProofPruneGuardResult(allowed=True)

    aggregate = {entry.proof_id: entry for entry in snapshot.whole_set}
    target = aggregate.get(str(proposal.proof_id or ""))
    reasons = []
    if target is None:
        reasons.append("target_not_in_snapshot")
    else:
        if not target.eligible_candidate:
            reasons.append("target_not_eligible")
        reasons.extend(target.protected_reasons)
        if proposal.expected_theorem_hash != target.canonical_theorem_hash:
            reasons.append("theorem_hash_mismatch")
        if proposal.expected_lean_hash != target.canonical_lean_hash:
            reasons.append("lean_hash_mismatch")
        if target.exact_identity_occurrence_count < 2:
            reasons.append("no_stronger_exact_identity_comparator")
        if target.dependency_extraction_status != "complete":
            reasons.append("dependency_extraction_incomplete")
        if target.dependent_count:
            reasons.append("active_dependency_root")
    return ProofPruneGuardResult(
        allowed=not reasons,
        proof_id=proposal.proof_id,
        reasons=list(dict.fromkeys(reasons)),
    )


class _ProofPruningRole:
    def __init__(
        self,
        *,
        role_id: str,
        task_prefix: str,
        role_config: ProofRoleConfigSnapshot,
    ) -> None:
        self.role_id = role_id
        self.task_prefix = task_prefix
        self.role_config = role_config
        self.task_sequence = 0
        api_client_manager.configure_role(
            role_id,
            model_config_from_proof_role(role_config),
        )

    def _task_id(self) -> str:
        self.task_sequence += 1
        return f"{self.task_prefix}_{self.task_sequence:03d}"

    @staticmethod
    def _extract_content(response) -> str:
        if not response or not response.get("choices"):
            raise ValueError("Proof pruning role returned no model choices.")
        content = extract_message_text(response["choices"][0].get("message", {}))
        if not content:
            raise ValueError("Proof pruning role returned empty model output.")
        return content

    async def _generate_parse_with_one_repair(
        self,
        *,
        prompt: str,
        contract: str,
        parser: Callable[[str], T],
    ) -> T:
        max_input_tokens = rag_config.get_available_input_tokens(
            self.role_config.context_window,
            self.role_config.max_output_tokens,
        )
        prompt_tokens = count_tokens(prompt)
        if prompt_tokens > max_input_tokens:
            raise ProofPruningContextError(
                "Mandatory proof-pruning context exceeds the configured input "
                f"budget ({prompt_tokens} > {max_input_tokens})."
            )

        task_id = self._task_id()
        response = await api_client_manager.generate_completion(
            task_id=task_id,
            role_id=self.role_id,
            model=self.role_config.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.role_config.max_output_tokens,
            temperature=0.0,
        )
        content = self._extract_content(response)
        try:
            return parser(content)
        except (ValueError, TypeError, ValidationError) as first_error:
            visible_output = sanitize_model_output_for_retry_context(
                content,
                max_chars=2000,
            )
            repair_prompt = build_proof_pruning_repair_prompt(
                contract=contract,
                error_summary=str(first_error),
            )
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": visible_output},
                {"role": "user", "content": repair_prompt},
            ]
            retry_tokens = sum(count_tokens(message["content"]) for message in messages)
            if retry_tokens > max_input_tokens:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": repair_prompt},
                ]
                retry_tokens = sum(
                    count_tokens(message["content"]) for message in messages
                )
            if retry_tokens > max_input_tokens:
                raise ProofPruningContextError(
                    "The bounded proof-pruning repair prompt does not fit the "
                    "configured input budget."
                ) from first_error
            retry_response = await api_client_manager.generate_completion(
                task_id=f"{task_id}_retry",
                role_id=self.role_id,
                model=self.role_config.model_id,
                messages=messages,
                max_tokens=self.role_config.max_output_tokens,
                temperature=0.0,
            )
            retry_content = self._extract_content(retry_response)
            try:
                return parser(retry_content)
            except (ValueError, TypeError, ValidationError) as retry_error:
                raise ProofPruningContractError(
                    "Proof-pruning output remained contract-invalid after one "
                    "bounded repair."
                ) from retry_error


class ProofPruningProposerAgent(_ProofPruningRole):
    async def propose(self, snapshot: ProofPruneSnapshot) -> ProofPruneProposal:
        prompt = build_proof_pruning_proposer_prompt(snapshot, include_lean=True)
        max_input = rag_config.get_available_input_tokens(
            self.role_config.context_window,
            self.role_config.max_output_tokens,
        )
        if count_tokens(prompt) > max_input:
            prompt = build_proof_pruning_proposer_prompt(
                snapshot,
                include_lean=False,
            )
        return await self._generate_parse_with_one_repair(
            prompt=prompt,
            contract=proposer_contract_text(),
            parser=parse_proof_prune_proposal,
        )


class ProofPruningValidatorAgent(_ProofPruningRole):
    async def validate(
        self,
        snapshot: ProofPruneSnapshot,
        proposal: ProofPruneProposal,
        guard: ProofPruneGuardResult,
    ) -> ProofPruneValidation:
        guard_summary = json.dumps(
            guard.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        prompt = build_proof_pruning_validator_prompt(
            snapshot,
            proposal,
            guard_summary=guard_summary,
            include_lean=True,
        )
        max_input = rag_config.get_available_input_tokens(
            self.role_config.context_window,
            self.role_config.max_output_tokens,
        )
        if count_tokens(prompt) > max_input:
            prompt = build_proof_pruning_validator_prompt(
                snapshot,
                proposal,
                guard_summary=guard_summary,
                include_lean=False,
            )
        return await self._generate_parse_with_one_repair(
            prompt=prompt,
            contract=validator_contract_text(),
            parser=lambda content: parse_proof_prune_validation(
                content,
                expected_proof_id=str(proposal.proof_id),
            ),
        )


class ProofPruningReviewService:
    """Run one side-effect-free proposer -> guard -> Validator review."""

    def __init__(
        self,
        *,
        runtime_snapshot: ProofRuntimeConfigSnapshot,
        scope: str,
        proof_run_id: str,
    ) -> None:
        suffix = proof_run_role_suffix(scope, proof_run_id)
        self.proposer = ProofPruningProposerAgent(
            role_id=f"autonomous_proof_prune_proposer_{suffix}",
            task_prefix="proof_prune_propose",
            role_config=runtime_snapshot.paper,
        )
        self.validator = ProofPruningValidatorAgent(
            role_id=f"autonomous_proof_prune_validator_{suffix}",
            task_prefix="proof_prune_validate",
            role_config=runtime_snapshot.validator,
        )

    async def review(
        self,
        snapshot: ProofPruneSnapshot,
        *,
        current_revision: Optional[int] = None,
        event_callback: Optional[
            Callable[[str, dict], Awaitable[None]]
        ] = None,
    ) -> ProofPruneReviewResult:
        proposal = await self.proposer.propose(snapshot)
        if proposal.action == "no_prune":
            return ProofPruneReviewResult(
                outcome="no_prune",
                proposal=proposal,
            )

        guard = validate_proposal_against_snapshot(proposal, snapshot)
        if event_callback is not None:
            await event_callback("proposed", proposal.model_dump(mode="json"))
        if not guard.allowed:
            return ProofPruneReviewResult(
                outcome="rejected",
                proposal=proposal,
                validation=ProofPruneValidation(
                    decision="reject",
                    proof_id=str(proposal.proof_id),
                    reasoning=(
                        "Deterministic proof identity, dependency, or protection "
                        "guards rejected the proposal before semantic validation: "
                        + ", ".join(guard.reasons)
                    )[:4000],
                ),
            )
        validation = await self.validator.validate(snapshot, proposal, guard)
        if validation.decision == "reject":
            return ProofPruneReviewResult(
                outcome="rejected",
                proposal=proposal,
                validation=validation,
            )
        if (
            current_revision is not None
            and current_revision != snapshot.proof_set_revision
        ):
            raise ProofPruningStaleSnapshotError(
                "The proof set changed while pruning validation was in flight."
            )
        return ProofPruneReviewResult(
            outcome="commit_intent",
            proposal=proposal,
            validation=validation,
            commit_intent=ProofPruneCommitIntent(
                snapshot_id=snapshot.snapshot_id,
                proof_id=str(proposal.proof_id),
                owning_run_id=snapshot.owning_run_id,
                proof_set_revision=snapshot.proof_set_revision,
                expected_theorem_hash=str(proposal.expected_theorem_hash),
                expected_lean_hash=str(proposal.expected_lean_hash),
                trigger_reasons=snapshot.trigger_reasons,
                proposer_reasoning=proposal.reasoning,
                validator_reasoning=validation.reasoning,
            ),
        )
