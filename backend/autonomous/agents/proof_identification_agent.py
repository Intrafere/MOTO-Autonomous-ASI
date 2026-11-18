"""
Proof identification agent for Lean 4 verification checkpoints.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.json_key_recovery import recover_key_labels
from backend.shared.json_parser import parse_json, sanitize_model_output_for_retry_context
from backend.shared.model_output_contract import (
    ItemContractProblem,
    ModelOutputContractError,
    report_model_output_normalizations,
)
from backend.shared.response_extraction import extract_message_text
from backend.shared.model_error_utils import (
    is_retryable_model_output_error,
    is_non_retryable_model_error,
    is_transient_model_call_error,
)
from backend.shared.models import ProofCandidate
from backend.shared.proof_kind_policy import ProofKindPolicy
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.utils import count_tokens
from backend.shared.config import rag_config
from backend.shared.prompt_feedback_budget import (
    fit_prompt_with_feedback_streams,
    shed_oldest_feedback,
)
from backend.shared.provider_errors import ProviderContextLengthError
from backend.autonomous.prompts.proof_prompts import (
    PROOF_IDENTIFICATION_RESPONSE_SCHEMA,
    PROOF_IDENTIFICATION_THEOREM_SCHEMA,
    build_proof_identification_prompt,
    build_smt_translation_prompt,
)

logger = logging.getLogger(__name__)

_NOVEL_PROOF_TIERS = {
    "major_mathematical_discovery",
    "mathematical_discovery",
    "novel_variant",
    "novel_formulation",
}
_REQUIRED_RATIONALE_FIELDS = (
    "prompt_relevance_rationale",
    "novelty_rationale",
    "why_not_standard_known_result",
)
_THEOREM_KNOWN_KEYS = frozenset(
    name
    for spec in PROOF_IDENTIFICATION_THEOREM_SCHEMA.keys
    for name in (spec.name, *spec.aliases)
)


def model_output_syntax_error(exc: ValueError, *, role: str = "") -> ModelOutputContractError:
    """Type a shared-parser failure as a syntax-stage contract error.

    Parser messages are system-authored diagnostics without output excerpts, so
    truncation and no-JSON explanations stay in the model-facing retry text.
    """
    if isinstance(exc, json.JSONDecodeError):
        return ModelOutputContractError.syntax(exc, role=role)
    return ModelOutputContractError(
        f"The response was not usable JSON: {str(exc)[:400]}",
        role=role,
        stage="syntax",
    )


def _retry_proposal_kind_line(proof_kinds: ProofKindPolicy) -> str:
    """The proposal_kind line of the retry shape, naming only allowed kinds."""
    if not proof_kinds.master:
        return ""
    kinds = "individual | master_proof" if proof_kinds.single_shot else "master_proof"
    return f',\n      "proposal_kind": "{kinds}"'


class ProofIdentificationContextError(ValueError):
    """Mandatory proof-discovery context exceeds the configured local budget."""

    def __init__(
        self,
        message: str,
        *,
        prompt_tokens: int,
        max_input_tokens: int,
    ) -> None:
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.max_input_tokens = max_input_tokens


class ProofIdentificationAgent:
    """Find complete theorem candidates in a brainstorm or paper."""

    def __init__(
        self,
        model_id: str,
        context_window: int,
        max_output_tokens: int,
        role_id: str,
        solution_path_manager: Any = None,
    ) -> None:
        self.model_id = model_id
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.role_id = role_id
        self.solution_path_manager = solution_path_manager
        self.task_sequence = 0

    def get_current_task_id(self) -> str:
        return f"proof_id_{self.task_sequence:03d}"

    @staticmethod
    def _extract_response_content(response: Dict[str, Any]) -> str:
        if not response or not response.get("choices"):
            raise ValueError("Proof identification returned no model choices.")
        message = response["choices"][0].get("message", {})
        content = extract_message_text(message)
        if not content:
            raise ValueError("Proof identification returned empty model output.")
        return content

    @staticmethod
    def _parse_candidate_payload(
        content: str,
        *,
        role: str = "",
        repairs: Optional[list] = None,
    ) -> Tuple[bool, List[ProofCandidate]]:
        """Parse the identification JSON; any malformed entry fails the whole list.

        ``repairs`` receives each visible normalization applied (stray characters,
        trailing commas, one relabeled key per theorem entry). Contract failures
        raise ``ModelOutputContractError`` naming each bad entry and field.
        """
        applied: list = []
        # Repairs run only for a caller that publishes them.
        collect = repairs is not None
        try:
            data = parse_json(content, repairs=applied if collect else None)
        except ValueError as exc:
            raise model_output_syntax_error(exc, role=role) from exc
        if isinstance(data, list):
            if not data:
                raise ModelOutputContractError(
                    "Proof identification returned an empty JSON array",
                    role=role,
                )
            data = data[0]
        if not isinstance(data, dict):
            raise ModelOutputContractError(
                "Proof identification returned JSON that was not an object",
                role=role,
            )
        if collect:
            data, key_repairs = recover_key_labels(data, PROOF_IDENTIFICATION_RESPONSE_SCHEMA)
            applied.extend(key_repairs)
            repairs.extend(applied)
        if "has_provable_theorems" not in data:
            raise ModelOutputContractError(
                "Proof identification JSON omitted has_provable_theorems",
                role=role,
                stage="missing_key",
                repairs_applied=applied,
            )
        if not isinstance(data.get("has_provable_theorems"), bool):
            raise ModelOutputContractError(
                "Proof identification has_provable_theorems must be a boolean",
                role=role,
                stage="invalid_value",
                repairs_applied=applied,
            )

        has_candidates = data["has_provable_theorems"]
        raw_theorems_value = data.get("theorems", [])
        if raw_theorems_value is None:
            raw_theorems_value = []
        if not isinstance(raw_theorems_value, list):
            raise ModelOutputContractError(
                "Proof identification theorems must be an array",
                role=role,
                stage="invalid_value",
                repairs_applied=applied,
            )
        if has_candidates and not raw_theorems_value:
            raise ModelOutputContractError(
                "Proof identification claimed provable theorems but returned no theorem entries",
                role=role,
                stage="missing_key",
                repairs_applied=applied,
            )

        theorem_candidates: List[ProofCandidate] = []
        problems: List[ItemContractProblem] = []
        non_novel_candidate_count = 0
        seen_theorem_ids: set[str] = set()
        for index, theorem in enumerate(raw_theorems_value):
            item_path = f"theorems[{index}]"
            if not isinstance(theorem, dict):
                problems.append(
                    ItemContractProblem(item_path=item_path, detail="entry was not a JSON object")
                )
                continue
            theorem_id = str(theorem.get("theorem_id") or theorem.get("id") or "").strip()
            missing: list[str] = []
            invalid: list[str] = []
            detail = ""
            if not theorem_id:
                missing.append("theorem_id")
            elif theorem_id in seen_theorem_ids:
                invalid.append("theorem_id")
                detail = "theorem_id repeats an earlier entry"
            statement = str(theorem.get("statement", "")).strip()
            if not statement:
                missing.append("statement")
            expected_novelty_tier = str(theorem.get("expected_novelty_tier", "")).strip().lower()
            if expected_novelty_tier == "not_novel" and not missing and not invalid:
                seen_theorem_ids.add(theorem_id)
                non_novel_candidate_count += 1
                logger.info(
                    "ProofIdentificationAgent skipped theorem %s because it was marked not_novel.",
                    theorem_id,
                )
                continue
            if not expected_novelty_tier:
                missing.append("expected_novelty_tier")
            elif expected_novelty_tier not in _NOVEL_PROOF_TIERS:
                invalid.append("expected_novelty_tier")
            rationales = {
                field_name: str(theorem.get(field_name, "")).strip()
                for field_name in _REQUIRED_RATIONALE_FIELDS
            }
            missing.extend(name for name, value in rationales.items() if not value)
            if theorem_id:
                seen_theorem_ids.add(theorem_id)
            if missing or invalid:
                problems.append(
                    ItemContractProblem(
                        item_path=item_path,
                        item_id=theorem_id,
                        missing_keys=tuple(missing),
                        unknown_keys=tuple(
                            key for key in theorem if key not in _THEOREM_KNOWN_KEYS
                        ),
                        invalid_fields=tuple(invalid),
                        detail=detail,
                    )
                )
                continue
            theorem_candidates.append(
                ProofCandidate(
                    theorem_id=theorem_id,
                    statement=statement,
                    formal_sketch=str(theorem.get("formal_sketch", "")).strip(),
                    expected_novelty_tier=expected_novelty_tier,
                    prompt_relevance_rationale=rationales["prompt_relevance_rationale"],
                    novelty_rationale=rationales["novelty_rationale"],
                    why_not_standard_known_result=rationales["why_not_standard_known_result"],
                    retry_open_target_id=str(
                        theorem.get("retry_open_target_id") or ""
                    ).strip(),
                    proposal_kind=(
                        "master_proof"
                        if str(theorem.get("proposal_kind") or "").strip().lower() == "master_proof"
                        else "individual"
                    ),
                )
            )

        if has_candidates and problems:
            # The candidate-list Validator judges whole lists, so one bad entry
            # fails the list; the error names every bad entry and field.
            total = len(raw_theorems_value)
            verb = "was" if len(problems) == 1 else "were"
            invalid_count = f"{len(problems)} of {total} theorem entries {verb} invalid"
            summary = (
                f"Proof identification claimed provable theorems but {invalid_count}"
                if theorem_candidates or non_novel_candidate_count
                else (
                    "Proof identification claimed provable theorems but returned no valid "
                    f"theorem candidates: {invalid_count}"
                )
            )
            raise ModelOutputContractError(
                summary,
                role=role,
                stage="missing_key" if all(p.missing_keys for p in problems) else "shape",
                problems=problems,
                repairs_applied=applied,
            )

        return has_candidates and bool(theorem_candidates), theorem_candidates

    async def _retry_identification_output(
        self,
        *,
        prompt: str,
        task_id: str,
        failed_output: str,
        error: Exception,
        max_input_tokens: int,
        proof_kinds: ProofKindPolicy,
    ) -> Tuple[bool, List[ProofCandidate]]:
        logger.info("ProofIdentificationAgent: initial output failed; attempting bounded JSON retry.")
        retry_prompt = (
            "Your previous proof-identification response could not be used.\n\n"
            f"ERROR: {error}\n\n"
            "Return the same proof-identification decision in valid JSON only. "
            "Correct every theorem entry and field named in the ERROR, using the exact field names below. "
            "Use this exact top-level shape:\n"
            "{\n"
            '  "has_provable_theorems": true or false,\n'
            '  "theorems": [\n'
            "    {\n"
            '      "theorem_id": "short_id",\n'
            '      "statement": "the theorem statement",\n'
            '      "formal_sketch": "Lean-relevant proof sketch",\n'
            '      "expected_novelty_tier": "major_mathematical_discovery | mathematical_discovery | novel_variant | novel_formulation | not_novel",\n'
            '      "prompt_relevance_rationale": "how this solves or plausibly advances the user prompt, directly or indirectly",\n'
            '      "novelty_rationale": "why this is not merely standard or routine",\n'
            '      "why_not_standard_known_result": "why this is not a standard Mathlib/textbook result",\n'
            '      "retry_open_target_id": "Target ID of the OPEN PROOF TARGET this retries, or an empty string"'
            f"{_retry_proposal_kind_line(proof_kinds)}\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Respond with ONLY the JSON object, no markdown and no explanation."
        )
        prompt_with_retry_instruction = f"{prompt}\n\n---\n{retry_prompt}"
        instruction_tokens = count_tokens(prompt_with_retry_instruction)
        if instruction_tokens <= max_input_tokens:
            messages = [{"role": "user", "content": prompt_with_retry_instruction}]
        else:
            logger.warning(
                "ProofIdentificationAgent retry instruction too large (%s > %s); retrying original prompt.",
                instruction_tokens,
                max_input_tokens,
            )
            messages = [{"role": "user", "content": prompt}]
        failed_output_preview = sanitize_model_output_for_retry_context(
            failed_output,
            max_chars=2000,
        )
        if failed_output_preview:
            conversation_tokens = (
                count_tokens(prompt)
                + count_tokens(failed_output_preview)
                + count_tokens(retry_prompt)
            )
            if conversation_tokens <= max_input_tokens:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": failed_output_preview},
                    {"role": "user", "content": retry_prompt},
                ]
            else:
                logger.warning(
                    "ProofIdentificationAgent retry conversation too large (%s > %s); retrying original prompt.",
                    conversation_tokens,
                    max_input_tokens,
                )
        response = await api_client_manager.generate_completion(
            task_id=f"{task_id}_retry",
            role_id=self.role_id,
            model=self.model_id,
            messages=messages,
            max_tokens=self.max_output_tokens,
            temperature=0.0,
        )
        retry_content = self._extract_response_content(response)
        return await self._parse_and_report(retry_content, task_id=f"{task_id}_retry")

    async def _parse_and_report(
        self,
        content: str,
        *,
        task_id: str,
    ) -> Tuple[bool, List[ProofCandidate]]:
        """Parse one response; publish its normalizations only if it was accepted."""
        repairs: list = []
        result = self._parse_candidate_payload(content, role=self.role_id, repairs=repairs)
        await report_model_output_normalizations(self.role_id, repairs, task_id=task_id)
        return result

    async def translate_candidate_to_smt(
        self,
        *,
        user_research_prompt: str,
        source_type: str,
        theorem_candidate: ProofCandidate,
        source_content: str,
        source_title: str = "",
    ) -> str:
        """Return an SMT-LIB translation for a conservative proof candidate when possible."""
        source_excerpt = theorem_candidate.source_excerpt or source_content[:4000]
        prompt = build_smt_translation_prompt(
            user_prompt=user_research_prompt,
            source_type=source_type,
            theorem_statement=theorem_candidate.statement,
            formal_sketch=theorem_candidate.formal_sketch,
            source_excerpt=source_excerpt,
            source_title=source_title,
        )
        prompt_tokens = count_tokens(prompt)
        max_input_tokens = rag_config.get_available_input_tokens(self.context_window, self.max_output_tokens)
        while prompt_tokens > max_input_tokens and len(source_excerpt) > 1200:
            source_excerpt = source_excerpt[: max(len(source_excerpt) // 2, 1200)]
            prompt = build_smt_translation_prompt(
                user_prompt=user_research_prompt,
                source_type=source_type,
                theorem_statement=theorem_candidate.statement,
                formal_sketch=theorem_candidate.formal_sketch,
                source_excerpt=source_excerpt,
                source_title=source_title,
            )
            prompt_tokens = count_tokens(prompt)

        task_id = self.get_current_task_id()
        await api_client_manager.prewarm_assistant_memory_context(
            task_id=task_id,
            role_id=self.role_id,
            prompt=prompt,
        )
        if prompt_tokens > max_input_tokens:
            logger.debug(
                "SMT translation prompt exceeds context window (%s > %s) for theorem %s",
                prompt_tokens,
                max_input_tokens,
                theorem_candidate.theorem_id,
            )
            return ""

        self.task_sequence += 1

        try:
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens,
                temperature=0.0,
            )
            if not response or not response.get("choices"):
                return ""

            message = response["choices"][0].get("message", {})
            content = extract_message_text(message)
            if not content:
                return ""

            data = parse_json(content)
            if isinstance(data, list):
                data = data[0] if data else {}
            if not isinstance(data, dict):
                return ""
            return str(data.get("smtlib", "") or data.get("smtlib2", "")).strip()
        except FreeModelExhaustedError:
            raise
        except Exception as exc:
            if (
                isinstance(exc, RetryableProviderError)
                or is_non_retryable_model_error(exc)
                or is_transient_model_call_error(exc)
            ):
                raise
            logger.debug(
                "ProofIdentificationAgent SMT translation failed for theorem %s: %s",
                theorem_candidate.theorem_id,
                exc,
            )
            return ""

    async def identify_candidates(
        self,
        user_research_prompt: str,
        source_type: str,
        source_id: str,
        source_content: str,
        source_title: str = "",
        proof_round_index: int = 1,
        proof_max_rounds: int = 1,
        prior_round_results: Sequence[str] = (),
        candidate_list_rejection_feedback: str = "",
        candidate_list_rejection_feedback_entries: Sequence[str] = (),
        open_proof_targets: str = "",
    ) -> Tuple[bool, List[ProofCandidate]]:
        """Return whether proof candidates exist and the extracted theorem list.

        ``prior_round_results`` holds round-memory entries, oldest first.

        Only the proof kinds the user allows are offered, and any candidate the
        model still returns as a disallowed kind is routed to an allowed one.
        """
        proof_kinds = ProofKindPolicy.from_config()
        has_candidates, candidates = await self._identify_candidates(
            user_research_prompt=user_research_prompt,
            source_type=source_type,
            source_id=source_id,
            source_content=source_content,
            source_title=source_title,
            proof_round_index=proof_round_index,
            proof_max_rounds=proof_max_rounds,
            prior_round_results=prior_round_results,
            candidate_list_rejection_feedback=candidate_list_rejection_feedback,
            candidate_list_rejection_feedback_entries=candidate_list_rejection_feedback_entries,
            open_proof_targets=open_proof_targets,
            proof_kinds=proof_kinds,
        )
        return has_candidates, proof_kinds.normalize(candidates)

    async def _identify_candidates(
        self,
        *,
        user_research_prompt: str,
        source_type: str,
        source_id: str,
        source_content: str,
        source_title: str,
        proof_round_index: int,
        proof_max_rounds: int,
        prior_round_results: Sequence[str],
        candidate_list_rejection_feedback: str,
        candidate_list_rejection_feedback_entries: Sequence[str],
        open_proof_targets: str,
        proof_kinds: ProofKindPolicy,
    ) -> Tuple[bool, List[ProofCandidate]]:
        max_input_tokens = rag_config.get_available_input_tokens(self.context_window, self.max_output_tokens)
        feedback_entries = tuple(candidate_list_rejection_feedback_entries)
        if not feedback_entries and candidate_list_rejection_feedback:
            feedback_entries = (candidate_list_rejection_feedback,)

        from backend.autonomous.core.proof_round_driver import normalize_round_memory
        from backend.shared.solution_path.integration import with_budgeted_solver_plan

        round_memory = normalize_round_memory(prior_round_results)

        def build_prompt(streams: tuple[tuple[str, ...], ...]) -> str:
            memory_entries, rejection_entries = streams
            prompt = build_proof_identification_prompt(
                user_prompt=user_research_prompt,
                source_type=source_type,
                source_id=source_id,
                source_content=source_content,
                source_title=source_title,
                proof_round_index=proof_round_index,
                proof_max_rounds=proof_max_rounds,
                prior_round_results="\n\n".join(memory_entries),
                candidate_list_rejection_feedback="\n\n".join(rejection_entries),
                open_proof_targets=open_proof_targets,
                allow_single_shot_proofs=proof_kinds.single_shot,
                allow_master_proofs=proof_kinds.master,
            )
            return with_budgeted_solver_plan(
                prompt, self.solution_path_manager, max_input_tokens
            )

        # Older round memory is shed before same-round rejection feedback; each
        # stream keeps its newest entry.
        fit = fit_prompt_with_feedback_streams(
            (round_memory, feedback_entries),
            build_prompt=build_prompt,
            available_tokens=max_input_tokens,
        )
        prompt = fit.prompt
        retained_feedback = fit.retained_streams
        prompt_tokens = count_tokens(prompt)
        task_id = self.get_current_task_id()
        await api_client_manager.prewarm_assistant_memory_context(
            task_id=task_id,
            role_id=self.role_id,
            prompt=prompt,
        )
        if prompt_tokens > max_input_tokens:
            message = (
                "Proof identification prompt exceeds the configured context window "
                f"({prompt_tokens} > {max_input_tokens}) for {source_type} {source_id}. "
                f"Configured total context={self.context_window}, max output reserve={self.max_output_tokens}, "
                f"safety buffer={rag_config.context_buffer_tokens}. "
                "Full source content is mandatory for proof discovery and was not "
                "truncated or replaced with an excerpt. Increase the proof role "
                "context window or reduce the source size before retrying."
            )
            logger.warning(message)
            raise ProofIdentificationContextError(
                message,
                prompt_tokens=prompt_tokens,
                max_input_tokens=max_input_tokens,
            )

        self.task_sequence += 1

        try:
            while True:
                try:
                    response = await api_client_manager.generate_completion(
                        task_id=task_id,
                        role_id=self.role_id,
                        model=self.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.max_output_tokens,
                        temperature=0.0,
                    )
                    break
                except ProviderContextLengthError:
                    shed = shed_oldest_feedback(retained_feedback)
                    if shed is None:
                        raise
                    retained_feedback = shed
                    prompt = build_prompt(retained_feedback)
            content = self._extract_response_content(response)
            try:
                return await self._parse_and_report(content, task_id=task_id)
            except Exception as parse_error:
                return await self._retry_identification_output(
                    prompt=prompt,
                    task_id=task_id,
                    failed_output=content,
                    error=parse_error,
                    max_input_tokens=max_input_tokens,
                    proof_kinds=proof_kinds,
                )
        except FreeModelExhaustedError:
            raise
        except Exception as exc:
            if is_retryable_model_output_error(exc):
                return await self._retry_identification_output(
                    prompt=prompt,
                    task_id=task_id,
                    failed_output="",
                    error=exc,
                    max_input_tokens=max_input_tokens,
                    proof_kinds=proof_kinds,
                )
            if (
                isinstance(exc, RetryableProviderError)
                or is_non_retryable_model_error(exc)
                or is_transient_model_call_error(exc)
            ):
                raise
            logger.error(
                "ProofIdentificationAgent failed for %s %s: %s",
                source_type,
                source_id,
                exc,
            )
            raise
