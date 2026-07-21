"""
Proof identification agent for Lean 4 verification checkpoints.
"""
import logging
from typing import Any, Dict, List, Tuple

from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.json_parser import parse_json, sanitize_model_output_for_retry_context
from backend.shared.response_extraction import extract_message_text
from backend.shared.model_error_utils import (
    is_retryable_model_output_error,
    is_non_retryable_model_error,
    is_transient_model_call_error,
)
from backend.shared.models import ProofCandidate
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.utils import count_tokens
from backend.shared.config import rag_config
from backend.autonomous.prompts.proof_prompts import (
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
    def _parse_candidate_payload(content: str) -> Tuple[bool, List[ProofCandidate]]:
        data = parse_json(content)
        if isinstance(data, list):
            if not data:
                raise ValueError("Proof identification returned an empty JSON array.")
            data = data[0]
        if not isinstance(data, dict):
            raise ValueError("Proof identification returned JSON that was not an object.")
        if "has_provable_theorems" not in data:
            raise ValueError("Proof identification JSON omitted has_provable_theorems.")
        if not isinstance(data.get("has_provable_theorems"), bool):
            raise ValueError("Proof identification has_provable_theorems must be a boolean.")

        has_candidates = data["has_provable_theorems"]
        raw_theorems_value = data.get("theorems", [])
        if raw_theorems_value is None:
            raw_theorems_value = []
        if not isinstance(raw_theorems_value, list):
            raise ValueError("Proof identification theorems must be an array.")
        if has_candidates and not raw_theorems_value:
            raise ValueError(
                "Proof identification claimed provable theorems but returned no theorem entries."
            )

        theorem_candidates: List[ProofCandidate] = []
        malformed_candidate_count = 0
        non_novel_candidate_count = 0
        for index, theorem in enumerate(raw_theorems_value, start=1):
            if not isinstance(theorem, dict):
                malformed_candidate_count += 1
                continue
            statement = str(theorem.get("statement", "")).strip()
            if not statement:
                malformed_candidate_count += 1
                continue
            theorem_id = theorem.get("theorem_id") or theorem.get("id") or f"thm_{index}"
            expected_novelty_tier = str(theorem.get("expected_novelty_tier", "")).strip().lower()
            if expected_novelty_tier == "not_novel":
                non_novel_candidate_count += 1
                logger.info(
                    "ProofIdentificationAgent skipped theorem %s because it was marked not_novel.",
                    theorem_id,
                )
                continue
            if expected_novelty_tier not in _NOVEL_PROOF_TIERS:
                malformed_candidate_count += 1
                logger.info(
                    "ProofIdentificationAgent skipped theorem %s because it did not include a valid expected_novelty_tier.",
                    theorem_id,
                )
                continue
            prompt_relevance_rationale = str(
                theorem.get("prompt_relevance_rationale", "")
            ).strip()
            novelty_rationale = str(theorem.get("novelty_rationale", "")).strip()
            why_not_standard_known_result = str(
                theorem.get("why_not_standard_known_result", "")
            ).strip()
            if not (
                prompt_relevance_rationale
                and novelty_rationale
                and why_not_standard_known_result
            ):
                malformed_candidate_count += 1
                logger.info(
                    "ProofIdentificationAgent skipped theorem %s because it lacked required prompt-relevance, novelty, or anti-standard-result rationale.",
                    theorem_id,
                )
                continue
            theorem_candidates.append(
                ProofCandidate(
                    theorem_id=str(theorem_id),
                    statement=statement,
                    formal_sketch=str(theorem.get("formal_sketch", "")).strip(),
                    expected_novelty_tier=expected_novelty_tier,
                    prompt_relevance_rationale=prompt_relevance_rationale,
                    novelty_rationale=novelty_rationale,
                    why_not_standard_known_result=why_not_standard_known_result,
                )
            )

        if has_candidates and not theorem_candidates and malformed_candidate_count:
            raise ValueError(
                "Proof identification claimed provable theorems but returned no valid theorem candidates "
                f"({malformed_candidate_count} malformed, {non_novel_candidate_count} not_novel)."
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
    ) -> Tuple[bool, List[ProofCandidate]]:
        logger.info("ProofIdentificationAgent: initial output failed; attempting bounded JSON retry.")
        retry_prompt = (
            "Your previous proof-identification response could not be used.\n\n"
            f"ERROR: {error}\n\n"
            "Return the same proof-identification decision in valid JSON only. "
            "Use this exact top-level shape:\n"
            "{\n"
            '  "has_provable_theorems": true or false,\n'
            '  "theorems": [\n'
            "    {\n"
            '      "theorem_id": "short_id",\n'
            '      "statement": "the theorem statement",\n'
            '      "formal_sketch": "Lean-relevant proof sketch",\n'
            '      "expected_novelty_tier": "major_mathematical_discovery | mathematical_discovery | novel_variant | novel_formulation | not_novel",\n'
            '      "prompt_relevance_rationale": "why this directly answers or substantially advances the user prompt",\n'
            '      "novelty_rationale": "why this is not merely standard or routine",\n'
            '      "why_not_standard_known_result": "why this is not a standard Mathlib/textbook result"\n'
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
        return self._parse_candidate_payload(retry_content)

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
        prior_round_results: str = "",
    ) -> Tuple[bool, List[ProofCandidate]]:
        """Return whether proof candidates exist and the extracted theorem list."""
        prompt = build_proof_identification_prompt(
            user_prompt=user_research_prompt,
            source_type=source_type,
            source_id=source_id,
            source_content=source_content,
            source_title=source_title,
            proof_round_index=proof_round_index,
            proof_max_rounds=proof_max_rounds,
            prior_round_results=prior_round_results,
        )
        max_input_tokens = rag_config.get_available_input_tokens(self.context_window, self.max_output_tokens)
        from backend.shared.solution_path.integration import with_budgeted_solver_plan
        prompt = with_budgeted_solver_plan(
            prompt, self.solution_path_manager, max_input_tokens
        )
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
            raise ValueError(message)

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
            content = self._extract_response_content(response)
            try:
                return self._parse_candidate_payload(content)
            except Exception as parse_error:
                return await self._retry_identification_output(
                    prompt=prompt,
                    task_id=task_id,
                    failed_output=content,
                    error=parse_error,
                    max_input_tokens=max_input_tokens,
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
