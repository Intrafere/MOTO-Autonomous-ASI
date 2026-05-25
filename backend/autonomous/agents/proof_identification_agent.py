"""
Proof identification agent for Lean 4 verification checkpoints.
"""
import logging
from typing import List, Tuple

from backend.shared.api_client_manager import api_client_manager
from backend.shared.json_parser import parse_json
from backend.shared.model_error_utils import is_non_retryable_model_error
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
    ) -> None:
        self.model_id = model_id
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.role_id = role_id
        self.task_sequence = 0

    def get_current_task_id(self) -> str:
        return f"proof_id_{self.task_sequence:03d}"

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

        if prompt_tokens > max_input_tokens:
            logger.debug(
                "SMT translation prompt exceeds context window (%s > %s) for theorem %s",
                prompt_tokens,
                max_input_tokens,
                theorem_candidate.theorem_id,
            )
            return ""

        task_id = self.get_current_task_id()
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
            content = message.get("content") or message.get("reasoning") or ""
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
            if is_non_retryable_model_error(exc):
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
    ) -> Tuple[bool, List[ProofCandidate]]:
        """Return whether proof candidates exist and the extracted theorem list."""
        prompt = build_proof_identification_prompt(
            user_prompt=user_research_prompt,
            source_type=source_type,
            source_id=source_id,
            source_content=source_content,
            source_title=source_title,
        )
        prompt_tokens = count_tokens(prompt)
        max_input_tokens = rag_config.get_available_input_tokens(self.context_window, self.max_output_tokens)
        if prompt_tokens > max_input_tokens:
            message = (
                "Proof identification prompt exceeds the configured context window "
                f"({prompt_tokens} > {max_input_tokens}) for {source_type} {source_id}. "
                "Full source content is mandatory for proof discovery and was not "
                "truncated or replaced with an excerpt. Increase the proof role "
                "context window or reduce the source size before retrying."
            )
            logger.warning(message)
            raise ValueError(message)

        task_id = self.get_current_task_id()
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
                return False, []

            message = response["choices"][0].get("message", {})
            content = message.get("content") or message.get("reasoning") or ""
            if not content:
                return False, []

            data = parse_json(content)
            if isinstance(data, list):
                data = data[0] if data else {}

            has_candidates = bool(data.get("has_provable_theorems", False))
            raw_theorems = data.get("theorems", []) or []
            theorem_candidates: List[ProofCandidate] = []
            for index, theorem in enumerate(raw_theorems, start=1):
                if not isinstance(theorem, dict):
                    continue
                statement = str(theorem.get("statement", "")).strip()
                if not statement:
                    continue
                theorem_id = theorem.get("theorem_id") or theorem.get("id") or f"thm_{index}"
                expected_novelty_tier = str(theorem.get("expected_novelty_tier", "")).strip().lower()
                if expected_novelty_tier == "not_novel":
                    logger.info(
                        "ProofIdentificationAgent skipped theorem %s because it was marked not_novel.",
                        theorem_id,
                    )
                    continue
                if expected_novelty_tier not in _NOVEL_PROOF_TIERS:
                    logger.info(
                        "ProofIdentificationAgent skipped theorem %s because it did not include a valid expected_novelty_tier.",
                        theorem_id,
                    )
                    continue
                theorem_candidates.append(
                    ProofCandidate(
                        theorem_id=str(theorem_id),
                        statement=statement,
                        formal_sketch=str(theorem.get("formal_sketch", "")).strip(),
                        expected_novelty_tier=expected_novelty_tier,
                        prompt_relevance_rationale=str(
                            theorem.get("prompt_relevance_rationale", "")
                        ).strip(),
                        novelty_rationale=str(theorem.get("novelty_rationale", "")).strip(),
                        why_not_standard_known_result=str(
                            theorem.get("why_not_standard_known_result", "")
                        ).strip(),
                    )
                )

            return has_candidates and bool(theorem_candidates), theorem_candidates
        except FreeModelExhaustedError:
            raise
        except Exception as exc:
            if is_non_retryable_model_error(exc):
                raise
            logger.error(
                "ProofIdentificationAgent failed for %s %s: %s",
                source_type,
                source_id,
                exc,
            )
            return False, []
