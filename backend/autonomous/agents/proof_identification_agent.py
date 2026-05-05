"""
Proof identification agent for Lean 4 verification checkpoints.
"""
import logging
from typing import List, Tuple

from backend.shared.api_client_manager import api_client_manager
from backend.shared.json_parser import parse_json
from backend.shared.models import ProofCandidate
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.utils import count_tokens
from backend.autonomous.prompts.proof_prompts import (
    build_proof_identification_prompt,
    build_smt_translation_prompt,
)

logger = logging.getLogger(__name__)


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
    ) -> str:
        """Return an SMT-LIB translation for a conservative proof candidate when possible."""
        source_excerpt = theorem_candidate.source_excerpt or source_content[:4000]
        prompt = build_smt_translation_prompt(
            user_prompt=user_research_prompt,
            source_type=source_type,
            theorem_statement=theorem_candidate.statement,
            formal_sketch=theorem_candidate.formal_sketch,
            source_excerpt=source_excerpt,
        )
        prompt_tokens = count_tokens(prompt)
        max_input_tokens = self.context_window - self.max_output_tokens
        while prompt_tokens > max_input_tokens and len(source_excerpt) > 1200:
            source_excerpt = source_excerpt[: max(len(source_excerpt) // 2, 1200)]
            prompt = build_smt_translation_prompt(
                user_prompt=user_research_prompt,
                source_type=source_type,
                theorem_statement=theorem_candidate.statement,
                formal_sketch=theorem_candidate.formal_sketch,
                source_excerpt=source_excerpt,
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
    ) -> Tuple[bool, List[ProofCandidate]]:
        """Return whether proof candidates exist and the extracted theorem list."""
        prompt = build_proof_identification_prompt(
            user_prompt=user_research_prompt,
            source_type=source_type,
            source_id=source_id,
            source_content=source_content,
        )
        prompt_tokens = count_tokens(prompt)
        max_input_tokens = self.context_window - self.max_output_tokens
        if prompt_tokens > max_input_tokens:
            logger.warning(
                "ProofIdentificationAgent prompt exceeds context window (%s > %s) for %s %s",
                prompt_tokens,
                max_input_tokens,
                source_type,
                source_id,
            )
            return False, []

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
                theorem_candidates.append(
                    ProofCandidate(
                        theorem_id=str(theorem_id),
                        statement=statement,
                        formal_sketch=str(theorem.get("formal_sketch", "")).strip(),
                    )
                )

            return has_candidates and bool(theorem_candidates), theorem_candidates
        except FreeModelExhaustedError:
            raise
        except Exception as exc:
            logger.error(
                "ProofIdentificationAgent failed for %s %s: %s",
                source_type,
                source_id,
                exc,
            )
            return False, []
