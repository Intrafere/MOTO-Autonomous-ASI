"""Independent pre-Lean Validator for complete proof-candidate lists."""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from backend.autonomous.prompts.proof_prompts import (
    build_proof_candidate_list_validation_prompt,
)
from backend.shared.api_client_manager import api_client_manager
from backend.shared.config import rag_config
from backend.shared.json_parser import (
    parse_json,
    sanitize_model_output_for_retry_context,
)
from backend.shared.models import (
    ProofCandidate,
    ProofCandidateListValidation,
)
from backend.shared.response_extraction import extract_message_text
from backend.shared.utils import count_tokens


class ProofCandidateListContractError(ValueError):
    """The original and bounded repair outputs violated the list contract."""


class ProofCandidateListContextError(ValueError):
    """Mandatory list-review context exceeded the Validator's configured budget."""


class ProofCandidateListValidator:
    """Review every proposed theorem and return a strict identity-complete decision."""

    def __init__(
        self,
        *,
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

    def _task_id(self) -> str:
        self.task_sequence += 1
        return f"proof_list_val_{self.task_sequence:03d}"

    @staticmethod
    def _extract_content(response: dict[str, Any]) -> str:
        if not response or not response.get("choices"):
            raise ValueError("Proof candidate-list Validator returned no model choices.")
        content = extract_message_text(response["choices"][0].get("message", {}))
        if not content:
            raise ValueError("Proof candidate-list Validator returned empty model output.")
        return content

    @staticmethod
    def parse_response(
        content: str,
        *,
        expected_candidate_ids: list[str],
    ) -> ProofCandidateListValidation:
        data = parse_json(content)
        if not isinstance(data, dict):
            raise ValueError("Candidate-list Validator output must be one JSON object.")
        validation = ProofCandidateListValidation.model_validate(data)
        actual_ids = [result.theorem_id for result in validation.results]
        if actual_ids != expected_candidate_ids:
            raise ValueError(
                "Candidate-list Validator results must contain exactly one result for "
                "every proposed theorem_id in the original order."
            )
        if len(set(actual_ids)) != len(actual_ids):
            raise ValueError("Candidate-list Validator returned duplicate theorem IDs.")
        return validation

    async def validate(
        self,
        *,
        user_prompt: str,
        source_type: str,
        source_id: str,
        source_title: str,
        candidates: list[ProofCandidate],
    ) -> ProofCandidateListValidation:
        expected_ids = [candidate.theorem_id for candidate in candidates]
        if not expected_ids or len(set(expected_ids)) != len(expected_ids):
            raise ValueError("Proposed proof candidates require unique non-empty theorem IDs.")
        candidates_json = json.dumps(
            [candidate.model_dump(mode="json") for candidate in candidates],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        prompt = build_proof_candidate_list_validation_prompt(
            user_prompt=user_prompt,
            source_type=source_type,
            source_id=source_id,
            source_title=source_title,
            candidates_json=candidates_json,
        )
        max_input_tokens = rag_config.get_available_input_tokens(
            self.context_window,
            self.max_output_tokens,
        )
        if count_tokens(prompt) > max_input_tokens:
            raise ProofCandidateListContextError(
                "Mandatory proof candidate-list review context exceeds the configured "
                "Validator input budget."
            )
        task_id = self._task_id()
        response = await api_client_manager.generate_completion(
            task_id=task_id,
            role_id=self.role_id,
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_output_tokens,
            temperature=0.0,
        )
        content = self._extract_content(response)
        try:
            return self.parse_response(content, expected_candidate_ids=expected_ids)
        except (ValueError, TypeError, ValidationError) as first_error:
            visible_output = sanitize_model_output_for_retry_context(
                content,
                max_chars=2000,
            )
            repair = (
                "Your prior response violated the required JSON contract. "
                f"Error: {first_error}. Return only one corrected JSON object with "
                "results in the exact supplied theorem_id order, one per candidate, "
                'using decision "approve_novel" or "reject_not_novel", plus feedback.'
            )
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": visible_output},
                {"role": "user", "content": repair},
            ]
            if sum(count_tokens(item["content"]) for item in messages) > max_input_tokens:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": repair},
                ]
            if sum(count_tokens(item["content"]) for item in messages) > max_input_tokens:
                raise ProofCandidateListContextError(
                    "The bounded candidate-list repair prompt does not fit the "
                    "configured Validator input budget."
                ) from first_error
            retry_response = await api_client_manager.generate_completion(
                task_id=f"{task_id}_repair",
                role_id=self.role_id,
                model=self.model_id,
                messages=messages,
                max_tokens=self.max_output_tokens,
                temperature=0.0,
            )
            retry_content = self._extract_content(retry_response)
            try:
                return self.parse_response(
                    retry_content,
                    expected_candidate_ids=expected_ids,
                )
            except (ValueError, TypeError, ValidationError) as repair_error:
                raise ProofCandidateListContractError(
                    f"Candidate-list Validator repair failed: {repair_error}"
                ) from repair_error

    @staticmethod
    def approved_candidates(
        candidates: list[ProofCandidate],
        validation: ProofCandidateListValidation,
    ) -> list[ProofCandidate]:
        approved_ids = {
            result.theorem_id
            for result in validation.results
            if result.decision == "approve_novel"
        }
        return [candidate for candidate in candidates if candidate.theorem_id in approved_ids]

    @staticmethod
    def threshold_met(*, approved_count: int, proposed_count: int) -> bool:
        return proposed_count > 0 and approved_count * 4 >= proposed_count * 3
