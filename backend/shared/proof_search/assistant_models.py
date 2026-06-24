"""Models for non-blocking Assistant proof-support retrieval."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from backend.shared.proof_search.models import ProofSearchCorpus, UnifiedProofSearchRecord


AssistantWorkflowMode = Literal[
    "aggregator",
    "compiler",
    "autonomous",
    "leanoj",
    "manual_proof_check",
]
AssistantTargetKind = Literal[
    "brainstorm_context",
    "writing_context",
    "outline_context",
    "reference_selection_context",
    "topic_context",
    "title_context",
    "completion_review_context",
    "path_context",
    "semantic_review_context",
    "final_answer_context",
    "proof_candidate",
    "lean_error",
    "theorem_discovery",
    "master_proof",
    "final_solver",
    "paper_claim",
]
AssistantFreshness = Literal["fresh", "cached", "stale-but-best-known"]
AssistantSelectionMode = Literal[
    "assistant_llm",
    "cached",
    "stale-but-best-known",
    "no_candidates",
    "unavailable",
    "cached_oauth_cooldown",
    "deterministic_oauth_cooldown",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AssistantTargetSnapshot(BaseModel):
    """A fast-moving memory-support target observed from the parent workflow."""

    workflow_mode: AssistantWorkflowMode
    target_kind: AssistantTargetKind
    workflow_phase: str = ""
    active_mode: str = ""
    user_prompt: str = ""
    current_prompt_or_topic: str = ""
    current_submission_or_draft: str = ""
    accepted_memory_summary: str = ""
    writing_goal: str = ""
    outline_summary: str = ""
    paper_or_proof_draft_summary: str = ""
    recent_activity_summary: str = ""
    source_titles: list[str] = Field(default_factory=list)
    target_statement: str = ""
    lean_template: str = ""
    formal_sketch: str = ""
    lean_error: str = ""
    rejection_feedback: str = ""
    proof_attempt_feedback: str = ""
    accepted_solver_summary: str = ""
    source_title: str = ""
    source_type: str = ""
    source_id: str = ""
    dependency_names: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=lambda: ["Mathlib"])
    target_hash: str = ""
    created_at: str = Field(default_factory=_now_iso)

    def stable_hash(self) -> str:
        if self.target_hash:
            return self.target_hash
        parts = [
            self.workflow_mode,
            self.target_kind,
            self.workflow_phase,
            self.active_mode,
            self.user_prompt,
            self.current_prompt_or_topic,
            self.current_submission_or_draft,
            self.accepted_memory_summary,
            self.writing_goal,
            self.outline_summary,
            self.paper_or_proof_draft_summary,
            self.recent_activity_summary,
            self.target_statement,
            self.lean_template,
            self.formal_sketch,
            self.lean_error,
            self.rejection_feedback,
            self.proof_attempt_feedback,
            self.source_title,
            self.source_type,
            self.source_id,
            " ".join(self.source_titles),
            " ".join(self.dependency_names),
            " ".join(self.imports),
        ]
        return hashlib.sha256("\n\n".join(parts).encode("utf-8")).hexdigest()

    def search_text(self) -> str:
        return "\n\n".join(
            part
            for part in [
                self.user_prompt,
                self.current_prompt_or_topic,
                self.current_submission_or_draft,
                self.accepted_memory_summary,
                self.writing_goal,
                self.outline_summary,
                self.paper_or_proof_draft_summary,
                self.recent_activity_summary,
                self.target_statement,
                self.lean_template,
                self.formal_sketch,
                self.lean_error,
                self.rejection_feedback,
                self.proof_attempt_feedback,
                self.accepted_solver_summary,
                self.source_title,
                " ".join(self.source_titles),
            ]
            if part and part.strip()
        )


class AssistantProofSupport(BaseModel):
    """One proof support selected for the Assistant pack."""

    search_id: str
    corpus: ProofSearchCorpus
    corpus_scope: str = ""
    source_kind: str = "verified_proof"
    proof_id: str
    session_id: str = ""
    fingerprint: str = ""
    theorem_name: str = ""
    theorem_statement: str
    proof_description: str = ""
    imports: list[str] = Field(default_factory=list)
    dependency_names: list[str] = Field(default_factory=list)
    theorem_statement_hash: str = ""
    lean_code_hash: str = ""
    canonical_uri: str = ""
    relevance_reason: str = ""
    transfer_hint: str = ""
    has_hydrated_code: bool = False
    lean_code: str = ""

    @classmethod
    def from_record(
        cls,
        record: UnifiedProofSearchRecord,
        *,
        relevance_reason: str = "",
        transfer_hint: str = "",
    ) -> "AssistantProofSupport":
        return cls(
            search_id=record.search_id,
            corpus=record.corpus,
            corpus_scope=record.corpus_scope or record.release_id,
            source_kind=record.source_kind,
            proof_id=record.proof_id,
            session_id=record.session_id,
            fingerprint=record.external_fingerprint,
            theorem_name=record.theorem_name or record.display_title,
            theorem_statement=record.theorem_statement,
            proof_description=record.proof_description or record.formal_sketch,
            imports=list(record.imports or []),
            dependency_names=list(record.dependency_names or []),
            theorem_statement_hash=record.theorem_statement_hash,
            lean_code_hash=record.lean_code_hash,
            canonical_uri=record.canonical_uri,
            relevance_reason=relevance_reason,
            transfer_hint=transfer_hint,
            has_hydrated_code=bool((record.lean_code or "").strip()),
            lean_code=record.lean_code or "",
        )

    def metadata_only_dump(self) -> dict:
        payload = self.model_dump(mode="json")
        payload["lean_code"] = ""
        payload["has_hydrated_code"] = bool(self.has_hydrated_code)
        return payload


class AssistantProofPack(BaseModel):
    """Latest non-blocking proof-support pack for one target snapshot."""

    schema_version: str = "moto.assistant_proof_pack.v1"
    created_at: str = Field(default_factory=_now_iso)
    workflow_mode: AssistantWorkflowMode
    target_kind: AssistantTargetKind
    target_hash: str
    query_summary: str = ""
    freshness: AssistantFreshness = "fresh"
    results: list[AssistantProofSupport] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    selection_mode: AssistantSelectionMode = "assistant_llm"
    assistant_role_id: str = ""
    assistant_model_id: str = ""
    candidate_count: int = 0
    shortlist_count: int = 0
    selection_reasoning: str = ""

    def to_prompt_context(self, *, max_code_chars_per_result: int = 4000) -> str:
        return self._to_prompt_context(
            heading="ASSISTANT RETRIEVED PROOF SUPPORT",
            max_code_chars_per_result=max_code_chars_per_result,
        )

    def to_memory_prompt_context(self, *, max_code_chars_per_result: int = 2500) -> str:
        return self._to_prompt_context(
            heading="ASSISTANT RETRIEVED MEMORY SUPPORT",
            max_code_chars_per_result=max_code_chars_per_result,
        )

    def _to_prompt_context(
        self,
        *,
        heading: str,
        max_code_chars_per_result: int,
    ) -> str:
        if not self.results:
            warning = " ".join(self.warnings).strip()
            support_label = "memory" if "MEMORY" in heading else "proof"
            return f"[Assistant {support_label} support unavailable. {warning}]".strip()

        lines = [
            heading,
            f"Target hash: {self.target_hash}",
            f"Freshness: {self.freshness}",
            f"Selection mode: {self.selection_mode}",
            f"Query summary: {self.query_summary or '[not provided]'}",
            (
                "Use these verified memory records only as relevant mathematical context, "
                "proof-pattern, dependency, or tactic guidance for the user's prompt/current target."
            ),
        ]
        for index, support in enumerate(self.results[:7], start=1):
            lean_code = support.lean_code or ""
            if len(lean_code) > max_code_chars_per_result:
                lean_code = (
                    lean_code[:max_code_chars_per_result]
                    + "\n-- [assistant proof code truncated for prompt budget]"
                )
            lines.extend(
                [
                    "",
                    f"{index}. {support.theorem_name or '[unnamed theorem]'}",
                    f"Source: {support.corpus} {support.corpus_scope}".strip(),
                    f"Source kind: {support.source_kind}",
                    f"Proof ID: {support.proof_id}",
                    f"Fingerprint: {support.fingerprint or '[none]'}",
                    f"Why relevant: {support.relevance_reason or '[ranked as relevant by Assistant retrieval]'}",
                    f"Transfer hint: {support.transfer_hint or '[inspect statement/dependencies for reusable proof shape]'}",
                    f"Statement: {support.theorem_statement}",
                    f"Description: {support.proof_description or '[none]'}",
                    f"Imports: {', '.join(support.imports) or '[none]'}",
                    f"Dependencies: {', '.join(support.dependency_names) or '[none]'}",
                    f"Theorem statement hash: {support.theorem_statement_hash or '[none]'}",
                    f"Lean code hash: {support.lean_code_hash or '[none]'}",
                    f"Canonical URI: {support.canonical_uri or '[none]'}",
                    "Lean code:",
                    lean_code or "[metadata-only support; use theorem/dependency shape only]",
                ]
            )
        if self.warnings:
            lines.extend(["", "Assistant warnings:", *[f"- {warning}" for warning in self.warnings]])
        return "\n".join(lines)

    def metadata_only_dump(self) -> dict:
        payload = self.model_dump(mode="json")
        payload["results"] = [result.metadata_only_dump() for result in self.results]
        return payload

