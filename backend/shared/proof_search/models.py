"""Normalized models for unified proof search."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.shared.config import system_config


ProofSearchCorpus = Literal["moto", "manual", "leanoj", "syntheticlib4"]
ProofSearchSourceKind = Literal["verified_proof", "partial_proof", "failed_attempt"]


def default_proof_search_corpora() -> list[ProofSearchCorpus]:
    """Return the enabled default proof-search corpora for AI retrieval."""
    corpora: list[ProofSearchCorpus] = []
    if system_config.agent_conversation_memory_enabled:
        corpora.extend(["moto", "manual", "leanoj"])
    if system_config.syntheticlib4_enabled:
        corpora.append("syntheticlib4")
    return corpora


class UnifiedProofSearchRecord(BaseModel):
    """One proof-like record normalized for local search and AI retrieval."""

    search_id: str
    corpus: ProofSearchCorpus
    corpus_scope: str = ""
    source_kind: ProofSearchSourceKind = "verified_proof"
    proof_id: str
    external_fingerprint: str = ""
    release_id: str = ""
    session_id: str = ""
    source_type: str = ""
    source_id: str = ""
    source_title: str = ""
    display_title: str = ""
    theorem_name: str = ""
    theorem_statement: str
    informal_statement: str = ""
    proof_description: str = ""
    formal_sketch: str = ""
    lean_code: str = ""
    lean_code_hash: str = ""
    theorem_statement_hash: str = ""
    imports: list[str] = Field(default_factory=list)
    dependency_names: list[str] = Field(default_factory=list)
    topic_tags: list[str] = Field(default_factory=list)
    domain_tags: list[str] = Field(default_factory=list)
    module: str = ""
    source_path: str = ""
    novelty_tier: str = ""
    novelty_reasoning: str = ""
    verified: bool = True
    created_at: str = ""
    canonical_uri: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def dedupe_keys(self) -> list[str]:
        """Return strong identity keys for exact duplicate suppression."""
        keys: list[str] = []
        if self.external_fingerprint:
            keys.append(f"fingerprint:{self.external_fingerprint}")
        if self.corpus != "syntheticlib4" and self.proof_id:
            keys.append(f"local:{self.corpus}:{self.proof_id}")
        if self.theorem_statement_hash and self.lean_code_hash:
            keys.append(f"hash:{self.theorem_statement_hash}:{self.lean_code_hash}")
        return keys


class ProofSearchRequest(BaseModel):
    """Backend search request shared by routes and the future AI-facing tool."""

    query: str = ""
    goal_statement: str = ""
    imports: list[str] = Field(default_factory=list)
    dependency_names: list[str] = Field(default_factory=list)
    corpora: list[ProofSearchCorpus] = Field(default_factory=default_proof_search_corpora)
    verified_only: bool = True
    include_partial: bool = False
    include_failed: bool = False
    novelty_filters: list[str] = Field(default_factory=list)
    module_filters: list[str] = Field(default_factory=list)
    source_filters: list[str] = Field(default_factory=list)
    limit: int = Field(default=7, ge=1)
    cursor: str | None = None
    exclude_ids: list[str] = Field(default_factory=list)
    hydrate_lean_code: bool = True
    search_mode: Literal["text", "exact", "hybrid"] = "hybrid"


class PublicProofSearchRequest(ProofSearchRequest):
    """Public REST proof-search request capped by the web contract."""

    limit: int = Field(default=7, ge=1, le=7)


class ProofSearchResponse(BaseModel):
    results: list[UnifiedProofSearchRecord]
    result_count: int
    next_cursor: str | None = None
    searched_corpora: list[str] = Field(default_factory=list)
    corpus_counts: dict[str, int] = Field(default_factory=dict)
    ranking_notes: str = ""
    weak_result_warning: str | None = None


class CorpusOverview(BaseModel):
    total_records: int
    verified_records: int
    partial_records: int
    failed_attempt_records: int
    corpora: list[dict[str, Any]]
    novelty_distribution: dict[str, int] = Field(default_factory=dict)
    top_modules: list[dict[str, Any]] = Field(default_factory=list)
    top_imports: list[dict[str, Any]] = Field(default_factory=list)
    top_dependencies: list[dict[str, Any]] = Field(default_factory=list)
    top_tags: list[dict[str, Any]] = Field(default_factory=list)
    search_fields: list[str] = Field(default_factory=list)
    recommended_queries: list[str] = Field(default_factory=list)
    result_cap: int = 7
    hydration_behavior: str = (
        "Search returns at most 7 combined records. Full Lean code is included "
        "only when available and requested; metadata-only records can be hydrated later."
    )

