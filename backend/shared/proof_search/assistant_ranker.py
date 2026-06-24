"""Lightweight Assistant proof-support ranking and diversification."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from backend.shared.proof_search.assistant_cache import AssistantCandidateStats
from backend.shared.proof_search.assistant_models import (
    AssistantProofSupport,
    AssistantTargetSnapshot,
)
from backend.shared.proof_search.models import UnifiedProofSearchRecord

_TOKEN_RE = re.compile(r"[A-Za-z0-9_'.]+")
_CORPUS_TRUST = {
    "moto": 1.0,
    "manual": 0.95,
    "leanoj": 0.95,
    "syntheticlib4": 0.9,
}


@dataclass(frozen=True)
class RankedProofCandidate:
    record: UnifiedProofSearchRecord
    score: float
    lexical_score: float
    dependency_score: float
    import_score: float
    exact_score: float
    trust_score: float
    recency_score: float
    relevance_reason: str
    transfer_hint: str


def rank_assistant_proof_candidates(
    records: list[UnifiedProofSearchRecord],
    target: AssistantTargetSnapshot,
    *,
    limit: int = 7,
    diversity_lambda: float = 0.25,
    candidate_stats: dict[str, AssistantCandidateStats] | None = None,
) -> list[AssistantProofSupport]:
    """Rank candidates with cheap scoring, persisted P-UCB state, and MMR diversity."""
    ranked = score_assistant_proof_candidates(records, target)
    return select_assistant_proof_supports(
        ranked,
        limit=limit,
        diversity_lambda=diversity_lambda,
        candidate_stats=candidate_stats,
    )


def score_assistant_proof_candidates(
    records: list[UnifiedProofSearchRecord],
    target: AssistantTargetSnapshot,
) -> list[RankedProofCandidate]:
    """Score and sort verified proof candidates before persistence/selection."""
    return _rank_candidates(records, target)


def select_assistant_proof_supports(
    ranked_candidates: list[RankedProofCandidate],
    *,
    limit: int = 7,
    diversity_lambda: float = 0.25,
    candidate_stats: dict[str, AssistantCandidateStats] | None = None,
    exploration_c: float = 0.2,
) -> list[AssistantProofSupport]:
    """Select final supports using P-UCB and MMR-style diversity."""
    ranked = list(ranked_candidates)
    stats = candidate_stats or {}
    total_visits = sum(max(0, item.visits) for item in stats.values()) + 1
    selected: list[RankedProofCandidate] = []
    used_keys: set[str] = set()

    while ranked and len(selected) < max(1, limit):
        best_index = 0
        best_score = -float("inf")
        for index, candidate in enumerate(ranked):
            keys = _dedupe_keys(candidate.record)
            if keys and used_keys.intersection(keys):
                continue
            candidate_stats_entry = stats.get(candidate.record.search_id, AssistantCandidateStats())
            pucb_score = _pucb_score(
                base_score=candidate.score,
                visits=candidate_stats_entry.visits,
                total_visits=total_visits,
                exploration_c=exploration_c,
                failure_penalty=candidate_stats_entry.failure_penalty,
            )
            similarity = max(
                (_record_similarity(candidate.record, chosen.record) for chosen in selected),
                default=0.0,
            )
            final_score = pucb_score - diversity_lambda * similarity
            if final_score > best_score:
                best_index = index
                best_score = final_score

        candidate = ranked.pop(best_index)
        keys = _dedupe_keys(candidate.record)
        if keys and used_keys.intersection(keys):
            continue
        used_keys.update(keys)
        selected.append(candidate)

    return [
        AssistantProofSupport.from_record(
            candidate.record,
            relevance_reason=candidate.relevance_reason,
            transfer_hint=candidate.transfer_hint,
        )
        for candidate in selected
    ]


def ranked_candidates_to_cache_rows(
    ranked_candidates: list[RankedProofCandidate],
) -> list[dict[str, object]]:
    """Convert ranked candidates into SQLite payload rows without full Lean code."""
    rows: list[dict[str, object]] = []
    for candidate in ranked_candidates:
        record = candidate.record
        rows.append(
            {
                "search_id": record.search_id,
                "proof_source": record.corpus,
                "proof_id": record.proof_id,
                "theorem_statement_hash": record.theorem_statement_hash,
                "lean_code_hash": record.lean_code_hash,
                "query_variant": "",
                "retrieval_score": candidate.score,
                "exact_match_score": candidate.exact_score,
                "semantic_score": candidate.lexical_score,
                "dependency_overlap_score": max(
                    candidate.dependency_score,
                    candidate.import_score,
                ),
                "corpus_trust_score": candidate.trust_score,
                "recency_score": candidate.recency_score,
                "duplicate_group": "|".join(sorted(_dedupe_keys(record))),
            }
        )
    return rows


def _rank_candidates(
    records: list[UnifiedProofSearchRecord],
    target: AssistantTargetSnapshot,
) -> list[RankedProofCandidate]:
    target_tokens = _tokens(target.search_text())
    dependency_targets = {value.lower() for value in target.dependency_names if value}
    import_targets = {value.lower() for value in target.imports if value}
    ranked: list[RankedProofCandidate] = []

    for record in records:
        if not record.verified or record.source_kind != "verified_proof":
            continue
        record_text = "\n".join(
            [
                record.theorem_name,
                record.theorem_statement,
                record.informal_statement,
                record.proof_description,
                record.formal_sketch,
                record.source_title,
                " ".join(record.dependency_names),
                " ".join(record.imports),
                " ".join(record.topic_tags),
                " ".join(record.domain_tags),
            ]
        )
        record_tokens = _tokens(record_text)
        lexical_score = _jaccard(target_tokens, record_tokens)
        dependency_score = _overlap_score(dependency_targets, {value.lower() for value in record.dependency_names})
        import_score = _overlap_score(import_targets, {value.lower() for value in record.imports})
        exact_score = _exact_score(target, record)
        trust_score = _CORPUS_TRUST.get(record.corpus, 0.75)
        recency_score = _recency_score(record.created_at)
        score = (
            0.30 * lexical_score
            + 0.20 * max(dependency_score, import_score)
            + 0.20 * exact_score
            + 0.15 * trust_score
            + 0.10 * lexical_score
            + 0.05 * recency_score
        )
        ranked.append(
            RankedProofCandidate(
                record=record,
                score=score,
                lexical_score=lexical_score,
                dependency_score=dependency_score,
                import_score=import_score,
                exact_score=exact_score,
                trust_score=trust_score,
                recency_score=recency_score,
                relevance_reason=_relevance_reason(
                    lexical_score=lexical_score,
                    dependency_score=dependency_score,
                    import_score=import_score,
                    exact_score=exact_score,
                ),
                transfer_hint=_transfer_hint(record),
            )
        )

    ranked.sort(key=lambda candidate: candidate.score, reverse=True)
    return ranked


def _tokens(value: str) -> set[str]:
    tokens = {
        match.group(0).lower()
        for match in _TOKEN_RE.finditer(value or "")
        if len(match.group(0)) > 2
    }
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / max(1, len(left.union(right)))


def _overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / max(1, len(left))


def _exact_score(target: AssistantTargetSnapshot, record: UnifiedProofSearchRecord) -> float:
    haystack = " ".join(
        [
            record.theorem_name,
            record.theorem_statement,
            record.display_title,
            record.module,
            record.source_path,
        ]
    ).lower()
    exact_parts = [
        target.target_statement,
        target.lean_template,
        target.source_title,
    ]
    for part in exact_parts:
        normalized = " ".join((part or "").lower().split())
        if normalized and len(normalized) > 24 and normalized in haystack:
            return 1.0
    target_names = Counter(_tokens(" ".join(exact_parts)))
    if not target_names:
        return 0.0
    matched = sum(count for token, count in target_names.items() if token in haystack)
    return min(1.0, matched / max(1, sum(target_names.values())))


def _pucb_score(
    *,
    base_score: float,
    visits: int,
    total_visits: int,
    exploration_c: float,
    failure_penalty: float,
) -> float:
    quality = _quality_score(base_score=base_score)
    exploration = exploration_c * math.sqrt(max(1, total_visits)) / (max(0, visits) + 1)
    return quality + exploration - max(0.0, failure_penalty)


def _quality_score(*, base_score: float) -> float:
    return min(1.0, max(0.0, base_score))


def _recency_score(created_at: str) -> float:
    if not created_at:
        return 0.0
    # ISO timestamps sort lexicographically; give a tiny stable bonus to dated records.
    return 0.5


def _dedupe_keys(record: UnifiedProofSearchRecord) -> set[str]:
    return set(record.dedupe_keys() or {f"{record.corpus}:{record.search_id}"})


def _record_similarity(left: UnifiedProofSearchRecord, right: UnifiedProofSearchRecord) -> float:
    if left.theorem_statement_hash and left.theorem_statement_hash == right.theorem_statement_hash:
        return 1.0
    left_tokens = _tokens(
        " ".join([left.theorem_statement, left.theorem_name, " ".join(left.dependency_names)])
    )
    right_tokens = _tokens(
        " ".join([right.theorem_statement, right.theorem_name, " ".join(right.dependency_names)])
    )
    return _jaccard(left_tokens, right_tokens)


def _relevance_reason(
    *,
    lexical_score: float,
    dependency_score: float,
    import_score: float,
    exact_score: float,
) -> str:
    reasons: list[str] = []
    if exact_score >= 0.5:
        reasons.append("strong theorem/statement term overlap")
    if dependency_score > 0:
        reasons.append("shares requested dependencies")
    if import_score > 0:
        reasons.append("shares imports")
    if lexical_score > 0:
        reasons.append("lexically similar to the active target")
    return "; ".join(reasons) or "selected as a diversified verified proof support"


def _transfer_hint(record: UnifiedProofSearchRecord) -> str:
    deps = ", ".join(record.dependency_names[:5])
    imports = ", ".join(record.imports[:5])
    if deps:
        return f"Check dependency/tactic transfer around: {deps}."
    if imports:
        return f"Check reusable Mathlib/import context around: {imports}."
    if record.proof_description:
        return "Use the proof description to identify reusable decomposition or tactic structure."
    return "Compare theorem statement shape and Lean code for reusable proof patterns."
