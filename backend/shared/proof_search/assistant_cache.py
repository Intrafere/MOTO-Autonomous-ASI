"""Persistent cache for Assistant proof-support ranking state."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backend.shared.config import system_config
from backend.shared.proof_search.assistant_models import (
    AssistantProofPack,
    AssistantTargetSnapshot,
)

DEFAULT_MAX_ASSISTANT_CACHE_TARGETS = 128


@dataclass(frozen=True)
class AssistantCandidateStats:
    """Persisted ranking state for one proof candidate under one target."""

    visits: int = 0
    failure_penalty: float = 0.0


def default_assistant_cache_path() -> Path:
    return Path(system_config.data_dir) / "proof_search" / "assistant_ranker.sqlite"


class AssistantRankCache:
    """Small SQLite cache over canonical proof records.

    This is intentionally rebuildable. Canonical proof storage remains the source
    of truth; this cache only keeps target-scoped ranking, pack, and goal reuse
    metadata for the non-blocking Assistant role.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else default_assistant_cache_path()

    def load_cached_pack(
        self,
        *,
        target_hash: str,
        goal_hash: str = "",
    ) -> AssistantProofPack | None:
        if not self.db_path.exists():
            return None
        with closing(self._connect()) as conn:
            self._create_schema(conn)
            row = conn.execute(
                """
                SELECT pack_json
                FROM assistant_proof_packs
                WHERE target_hash = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (target_hash,),
            ).fetchone()
            if row is None and goal_hash:
                row = conn.execute(
                    """
                    SELECT packs.pack_json
                    FROM assistant_goal_cache AS goals
                    JOIN assistant_proof_packs AS packs
                      ON packs.target_hash = goals.pack_target_hash
                    WHERE goals.goal_hash = ?
                    ORDER BY packs.created_at DESC
                    LIMIT 1
                    """,
                    (goal_hash,),
                ).fetchone()
            if row is None:
                return None
        try:
            pack = AssistantProofPack.model_validate_json(row["pack_json"])
        except Exception:
            return None
        freshness = "cached" if pack.target_hash == target_hash else "stale-but-best-known"
        return pack.model_copy(update={"target_hash": target_hash, "freshness": freshness})

    def load_candidate_stats(self, target_hash: str) -> dict[str, AssistantCandidateStats]:
        if not self.db_path.exists():
            return {}
        with closing(self._connect()) as conn:
            self._create_schema(conn)
            rows = conn.execute(
                """
                SELECT search_id, visits, failure_penalty
                FROM assistant_proof_candidates
                WHERE target_hash = ?
                """,
                (target_hash,),
            ).fetchall()
        return {
            row["search_id"]: AssistantCandidateStats(
                visits=int(row["visits"] or 0),
                failure_penalty=float(row["failure_penalty"] or 0.0),
            )
            for row in rows
        }

    def upsert_candidates(
        self,
        *,
        target_hash: str,
        candidates: Iterable[dict[str, Any]],
    ) -> None:
        now = _now_iso()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            self._create_schema(conn)
            conn.executemany(
                """
                INSERT INTO assistant_proof_candidates (
                    target_hash,
                    search_id,
                    proof_source,
                    proof_id,
                    theorem_statement_hash,
                    lean_code_hash,
                    query_variant,
                    retrieval_score,
                    exact_match_score,
                    semantic_score,
                    dependency_overlap_score,
                    corpus_trust_score,
                    recency_score,
                    duplicate_group,
                    created_at,
                    updated_at
                )
                VALUES (
                    :target_hash,
                    :search_id,
                    :proof_source,
                    :proof_id,
                    :theorem_statement_hash,
                    :lean_code_hash,
                    :query_variant,
                    :retrieval_score,
                    :exact_match_score,
                    :semantic_score,
                    :dependency_overlap_score,
                    :corpus_trust_score,
                    :recency_score,
                    :duplicate_group,
                    :created_at,
                    :updated_at
                )
                ON CONFLICT(target_hash, search_id) DO UPDATE SET
                    proof_source = excluded.proof_source,
                    proof_id = excluded.proof_id,
                    theorem_statement_hash = excluded.theorem_statement_hash,
                    lean_code_hash = excluded.lean_code_hash,
                    query_variant = excluded.query_variant,
                    retrieval_score = excluded.retrieval_score,
                    exact_match_score = excluded.exact_match_score,
                    semantic_score = excluded.semantic_score,
                    dependency_overlap_score = excluded.dependency_overlap_score,
                    corpus_trust_score = excluded.corpus_trust_score,
                    recency_score = excluded.recency_score,
                    duplicate_group = excluded.duplicate_group,
                    updated_at = excluded.updated_at
                """,
                (
                    {
                        **candidate,
                        "target_hash": target_hash,
                        "created_at": now,
                        "updated_at": now,
                    }
                    for candidate in candidates
                ),
            )
            conn.commit()

    def record_pack(
        self,
        *,
        snapshot: AssistantTargetSnapshot,
        pack: AssistantProofPack,
        selected_search_ids: list[str],
    ) -> None:
        now = _now_iso()
        pack_id = hashlib.sha256(
            f"{pack.target_hash}\n{pack.created_at}\n{','.join(selected_search_ids)}".encode(
                "utf-8"
            )
        ).hexdigest()
        metadata_pack_json = json.dumps(pack.metadata_only_dump(), ensure_ascii=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            self._create_schema(conn)
            conn.execute(
                """
                INSERT OR REPLACE INTO assistant_proof_packs (
                    pack_id,
                    target_hash,
                    created_at,
                    workflow_mode,
                    target_kind,
                    query_summary,
                    selected_candidate_ids_json,
                    warnings_json,
                    pack_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pack_id,
                    pack.target_hash,
                    pack.created_at,
                    pack.workflow_mode,
                    pack.target_kind,
                    pack.query_summary,
                    json.dumps(selected_search_ids),
                    json.dumps(pack.warnings),
                    metadata_pack_json,
                ),
            )
            conn.executemany(
                """
                UPDATE assistant_proof_candidates
                SET visits = visits + 1,
                    last_selected_at = ?,
                    updated_at = ?
                WHERE target_hash = ? AND search_id = ?
                """,
                ((now, now, pack.target_hash, search_id) for search_id in selected_search_ids),
            )
            goal_hash = goal_hash_for_snapshot(snapshot)
            if goal_hash:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO assistant_goal_cache (
                        goal_hash,
                        normalized_goal_text,
                        imports_hash,
                        source_context_hash,
                        result_kind,
                        proof_id,
                        tactic_sequence_hash,
                        feedback_summary,
                        pack_target_hash,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        goal_hash,
                        _normalize_goal_text(snapshot),
                        _hash_json(snapshot.imports),
                        _hash_text("\n".join([snapshot.source_type, snapshot.source_id])),
                        "retrieved_support",
                        selected_search_ids[0] if selected_search_ids else "",
                        "",
                        _trim_feedback(snapshot),
                        pack.target_hash,
                        now,
                    ),
                )
            self._prune_cache(conn, max_targets=DEFAULT_MAX_ASSISTANT_CACHE_TARGETS)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_proof_candidates (
                target_hash TEXT NOT NULL,
                search_id TEXT NOT NULL,
                proof_source TEXT NOT NULL,
                proof_id TEXT NOT NULL,
                theorem_statement_hash TEXT NOT NULL,
                lean_code_hash TEXT NOT NULL,
                query_variant TEXT NOT NULL DEFAULT '',
                retrieval_score REAL NOT NULL DEFAULT 0,
                exact_match_score REAL NOT NULL DEFAULT 0,
                semantic_score REAL NOT NULL DEFAULT 0,
                dependency_overlap_score REAL NOT NULL DEFAULT 0,
                corpus_trust_score REAL NOT NULL DEFAULT 0,
                recency_score REAL NOT NULL DEFAULT 0,
                visits INTEGER NOT NULL DEFAULT 0,
                last_selected_at TEXT,
                failure_penalty REAL NOT NULL DEFAULT 0,
                duplicate_group TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (target_hash, search_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_proof_packs (
                pack_id TEXT PRIMARY KEY,
                target_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                workflow_mode TEXT NOT NULL,
                target_kind TEXT NOT NULL,
                query_summary TEXT NOT NULL,
                selected_candidate_ids_json TEXT NOT NULL,
                warnings_json TEXT NOT NULL,
                pack_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_goal_cache (
                goal_hash TEXT PRIMARY KEY,
                normalized_goal_text TEXT NOT NULL,
                imports_hash TEXT NOT NULL,
                source_context_hash TEXT NOT NULL,
                result_kind TEXT NOT NULL,
                proof_id TEXT NOT NULL,
                tactic_sequence_hash TEXT NOT NULL,
                feedback_summary TEXT NOT NULL,
                pack_target_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assistant_candidates_target ON assistant_proof_candidates(target_hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assistant_packs_target ON assistant_proof_packs(target_hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assistant_goal_cache_pack ON assistant_goal_cache(pack_target_hash)"
        )

    def _prune_cache(self, conn: sqlite3.Connection, *, max_targets: int) -> None:
        rows = conn.execute(
            """
            SELECT target_hash, MAX(created_at) AS newest_pack
            FROM assistant_proof_packs
            GROUP BY target_hash
            ORDER BY newest_pack DESC
            """
        ).fetchall()
        stale_targets = [row["target_hash"] for row in rows[max(1, max_targets):]]
        if not stale_targets:
            return
        placeholders = ",".join("?" for _ in stale_targets)
        conn.execute(
            f"DELETE FROM assistant_proof_packs WHERE target_hash IN ({placeholders})",
            stale_targets,
        )
        conn.execute(
            f"DELETE FROM assistant_proof_candidates WHERE target_hash IN ({placeholders})",
            stale_targets,
        )
        conn.execute(
            f"DELETE FROM assistant_goal_cache WHERE pack_target_hash IN ({placeholders})",
            stale_targets,
        )


def goal_hash_for_snapshot(snapshot: AssistantTargetSnapshot) -> str:
    goal_text = _normalize_goal_text(snapshot)
    if not goal_text:
        return ""
    return _hash_text(
        "\n\n".join(
            [
                snapshot.workflow_mode,
                snapshot.target_kind,
                goal_text,
                _hash_json(snapshot.imports),
                _source_scope_hash(snapshot),
            ]
        )
    )


def _normalize_goal_text(snapshot: AssistantTargetSnapshot) -> str:
    if _is_broad_workflow_target(snapshot):
        return " ".join(
            part
            for part in [
                _normalize_broad_fragment(snapshot.user_prompt),
                _normalize_broad_fragment(snapshot.current_prompt_or_topic),
                _normalize_broad_fragment(snapshot.writing_goal),
                _normalize_broad_fragment(snapshot.outline_summary),
                _normalize_broad_fragment(snapshot.paper_or_proof_draft_summary),
                _normalize_broad_fragment(snapshot.accepted_memory_summary),
            ]
            if part
        )
    return " ".join(
        part
        for part in [
            _normalize_lean_fragment(snapshot.target_statement),
            _normalize_lean_fragment(snapshot.lean_template),
            _normalize_lean_fragment(snapshot.lean_error),
        ]
        if part
    )


def _is_broad_workflow_target(snapshot: AssistantTargetSnapshot) -> bool:
    return snapshot.target_kind in {
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
    }


def _normalize_broad_fragment(value: str) -> str:
    text = " ".join((value or "").split())
    if not text:
        return ""
    return text[:2000]


def _source_scope_hash(snapshot: AssistantTargetSnapshot) -> str:
    if _is_broad_workflow_target(snapshot):
        return _hash_text(snapshot.source_type or snapshot.active_mode or snapshot.workflow_mode)
    return _hash_text("\n".join([snapshot.source_type, snapshot.source_id]))


def _normalize_lean_fragment(value: str) -> str:
    text = " ".join((value or "").split())
    if not text:
        return ""
    # Lean diagnostics often include volatile temp-file positions; those should
    # not prevent reuse for the same mathematical goal/error shape.
    text = re.sub(r"\bline\s+\d+\s*,\s*column\s+\d+\b", "line <n>, column <n>", text, flags=re.I)
    text = re.sub(r":\d+:\d+:", ":<n>:<n>:", text)
    return text


def _trim_feedback(snapshot: AssistantTargetSnapshot) -> str:
    feedback = " ".join(
        part
        for part in [
            snapshot.rejection_feedback,
            snapshot.proof_attempt_feedback,
            snapshot.accepted_solver_summary,
        ]
        if part
    )
    feedback = " ".join(feedback.split())
    return feedback[:600]


def _hash_json(values: list[str]) -> str:
    return _hash_text(json.dumps(sorted(values or []), ensure_ascii=True))


def _hash_text(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
