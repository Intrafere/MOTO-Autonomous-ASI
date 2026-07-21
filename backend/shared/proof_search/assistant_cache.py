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
    ASSISTANT_PROOF_PACK_SCHEMA_VERSION,
    AssistantProofPack,
    AssistantTargetSnapshot,
)

DEFAULT_MAX_ASSISTANT_CACHE_TARGETS = 128


@dataclass(frozen=True)
class AssistantCandidateStats:
    """Persisted ranking state for one proof candidate under one target."""

    visits: int = 0
    failure_penalty: float = 0.0


@dataclass(frozen=True)
class AssistantCooldownState:
    """Durable Assistant proof-memory backoff state for one run scope."""

    run_key: str
    zero_attempts_in_batch: int = 0
    zero_cooldown_stage: int = 0
    zero_cooldown_skips_remaining: int = 0
    zero_steady_81_batches: int = 0
    zero_shutdown_active: bool = False
    stagnant_same_count: int = 0
    stagnant_attempts_in_batch: int = 0
    stagnant_cooldown_stage: int = 0
    stagnant_cooldown_skips_remaining: int = 0
    last_signature: str = ""
    last_reason: str = ""
    updated_at: str = ""

    @classmethod
    def empty(cls, run_key: str) -> "AssistantCooldownState":
        return cls(run_key=run_key, updated_at=_now_iso())

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_key": self.run_key,
            "zero_attempts_in_batch": self.zero_attempts_in_batch,
            "zero_cooldown_stage": self.zero_cooldown_stage,
            "zero_cooldown_skips_remaining": self.zero_cooldown_skips_remaining,
            "zero_steady_81_batches": self.zero_steady_81_batches,
            "zero_shutdown_active": self.zero_shutdown_active,
            "stagnant_same_count": self.stagnant_same_count,
            "stagnant_attempts_in_batch": self.stagnant_attempts_in_batch,
            "stagnant_cooldown_stage": self.stagnant_cooldown_stage,
            "stagnant_cooldown_skips_remaining": self.stagnant_cooldown_skips_remaining,
            "last_signature": self.last_signature,
            "last_reason": self.last_reason,
            "updated_at": self.updated_at,
        }


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
        if pack.schema_version != ASSISTANT_PROOF_PACK_SCHEMA_VERSION:
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

    def load_cooldown_state(self, run_key: str) -> AssistantCooldownState:
        if not self.db_path.exists():
            return AssistantCooldownState.empty(run_key)
        with closing(self._connect()) as conn:
            self._create_schema(conn)
            row = conn.execute(
                """
                SELECT *
                FROM assistant_cooldown_state
                WHERE run_key = ?
                LIMIT 1
                """,
                (run_key,),
            ).fetchone()
        if row is None:
            return AssistantCooldownState.empty(run_key)
        return _cooldown_state_from_row(row)

    def save_cooldown_state(self, state: AssistantCooldownState) -> None:
        now_state = state if state.updated_at else _replace_cooldown_state(state, updated_at=_now_iso())
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            self._create_schema(conn)
            conn.execute(
                """
                INSERT INTO assistant_cooldown_state (
                    run_key,
                    zero_attempts_in_batch,
                    zero_cooldown_stage,
                    zero_cooldown_skips_remaining,
                    zero_steady_81_batches,
                    zero_shutdown_active,
                    stagnant_same_count,
                    stagnant_attempts_in_batch,
                    stagnant_cooldown_stage,
                    stagnant_cooldown_skips_remaining,
                    last_signature,
                    last_reason,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_key) DO UPDATE SET
                    zero_attempts_in_batch = excluded.zero_attempts_in_batch,
                    zero_cooldown_stage = excluded.zero_cooldown_stage,
                    zero_cooldown_skips_remaining = excluded.zero_cooldown_skips_remaining,
                    zero_steady_81_batches = excluded.zero_steady_81_batches,
                    zero_shutdown_active = excluded.zero_shutdown_active,
                    stagnant_same_count = excluded.stagnant_same_count,
                    stagnant_attempts_in_batch = excluded.stagnant_attempts_in_batch,
                    stagnant_cooldown_stage = excluded.stagnant_cooldown_stage,
                    stagnant_cooldown_skips_remaining = excluded.stagnant_cooldown_skips_remaining,
                    last_signature = excluded.last_signature,
                    last_reason = excluded.last_reason,
                    updated_at = excluded.updated_at
                """,
                (
                    now_state.run_key,
                    now_state.zero_attempts_in_batch,
                    now_state.zero_cooldown_stage,
                    now_state.zero_cooldown_skips_remaining,
                    now_state.zero_steady_81_batches,
                    int(now_state.zero_shutdown_active),
                    now_state.stagnant_same_count,
                    now_state.stagnant_attempts_in_batch,
                    now_state.stagnant_cooldown_stage,
                    now_state.stagnant_cooldown_skips_remaining,
                    now_state.last_signature,
                    now_state.last_reason,
                    now_state.updated_at,
                ),
            )
            conn.commit()

    def clear_cooldown_state(self, run_key: str | None = None) -> None:
        if not self.db_path.exists():
            return
        with closing(self._connect()) as conn:
            self._create_schema(conn)
            if run_key:
                conn.execute("DELETE FROM assistant_cooldown_state WHERE run_key = ?", (run_key,))
            else:
                conn.execute("DELETE FROM assistant_cooldown_state")
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_cooldown_state (
                run_key TEXT PRIMARY KEY,
                zero_attempts_in_batch INTEGER NOT NULL DEFAULT 0,
                zero_cooldown_stage INTEGER NOT NULL DEFAULT 0,
                zero_cooldown_skips_remaining INTEGER NOT NULL DEFAULT 0,
                zero_steady_81_batches INTEGER NOT NULL DEFAULT 0,
                zero_shutdown_active INTEGER NOT NULL DEFAULT 0,
                stagnant_same_count INTEGER NOT NULL DEFAULT 0,
                stagnant_attempts_in_batch INTEGER NOT NULL DEFAULT 0,
                stagnant_cooldown_stage INTEGER NOT NULL DEFAULT 0,
                stagnant_cooldown_skips_remaining INTEGER NOT NULL DEFAULT 0,
                last_signature TEXT NOT NULL DEFAULT '',
                last_reason TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            )
            """
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


def _cooldown_state_from_row(row: sqlite3.Row) -> AssistantCooldownState:
    return AssistantCooldownState(
        run_key=str(row["run_key"] or ""),
        zero_attempts_in_batch=int(row["zero_attempts_in_batch"] or 0),
        zero_cooldown_stage=int(row["zero_cooldown_stage"] or 0),
        zero_cooldown_skips_remaining=int(row["zero_cooldown_skips_remaining"] or 0),
        zero_steady_81_batches=int(row["zero_steady_81_batches"] or 0),
        zero_shutdown_active=bool(row["zero_shutdown_active"]),
        stagnant_same_count=int(row["stagnant_same_count"] or 0),
        stagnant_attempts_in_batch=int(row["stagnant_attempts_in_batch"] or 0),
        stagnant_cooldown_stage=int(row["stagnant_cooldown_stage"] or 0),
        stagnant_cooldown_skips_remaining=int(row["stagnant_cooldown_skips_remaining"] or 0),
        last_signature=str(row["last_signature"] or ""),
        last_reason=str(row["last_reason"] or ""),
        updated_at=str(row["updated_at"] or ""),
    )


def _replace_cooldown_state(state: AssistantCooldownState, **updates: Any) -> AssistantCooldownState:
    payload = state.to_payload()
    payload.update(updates)
    return AssistantCooldownState(**payload)
