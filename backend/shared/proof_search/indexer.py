"""SQLite/FTS index for unified proof search."""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable

from backend.shared.proof_search.models import (
    CorpusOverview,
    ProofSearchRequest,
    ProofSearchResponse,
    UnifiedProofSearchRecord,
)

RESULT_CAP = 7


class ProofSearchIndexer:
    """Small local SQLite FTS index for proof metadata and bounded retrieval."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def rebuild(self, records: Iterable[UnifiedProofSearchRecord]) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        unique_records = {record.search_id: record for record in records}
        with closing(self._connect()) as conn:
            self._create_schema(conn)
            conn.execute("DELETE FROM proof_fts")
            conn.execute("DELETE FROM proof_records")
            conn.executemany(
                _PROOF_RECORD_INSERT_SQL,
                (self._record_payload(record) for record in unique_records.values()),
            )
            conn.executemany(
                _PROOF_FTS_INSERT_SQL,
                (self._fts_payload(record) for record in unique_records.values()),
            )
            conn.commit()

    def search(self, request: ProofSearchRequest) -> ProofSearchResponse:
        if not self.db_path.exists():
            return ProofSearchResponse(
                results=[],
                result_count=0,
                weak_result_warning="Proof-search index is not built yet.",
            )

        with closing(self._connect()) as conn:
            self._create_schema(conn)
            candidates = self._query_candidates(conn, request)
            records = self._dedupe_and_limit(candidates, request, result_cap=RESULT_CAP)
            corpus_counts = self._corpus_counts(conn, request.corpora)

        if not request.hydrate_lean_code:
            records = [record.model_copy(update={"lean_code": ""}) for record in records]

        warning = None
        if not records:
            warning = (
                "No proof records matched this query. Try a theorem name, dependency, "
                "module, import, or a broader goal statement."
            )

        return ProofSearchResponse(
            results=records,
            result_count=len(records),
            next_cursor=None,
            searched_corpora=sorted(set(request.corpora)),
            corpus_counts=corpus_counts,
            ranking_notes=(
                "Ranked with SQLite FTS lexical matching, exact import/dependency boosts, "
                "and duplicate suppression by fingerprint/local ID/hash."
            ),
            weak_result_warning=warning,
        )

    def search_candidate_pool(
        self,
        request: ProofSearchRequest,
        *,
        pool_limit: int,
        exclude_corpus_scopes: Iterable[str] | None = None,
        exclude_session_ids: Iterable[str] | None = None,
    ) -> list[UnifiedProofSearchRecord]:
        """Return a wider internal candidate pool without changing public route caps."""
        if not self.db_path.exists():
            return []

        with closing(self._connect()) as conn:
            self._create_schema(conn)
            candidates = self._query_candidates(
                conn,
                request,
                candidate_limit=pool_limit * 4,
                exclude_corpus_scopes=exclude_corpus_scopes,
                exclude_session_ids=exclude_session_ids,
            )
            records = self._dedupe_and_limit(candidates, request, result_cap=pool_limit)

        if not request.hydrate_lean_code:
            records = [record.model_copy(update={"lean_code": ""}) for record in records]
        return records

    def get_record(
        self,
        *,
        corpus: str,
        proof_id: str,
        session_id: str | None = None,
    ) -> UnifiedProofSearchRecord | None:
        """Fetch one indexed record for detail/hydration flows."""
        if not self.db_path.exists():
            return None

        with closing(self._connect()) as conn:
            self._create_schema(conn)
            clauses = ["corpus = ?", "(proof_id = ? OR search_id = ?)"]
            params: list[Any] = [corpus, proof_id, proof_id]
            if session_id:
                clauses.append("session_id = ?")
                params.append(session_id)
            row = conn.execute(
                f"""
                SELECT *
                FROM proof_records
                WHERE {' AND '.join(clauses)}
                ORDER BY
                    CASE corpus_scope
                        WHEN 'active' THEN 0
                        WHEN 'current' THEN 1
                        ELSE 2
                    END,
                    created_at DESC
                LIMIT 1
                """,
                params,
            ).fetchone()

        return self._row_to_record(row) if row else None

    def overview(self, corpora: Iterable[str] | None = None) -> CorpusOverview:
        if not self.db_path.exists():
            return CorpusOverview(
                total_records=0,
                verified_records=0,
                partial_records=0,
                failed_attempt_records=0,
                corpora=[],
                search_fields=_SEARCH_FIELDS,
                recommended_queries=_RECOMMENDED_QUERIES,
            )

        with closing(self._connect()) as conn:
            self._create_schema(conn)
            if corpora is None:
                rows = conn.execute("SELECT * FROM proof_records").fetchall()
            else:
                corpus_list = list(corpora)
                if corpus_list:
                    placeholders = ",".join("?" for _ in corpus_list)
                    rows = conn.execute(
                        f"SELECT * FROM proof_records WHERE corpus IN ({placeholders})",
                        corpus_list,
                    ).fetchall()
                else:
                    rows = []

        records = [self._row_to_record(row) for row in rows]
        corpus_counts: Counter[str] = Counter(record.corpus for record in records)
        source_counts: Counter[str] = Counter(record.source_kind for record in records)
        novelty_counts: Counter[str] = Counter(
            record.novelty_tier or "unknown" for record in records
        )
        module_counts: Counter[str] = Counter(
            value for record in records for value in [record.module or record.source_path] if value
        )
        import_counts: Counter[str] = Counter(
            value for record in records for value in record.imports if value
        )
        dependency_counts: Counter[str] = Counter(
            value for record in records for value in record.dependency_names if value
        )
        tag_counts: Counter[str] = Counter(
            value for record in records for value in [*record.topic_tags, *record.domain_tags] if value
        )

        corpora = [
            {
                "id": corpus,
                "count": count,
                "freshness": self._freshness_for_corpus(corpus, records),
                "description": _CORPUS_DESCRIPTIONS.get(corpus, "Proof records"),
            }
            for corpus, count in sorted(corpus_counts.items())
        ]

        return CorpusOverview(
            total_records=len(records),
            verified_records=source_counts.get("verified_proof", 0),
            partial_records=source_counts.get("partial_proof", 0),
            failed_attempt_records=source_counts.get("failed_attempt", 0),
            corpora=corpora,
            novelty_distribution=dict(novelty_counts),
            top_modules=_top_counts(module_counts),
            top_imports=_top_counts(import_counts),
            top_dependencies=_top_counts(dependency_counts),
            top_tags=_top_counts(tag_counts),
            search_fields=_SEARCH_FIELDS,
            recommended_queries=_RECOMMENDED_QUERIES,
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS proof_records (
                search_id TEXT PRIMARY KEY,
                corpus TEXT NOT NULL,
                corpus_scope TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                proof_id TEXT NOT NULL,
                external_fingerprint TEXT NOT NULL,
                release_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_title TEXT NOT NULL,
                display_title TEXT NOT NULL,
                theorem_name TEXT NOT NULL,
                theorem_statement TEXT NOT NULL,
                informal_statement TEXT NOT NULL,
                proof_description TEXT NOT NULL,
                formal_sketch TEXT NOT NULL,
                lean_code TEXT NOT NULL,
                lean_code_hash TEXT NOT NULL,
                theorem_statement_hash TEXT NOT NULL,
                imports_json TEXT NOT NULL,
                dependency_names_json TEXT NOT NULL,
                topic_tags_json TEXT NOT NULL,
                domain_tags_json TEXT NOT NULL,
                module TEXT NOT NULL,
                source_path TEXT NOT NULL,
                novelty_tier TEXT NOT NULL,
                novelty_reasoning TEXT NOT NULL,
                verified INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                canonical_uri TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS proof_fts USING fts5(
                search_id UNINDEXED,
                theorem_name,
                theorem_statement,
                informal_statement,
                proof_description,
                formal_sketch,
                source_title,
                module,
                source_path,
                novelty_reasoning,
                dependencies,
                imports,
                tags,
                display_title
            )
            """
        )

    def _record_payload(self, record: UnifiedProofSearchRecord) -> dict[str, Any]:
        payload = record.model_dump(mode="json")
        return {
            **payload,
            "imports_json": json.dumps(record.imports),
            "dependency_names_json": json.dumps(record.dependency_names),
            "topic_tags_json": json.dumps(record.topic_tags),
            "domain_tags_json": json.dumps(record.domain_tags),
            "metadata_json": json.dumps(record.metadata),
            "verified": 1 if record.verified else 0,
        }

    def _fts_payload(self, record: UnifiedProofSearchRecord) -> tuple[str, ...]:
        return (
            record.search_id,
            record.theorem_name,
            record.theorem_statement,
            record.informal_statement,
            record.proof_description,
            record.formal_sketch,
            record.source_title,
            record.module,
            record.source_path,
            record.novelty_reasoning,
            " ".join(record.dependency_names),
            " ".join(record.imports),
            " ".join([*record.topic_tags, *record.domain_tags]),
            record.display_title,
        )

    def _upsert_record(self, conn: sqlite3.Connection, record: UnifiedProofSearchRecord) -> None:
        conn.execute(
            _PROOF_RECORD_INSERT_SQL,
            self._record_payload(record),
        )
        conn.execute("DELETE FROM proof_fts WHERE search_id = ?", (record.search_id,))
        conn.execute(
            _PROOF_FTS_INSERT_SQL,
            self._fts_payload(record),
        )

    def _query_candidates(
        self,
        conn: sqlite3.Connection,
        request: ProofSearchRequest,
        *,
        candidate_limit: int | None = None,
        exclude_corpus_scopes: Iterable[str] | None = None,
        exclude_session_ids: Iterable[str] | None = None,
    ) -> list[tuple[UnifiedProofSearchRecord, float]]:
        clauses = []
        params: list[Any] = []

        if request.corpora:
            placeholders = ",".join("?" for _ in request.corpora)
            clauses.append(f"r.corpus IN ({placeholders})")
            params.extend(request.corpora)
        if request.verified_only:
            clauses.append("r.verified = 1")
        if not request.include_partial:
            clauses.append("r.source_kind != 'partial_proof'")
        if not request.include_failed:
            clauses.append("r.source_kind != 'failed_attempt'")
        if request.novelty_filters:
            placeholders = ",".join("?" for _ in request.novelty_filters)
            clauses.append(f"r.novelty_tier IN ({placeholders})")
            params.extend(request.novelty_filters)
        if request.module_filters:
            module_clauses = []
            for module in request.module_filters:
                module_clauses.append("(r.module LIKE ? OR r.source_path LIKE ?)")
                params.extend([f"%{module}%", f"%{module}%"])
            clauses.append(f"({' OR '.join(module_clauses)})")
        if request.source_filters:
            placeholders = ",".join("?" for _ in request.source_filters)
            clauses.append(f"r.source_type IN ({placeholders})")
            params.extend(request.source_filters)
        if request.exclude_ids:
            placeholders = ",".join("?" for _ in request.exclude_ids)
            clauses.append(f"r.search_id NOT IN ({placeholders})")
            params.extend(request.exclude_ids)
        excluded_scopes = [scope for scope in (exclude_corpus_scopes or []) if scope]
        if excluded_scopes:
            placeholders = ",".join("?" for _ in excluded_scopes)
            clauses.append(f"(r.corpus = 'syntheticlib4' OR r.corpus_scope NOT IN ({placeholders}))")
            params.extend(excluded_scopes)
        excluded_sessions = [session_id for session_id in (exclude_session_ids or []) if session_id]
        if excluded_sessions:
            placeholders = ",".join("?" for _ in excluded_sessions)
            clauses.append(f"r.session_id NOT IN ({placeholders})")
            params.extend(excluded_sessions)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        fts_query = _build_fts_query(
            " ".join(
                [
                    request.query,
                    request.goal_statement,
                    " ".join(request.imports),
                    " ".join(request.dependency_names),
                ]
            )
        )
        limit = candidate_limit or max(RESULT_CAP * 4, min(request.limit, RESULT_CAP) * 4)

        if fts_query:
            sql = f"""
                SELECT r.*, bm25(proof_fts) AS rank
                FROM proof_fts
                JOIN proof_records r ON r.search_id = proof_fts.search_id
                {where_sql} {'AND' if where_sql else 'WHERE'} proof_fts MATCH ?
                ORDER BY rank ASC
                LIMIT ?
            """
            rows = conn.execute(sql, [*params, fts_query, limit]).fetchall()
        else:
            sql = f"SELECT r.*, 0.0 AS rank FROM proof_records r {where_sql} LIMIT ?"
            rows = conn.execute(sql, [*params, limit]).fetchall()

        candidates: list[tuple[UnifiedProofSearchRecord, float]] = []
        for row in rows:
            record = self._row_to_record(row)
            score = -float(row["rank"] or 0.0)
            score += self._exact_boost(record, request)
            candidates.append((record, score))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates

    def _dedupe_and_limit(
        self,
        candidates: list[tuple[UnifiedProofSearchRecord, float]],
        request: ProofSearchRequest,
        *,
        result_cap: int,
    ) -> list[UnifiedProofSearchRecord]:
        result_limit = min(request.limit, result_cap)
        used_keys: set[str] = set()
        records: list[UnifiedProofSearchRecord] = []

        for record, _score in candidates:
            keys = record.dedupe_keys()
            if keys and any(key in used_keys for key in keys):
                continue
            used_keys.update(keys)
            records.append(record)
            if len(records) >= result_limit:
                break

        return records

    def _exact_boost(self, record: UnifiedProofSearchRecord, request: ProofSearchRequest) -> float:
        boost = 0.0
        imports = {value.lower() for value in request.imports}
        dependencies = {value.lower() for value in request.dependency_names}
        if imports.intersection(value.lower() for value in record.imports):
            boost += 3.0
        if dependencies.intersection(value.lower() for value in record.dependency_names):
            boost += 3.0
        query = (request.query or "").strip().lower()
        if query and query in record.theorem_name.lower():
            boost += 4.0
        if record.verified:
            boost += 1.0
        return boost

    def _row_to_record(self, row: sqlite3.Row) -> UnifiedProofSearchRecord:
        def _json_list(field: str) -> list[str]:
            try:
                value = json.loads(row[field] or "[]")
                return [str(item) for item in value if str(item).strip()]
            except json.JSONDecodeError:
                return []

        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}

        return UnifiedProofSearchRecord(
            search_id=row["search_id"],
            corpus=row["corpus"],
            corpus_scope=row["corpus_scope"],
            source_kind=row["source_kind"],
            proof_id=row["proof_id"],
            external_fingerprint=row["external_fingerprint"],
            release_id=row["release_id"],
            session_id=row["session_id"],
            source_type=row["source_type"],
            source_id=row["source_id"],
            source_title=row["source_title"],
            display_title=row["display_title"],
            theorem_name=row["theorem_name"],
            theorem_statement=row["theorem_statement"],
            informal_statement=row["informal_statement"],
            proof_description=row["proof_description"],
            formal_sketch=row["formal_sketch"],
            lean_code=row["lean_code"],
            lean_code_hash=row["lean_code_hash"],
            theorem_statement_hash=row["theorem_statement_hash"],
            imports=_json_list("imports_json"),
            dependency_names=_json_list("dependency_names_json"),
            topic_tags=_json_list("topic_tags_json"),
            domain_tags=_json_list("domain_tags_json"),
            module=row["module"],
            source_path=row["source_path"],
            novelty_tier=row["novelty_tier"],
            novelty_reasoning=row["novelty_reasoning"],
            verified=bool(row["verified"]),
            created_at=row["created_at"],
            canonical_uri=row["canonical_uri"],
            metadata=metadata,
        )

    def _corpus_counts(self, conn: sqlite3.Connection, corpora: Iterable[str] | None = None) -> dict[str, int]:
        corpus_list = list(corpora or [])
        if corpus_list:
            placeholders = ",".join("?" for _ in corpus_list)
            rows = conn.execute(
                f"""
                SELECT corpus, COUNT(*) AS count
                FROM proof_records
                WHERE corpus IN ({placeholders})
                GROUP BY corpus
                """,
                corpus_list,
            ).fetchall()
        elif corpora is not None:
            rows = []
        else:
            rows = conn.execute(
                "SELECT corpus, COUNT(*) AS count FROM proof_records GROUP BY corpus"
            ).fetchall()
        return {str(row["corpus"]): int(row["count"]) for row in rows}

    def _freshness_for_corpus(
        self,
        corpus: str,
        records: list[UnifiedProofSearchRecord],
    ) -> str:
        if corpus == "syntheticlib4":
            release_ids = sorted({record.release_id for record in records if record.release_id})
            return f"release {release_ids[-1]}" if release_ids else "local fixture"
        return "current"


def _build_fts_query(raw_query: str) -> str:
    terms = [term for term in re.findall(r"[A-Za-z0-9_'.]+", raw_query or "") if len(term) > 1]
    unique_terms = []
    for term in terms:
        cleaned = term.replace("'", "''")
        if cleaned not in unique_terms:
            unique_terms.append(cleaned)
    return " OR ".join(f'"{term}"' for term in unique_terms[:16])


def _top_counts(counter: Counter[str], limit: int = 10) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


_PROOF_RECORD_INSERT_SQL = """
    INSERT OR REPLACE INTO proof_records (
        search_id, corpus, corpus_scope, source_kind, proof_id,
        external_fingerprint, release_id, session_id, source_type, source_id,
        source_title, display_title, theorem_name, theorem_statement,
        informal_statement, proof_description, formal_sketch, lean_code,
        lean_code_hash, theorem_statement_hash, imports_json,
        dependency_names_json, topic_tags_json, domain_tags_json, module,
        source_path, novelty_tier, novelty_reasoning, verified, created_at,
        canonical_uri, metadata_json
    ) VALUES (
        :search_id, :corpus, :corpus_scope, :source_kind, :proof_id,
        :external_fingerprint, :release_id, :session_id, :source_type, :source_id,
        :source_title, :display_title, :theorem_name, :theorem_statement,
        :informal_statement, :proof_description, :formal_sketch, :lean_code,
        :lean_code_hash, :theorem_statement_hash, :imports_json,
        :dependency_names_json, :topic_tags_json, :domain_tags_json, :module,
        :source_path, :novelty_tier, :novelty_reasoning, :verified, :created_at,
        :canonical_uri, :metadata_json
    )
"""

_PROOF_FTS_INSERT_SQL = """
    INSERT INTO proof_fts (
        search_id, theorem_name, theorem_statement, informal_statement,
        proof_description, formal_sketch, source_title, module, source_path,
        novelty_reasoning, dependencies, imports, tags, display_title
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


_CORPUS_DESCRIPTIONS = {
    "moto": "Autonomous MOTO proof records from canonical ProofDatabase stores",
    "manual": "Active and archived manual Aggregator/Compiler proof records",
    "leanoj": "LeanOJ verified proof records registered into MOTO proof storage",
    "syntheticlib4": "Authorized local SyntheticLib4 snapshot or offline fixtures",
}

_SEARCH_FIELDS = [
    "theorem_name",
    "theorem_statement",
    "informal_statement",
    "proof_description",
    "formal_sketch",
    "source_title",
    "module",
    "source_path",
    "imports",
    "dependency_names",
    "topic_tags",
    "domain_tags",
]

_RECOMMENDED_QUERIES = [
    "Search by theorem goal: finite sum cancellation over Nat",
    "Search by dependency: Finset.sum_congr Nat.add_comm",
    "Search by module or source path: SyntheticLib4.Finset",
]

