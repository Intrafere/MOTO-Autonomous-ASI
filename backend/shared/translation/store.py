"""SQLite store for Auto Top-p Translate.

Holds the final translation of each unit (a cross-document cache, masked with
the unit's own placeholder tokens), document plans, the doc keys registered on
them, one persisted job state per document plan and language, and the
translations users requested by doc key and language under
``<data root>/auto_translations/``. A plan is identified by its
content hash and several doc keys may share one plan; each doc key points at
exactly one plan. A plan that loses its last doc key is deleted together with
its job states. Everything here is a derived cache of translations for outputs
that live elsewhere, so a schema-version mismatch simply drops and recreates
the tables.

Every public method is synchronous and opens one short-lived connection under a
process-local lock; async callers use ``asyncio.to_thread``. The store path is
resolved from ``system_config.data_dir`` on every call so runtime-root
rebinding is honored.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from backend.shared.config import system_config

TRANSLATIONS_DIR_NAME = "auto_translations"
STORE_FILE_NAME = "translation_store.sqlite3"
SCHEMA_VERSION = 5

# Bounded IN (...) lists stay below SQLite's host-parameter limit.
_QUERY_CHUNK_SIZE = 500
# Cleared by ``clear_all``; a user's translation requests outlive a clear.
_TRANSLATION_TABLES = (
    "segment_translations",
    "document_plans",
    "document_keys",
    "translation_jobs",
)
_DATA_TABLES = (*_TRANSLATION_TABLES, "translation_requests")
# Tables of older schema versions, dropped when the schema is recreated.
_LEGACY_TABLES = ("documents", "segment_feedback", "segment_sources")
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS segment_translations (
    source_hash TEXT NOT NULL,
    language_key TEXT NOT NULL,
    translation TEXT NOT NULL,
    translator_notes TEXT NOT NULL,
    submitter_role TEXT NOT NULL,
    submitter_model TEXT NOT NULL,
    validator_model TEXT NOT NULL,
    rounds INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_hash, language_key)
);
CREATE TABLE IF NOT EXISTS document_plans (
    doc_hash TEXT PRIMARY KEY,
    plan_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS document_keys (
    doc_key TEXT PRIMARY KEY,
    doc_hash TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    priority_class INTEGER NOT NULL,
    origin TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    last_viewed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_document_keys_doc_hash ON document_keys (doc_hash);
CREATE TABLE IF NOT EXISTS translation_jobs (
    doc_hash TEXT NOT NULL,
    language_key TEXT NOT NULL,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (doc_hash, language_key)
);
CREATE TABLE IF NOT EXISTS translation_requests (
    doc_key TEXT NOT NULL,
    language_key TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    PRIMARY KEY (doc_key, language_key)
);
"""
# One registered doc key joined with the plan it points at.
_DOCUMENT_SELECT = (
    "SELECT k.doc_hash AS doc_hash, k.doc_key AS doc_key, k.kind AS kind, "
    "k.title AS title, k.priority_class AS priority_class, p.plan_json AS plan_json, "
    "k.origin AS origin, k.created_at AS created_at, "
    "k.last_seen_at AS last_seen_at, k.last_viewed_at AS last_viewed_at "
    "FROM document_keys k JOIN document_plans p ON p.doc_hash = k.doc_hash"
)


@dataclass(frozen=True)
class DocumentRecord:
    """One registered doc key and the stored segmentation plan it points at.

    The plan (``plan_json``) belongs to the content hash and is shared by
    every key on that hash; the other fields belong to the key.
    """

    doc_hash: str
    doc_key: str
    kind: str
    title: str
    priority_class: int
    plan_json: str
    origin: str  # "auto" | "backlog" | "viewer"
    created_at: str
    last_seen_at: str
    last_viewed_at: str  # "" when never viewed


@dataclass(frozen=True)
class SegmentTranslation:
    """The final translation of one unit's masked text, in unit-local tokens.

    Keyed by source hash and language, it is reused by every document that
    contains the same unit.
    """

    source_hash: str
    language_key: str
    translation: str
    translator_notes: str
    submitter_role: str
    submitter_model: str
    validator_model: str
    rounds: int
    created_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _chunks(values: Sequence[str]) -> Iterator[Sequence[str]]:
    for start in range(0, len(values), _QUERY_CHUNK_SIZE):
        yield values[start:start + _QUERY_CHUNK_SIZE]


def _unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        doc_hash=row["doc_hash"],
        doc_key=row["doc_key"],
        kind=row["kind"],
        title=row["title"],
        priority_class=int(row["priority_class"]),
        plan_json=row["plan_json"],
        origin=row["origin"],
        created_at=row["created_at"],
        last_seen_at=row["last_seen_at"],
        last_viewed_at=row["last_viewed_at"],
    )


class TranslationStore:
    """Synchronous SQLite persistence for segment translations, documents, and jobs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._schema_ready_path: Optional[Path] = None

    def store_path(self) -> Path:
        return Path(system_config.data_dir) / TRANSLATIONS_DIR_NAME / STORE_FILE_NAME

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection inside one committed-or-rolled-back transaction."""
        with self._lock:
            path = self.store_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            existed = path.exists()
            conn = sqlite3.connect(str(path), timeout=30.0)
            try:
                conn.row_factory = sqlite3.Row
                if not existed or self._schema_ready_path != path:
                    self._ensure_schema(conn)
                    self._schema_ready_path = path
                with conn:
                    yield conn
            finally:
                conn.close()

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None or row["value"] != str(SCHEMA_VERSION):
            for table in (*_DATA_TABLES, *_LEGACY_TABLES):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.executescript(_SCHEMA_SQL)
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()

    def register_document(self, record: DocumentRecord) -> None:
        """Register ``record.doc_key`` on the plan of ``record.doc_hash``.

        Inserts the plan when the hash is new (a known hash keeps its plan) and
        upserts only this doc key's row. A key that moves to another hash leaves
        its previous plan, which is deleted together with its job states once
        no key points at it, so callers that carry job progress over to the new
        content read the previous hash's job states before registering.
        Re-registering a key on the same hash keeps its original ``created_at``
        and never moves ``last_seen_at`` / ``last_viewed_at`` backwards.
        """
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO document_plans (doc_hash, plan_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(doc_hash) DO NOTHING
                """,
                (record.doc_hash, record.plan_json, record.created_at),
            )
            previous = conn.execute(
                "SELECT doc_hash FROM document_keys WHERE doc_key = ?",
                (record.doc_key,),
            ).fetchone()
            conn.execute(
                """
                INSERT INTO document_keys (
                    doc_key, doc_hash, kind, title, priority_class, origin,
                    created_at, last_seen_at, last_viewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_key) DO UPDATE SET
                    kind = excluded.kind,
                    title = excluded.title,
                    priority_class = excluded.priority_class,
                    origin = excluded.origin,
                    created_at = CASE WHEN document_keys.doc_hash = excluded.doc_hash
                        THEN document_keys.created_at ELSE excluded.created_at END,
                    last_seen_at = CASE WHEN document_keys.doc_hash = excluded.doc_hash
                        THEN MAX(document_keys.last_seen_at, excluded.last_seen_at)
                        ELSE excluded.last_seen_at END,
                    last_viewed_at = CASE WHEN document_keys.doc_hash = excluded.doc_hash
                        THEN MAX(document_keys.last_viewed_at, excluded.last_viewed_at)
                        ELSE excluded.last_viewed_at END,
                    doc_hash = excluded.doc_hash
                """,
                (
                    record.doc_key,
                    record.doc_hash,
                    record.kind,
                    record.title,
                    int(record.priority_class),
                    record.origin,
                    record.created_at,
                    record.last_seen_at,
                    record.last_viewed_at,
                ),
            )
            if previous is not None and previous["doc_hash"] != record.doc_hash:
                self._prune_unreferenced_plan(conn, previous["doc_hash"])

    @staticmethod
    def _prune_unreferenced_plan(conn: sqlite3.Connection, doc_hash: str) -> None:
        """Delete the plan of ``doc_hash`` and its job states when no doc key points at it."""
        referenced = conn.execute(
            "SELECT 1 FROM document_keys WHERE doc_hash = ? LIMIT 1",
            (doc_hash,),
        ).fetchone()
        if referenced is not None:
            return
        conn.execute("DELETE FROM document_plans WHERE doc_hash = ?", (doc_hash,))
        TranslationStore._delete_job_states(conn, doc_hash)

    @staticmethod
    def _delete_job_states(conn: sqlite3.Connection, doc_hash: str) -> None:
        conn.execute("DELETE FROM translation_jobs WHERE doc_hash = ?", (doc_hash,))

    def mark_documents_viewed(self, doc_keys: Sequence[str], now: str) -> None:
        """Set the seen and viewed times of ``doc_keys`` to ``now`` in one transaction."""
        if not doc_keys:
            return
        with self._connection() as conn:
            conn.executemany(
                "UPDATE document_keys SET last_seen_at = ?, last_viewed_at = ? WHERE doc_key = ?",
                [(now, now, doc_key) for doc_key in doc_keys],
            )

    def load_documents(self) -> List[DocumentRecord]:
        """Every registered doc key with its plan, oldest registration first."""
        with self._connection() as conn:
            rows = conn.execute(f"{_DOCUMENT_SELECT} ORDER BY k.created_at, k.doc_key").fetchall()
        return [_document_from_row(row) for row in rows]

    def translated_hashes_by_language(self) -> Dict[str, Set[str]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT language_key, source_hash FROM segment_translations"
            ).fetchall()
        result: Dict[str, Set[str]] = {}
        for row in rows:
            result.setdefault(row["language_key"], set()).add(row["source_hash"])
        return result

    def get_translations(self, source_hashes: Sequence[str], language_key: str) -> Dict[str, str]:
        hashes = _unique(source_hashes)
        if not hashes:
            return {}
        translations: Dict[str, str] = {}
        with self._connection() as conn:
            for chunk in _chunks(hashes):
                placeholders = ", ".join("?" for _ in chunk)
                rows = conn.execute(
                    "SELECT source_hash, translation FROM segment_translations "
                    f"WHERE language_key = ? AND source_hash IN ({placeholders})",
                    (language_key, *chunk),
                ).fetchall()
                for row in rows:
                    translations[row["source_hash"]] = row["translation"]
        return translations

    def save_translation(self, item: SegmentTranslation) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO segment_translations (
                    source_hash, language_key, translation, translator_notes, submitter_role,
                    submitter_model, validator_model, rounds, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.source_hash,
                    item.language_key,
                    item.translation,
                    item.translator_notes,
                    item.submitter_role,
                    item.submitter_model,
                    item.validator_model,
                    int(item.rounds),
                    item.created_at,
                ),
            )

    def get_job_state(self, doc_hash: str, language_key: str) -> Optional[dict]:
        """Return the persisted job state of a plan and language, or None.

        A stored value that is not a JSON object reads as None.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT state_json FROM translation_jobs WHERE doc_hash = ? AND language_key = ?",
                (doc_hash, language_key),
            ).fetchone()
        if row is None:
            return None
        try:
            state = json.loads(row["state_json"])
        except json.JSONDecodeError:
            return None
        return state if isinstance(state, dict) else None

    def save_job_state(self, doc_hash: str, language_key: str, state: dict) -> None:
        """Persist (replace) the job state of a plan and language.

        Raises TypeError when ``state`` is not a JSON-serializable dict.
        """
        if not isinstance(state, dict):
            raise TypeError("A translation job state must be a dict.")
        state_json = _dumps(state)
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO translation_jobs (doc_hash, language_key, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (doc_hash, language_key, state_json, _now_iso()),
            )

    def load_job_keys(self) -> List[Tuple[str, str]]:
        """Every persisted ``(doc_hash, language_key)`` job state key, sorted."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT doc_hash, language_key FROM translation_jobs ORDER BY doc_hash, language_key"
            ).fetchall()
        return [(row["doc_hash"], row["language_key"]) for row in rows]

    def save_requests(self, pairs: Sequence[Tuple[str, str]], now: str) -> None:
        """Record that ``(doc_key, language_key)`` pairs were requested; a repeated pair keeps its first time."""
        if not pairs:
            return
        with self._connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO translation_requests (doc_key, language_key, requested_at) VALUES (?, ?, ?)",
                [(doc_key, language_key, now) for doc_key, language_key in pairs],
            )

    def load_requests(self) -> Dict[str, Set[str]]:
        """Doc key -> the languages a user requested it in."""
        with self._connection() as conn:
            rows = conn.execute("SELECT doc_key, language_key FROM translation_requests").fetchall()
        requests: Dict[str, Set[str]] = {}
        for row in rows:
            requests.setdefault(row["doc_key"], set()).add(row["language_key"])
        return requests

    def clear_all(self) -> None:
        """Delete every stored translation, plan, doc key, and job state; requests are kept."""
        with self._connection() as conn:
            for table in _TRANSLATION_TABLES:
                conn.execute(f"DELETE FROM {table}")


translation_store = TranslationStore()
