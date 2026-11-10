from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from backend.shared.config import bind_runtime_roots, system_config
from backend.shared.translation import store as store_module
from backend.shared.translation.store import (
    STORE_FILE_NAME,
    TRANSLATIONS_DIR_NAME,
    DocumentRecord,
    SegmentTranslation,
    TranslationStore,
)


def _keys_for_hash(store, doc_hash):
    """Doc keys registered on ``doc_hash``, sorted by key."""
    return sorted(
        (record for record in store.load_documents() if record.doc_hash == doc_hash),
        key=lambda record: record.doc_key,
    )


def _documents_by_key(store):
    """Every registered record, keyed by doc key."""
    return {record.doc_key: record for record in store.load_documents()}


def _row_counts(store):
    """Stored document plans and translations, counted directly in the store file."""
    with closing(sqlite3.connect(str(store.store_path()))) as conn:
        documents = conn.execute("SELECT COUNT(*) FROM document_plans").fetchone()[0]
        translations = conn.execute("SELECT COUNT(*) FROM segment_translations").fetchone()[0]
    return {"documents": documents, "translations": translations}


@pytest.fixture
def runtime_root(tmp_path):
    original_data = system_config.data_dir
    original_logs = system_config.logs_dir
    bind_runtime_roots(tmp_path / "data", tmp_path / "logs")
    yield tmp_path / "data"
    bind_runtime_roots(original_data, original_logs)


@pytest.fixture
def store(runtime_root):
    return TranslationStore()


def _document(doc_hash: str, doc_key: str = "paper:legacy:p1", **overrides) -> DocumentRecord:
    payload = {
        "doc_hash": doc_hash,
        "doc_key": doc_key,
        "kind": "paper",
        "title": "Título — 論文",
        "priority_class": 1,
        "plan_json": '{"profile": "document"}',
        "origin": "auto",
        "created_at": "2026-11-01T10:00:00+00:00",
        "last_seen_at": "2026-11-01T10:00:00+00:00",
        "last_viewed_at": "",
    }
    payload.update(overrides)
    return DocumentRecord(**payload)


def _translation(source_hash: str, language_key: str, text: str) -> SegmentTranslation:
    return SegmentTranslation(
        source_hash=source_hash,
        language_key=language_key,
        translation=text,
        translator_notes="note",
        submitter_role="translation_submitter",
        submitter_model="vendor/translator",
        validator_model="vendor/validator",
        rounds=2,
        created_at="2026-11-01T10:00:00+00:00",
    )


def test_store_path_resolves_lazily_under_data_root(store, runtime_root):
    assert store.store_path() == Path(system_config.data_dir) / TRANSLATIONS_DIR_NAME / STORE_FILE_NAME
    assert not store.store_path().exists()

    store.load_documents()

    assert store.store_path().is_file()
    assert store.store_path().parent == Path(runtime_root).resolve() / TRANSLATIONS_DIR_NAME


def test_documents_round_trip_and_doc_key_replacement(store):
    first = _document("d1")
    other = _document("d9", doc_key="proof:manual:run:p9", kind="proof", priority_class=0)
    store.register_document(first)
    store.register_document(other)

    assert _documents_by_key(store) == {
        "paper:legacy:p1": first,
        "proof:manual:run:p9": other,
    }

    replacement = _document("d2", origin="viewer")
    store.register_document(replacement)

    assert _documents_by_key(store)["paper:legacy:p1"] == replacement
    assert _keys_for_hash(store, "d1") == []
    assert {(record.doc_key, record.doc_hash) for record in store.load_documents()} == {
        ("paper:legacy:p1", "d2"),
        ("proof:manual:run:p9", "d9"),
    }
    # The plan that lost its only key is pruned.
    assert _row_counts(store)["documents"] == 2

    # Translate all registers older outputs with the "backlog" origin, which round-trips.
    backlog = _document("d3", doc_key="paper:old:p1", origin="backlog")
    store.register_document(backlog)
    assert _documents_by_key(store)["paper:old:p1"] == backlog


def test_several_doc_keys_share_one_plan(store):
    auto = _document("d1", doc_key="final_answer:s1", kind="final_answer")
    viewer = _document(
        "d1",
        doc_key="final:a1:p1",
        kind="final_answer",
        origin="viewer",
        plan_json='{"ignored": "a known hash keeps its plan"}',
        last_viewed_at="2026-11-01T10:05:00+00:00",
    )
    store.register_document(auto)
    store.register_document(viewer)

    keys = _keys_for_hash(store, "d1")
    assert [record.doc_key for record in keys] == ["final:a1:p1", "final_answer:s1"]
    assert all(record.plan_json == '{"profile": "document"}' for record in keys)
    documents = _documents_by_key(store)
    assert documents["final_answer:s1"].origin == "auto"
    assert documents["final:a1:p1"].origin == "viewer"
    assert documents["final:a1:p1"].last_viewed_at == "2026-11-01T10:05:00+00:00"
    assert _row_counts(store)["documents"] == 1

    # Moving one key to new content keeps the plan for the key that stays.
    store.register_document(_document("d2", doc_key="final:a1:p1", origin="viewer"))
    assert [record.doc_key for record in _keys_for_hash(store, "d1")] == ["final_answer:s1"]
    assert [record.doc_key for record in _keys_for_hash(store, "d2")] == ["final:a1:p1"]
    assert _row_counts(store)["documents"] == 2

    # Once the last key leaves, the old plan is pruned.
    store.register_document(_document("d2", doc_key="final_answer:s1"))
    assert _keys_for_hash(store, "d1") == []
    assert _row_counts(store)["documents"] == 1


def test_reregistering_a_document_keeps_created_at_and_never_rewinds_timestamps(store):
    store.register_document(
        _document("d1", last_seen_at="2026-11-01T12:00:00+00:00", last_viewed_at="2026-11-01T12:00:00+00:00")
    )
    store.register_document(
        _document(
            "d1",
            title="Updated",
            created_at="2026-11-02T00:00:00+00:00",
            last_seen_at="2026-11-01T11:00:00+00:00",
            last_viewed_at="",
        )
    )

    record = _documents_by_key(store).get("paper:legacy:p1")
    assert record is not None
    assert record.title == "Updated"
    assert record.created_at == "2026-11-01T10:00:00+00:00"
    assert record.last_seen_at == "2026-11-01T12:00:00+00:00"
    assert record.last_viewed_at == "2026-11-01T12:00:00+00:00"

    # A key that moves to new content starts fresh timestamps for that content.
    store.register_document(
        _document("d2", created_at="2026-11-03T00:00:00+00:00", last_seen_at="2026-11-03T00:00:00+00:00")
    )
    moved = _documents_by_key(store)["paper:legacy:p1"]
    assert (moved.doc_hash, moved.created_at, moved.last_viewed_at) == ("d2", "2026-11-03T00:00:00+00:00", "")


def test_mark_documents_viewed_updates_every_key_in_one_transaction(store, monkeypatch):
    for index in range(3):
        store.register_document(_document("d1", doc_key=f"paper:legacy:p{index}"))
    store.register_document(_document("d1", doc_key="paper:legacy:copy"))
    opened = []
    connection = store._connection

    def counting_connection():
        opened.append(1)
        return connection()

    now = "2026-11-01T12:00:00+00:00"
    with monkeypatch.context() as patch:
        patch.setattr(store, "_connection", counting_connection)
        store.mark_documents_viewed([f"paper:legacy:p{index}" for index in range(3)] + ["missing"], now)
        assert len(opened) == 1
        # Nothing to mark opens no connection.
        store.mark_documents_viewed([], now)
        assert len(opened) == 1

    documents = _documents_by_key(store)
    for index in range(3):
        record = documents[f"paper:legacy:p{index}"]
        assert (record.last_seen_at, record.last_viewed_at) == (now, now)
    # Only the listed keys change, and a missing key is ignored.
    assert documents["paper:legacy:copy"].last_viewed_at == ""
    assert "missing" not in documents


def test_translations_round_trip_and_subset_lookup(store):
    store.save_translation(_translation("s1", "es", "Hola [[M1]]"))
    store.save_translation(_translation("s2", "es", "Mundo"))
    store.save_translation(_translation("s1", "zh-Hans", "你好 [[M1]]"))
    store.save_translation(_translation("s1", "es", "Hola de nuevo [[M1]]"))

    assert store.get_translations(["s1", "s2", "s3"], "es") == {
        "s1": "Hola de nuevo [[M1]]",
        "s2": "Mundo",
    }
    assert store.get_translations(["s2"], "es") == {"s2": "Mundo"}
    assert store.get_translations(["s1"], "zh-Hans") == {"s1": "你好 [[M1]]"}
    assert store.get_translations(["s1"], "fr") == {}
    assert store.get_translations([], "es") == {}
    assert store.translated_hashes_by_language() == {"es": {"s1", "s2"}, "zh-Hans": {"s1"}}
    assert _row_counts(store)["translations"] == 3


def test_translation_lookup_handles_more_hashes_than_one_query_chunk(store, monkeypatch):
    monkeypatch.setattr(store_module, "_QUERY_CHUNK_SIZE", 3)
    for index in range(7):
        store.save_translation(_translation(f"s{index}", "es", f"t{index}"))

    hashes = [f"s{index}" for index in range(7)]
    assert store.get_translations(hashes, "es") == {f"s{index}": f"t{index}" for index in range(7)}


def _job_state(doc_hash: str, language_key: str = "es", **overrides) -> dict:
    state = {
        "version": 1,
        "doc_hash": doc_hash,
        "language_key": language_key,
        "finals": {"0": "Título [[M1]]", "1": "論文 — \"citado\"\n\n- punto"},
        "part": {
            "start": 2,
            "end": 3,
            "locked": True,
            "attempts": [
                {
                    "number": 1,
                    "action": "translate",
                    "units": {"2": "Uno", "3": "Dos"},
                    "edits": [{"occurrence": 0, "find": "Título", "replace": "Titular"}],
                    "keep_attempts": [],
                    "notes": "",
                    "status": "accepted",
                    "feedback": "",
                    "reasoning": "Faithful.",
                    "accepted_order": 1,
                }
            ],
            "keep_streak": 0,
            "truncations": {"translation_submitter": 0, "translation_validator": 0},
        },
    }
    state.update(overrides)
    return state


def test_job_state_round_trips_and_replaces(store):
    assert store.get_job_state("d1", "es") is None
    assert store.load_job_keys() == []

    state = _job_state("d1")
    store.save_job_state("d1", "es", state)
    assert store.get_job_state("d1", "es") == state
    assert store.get_job_state("d1", "fr") is None
    assert store.get_job_state("d2", "es") is None

    replacement = _job_state("d1", finals={"0": "Otro"}, part=None)
    store.save_job_state("d1", "es", replacement)
    assert store.get_job_state("d1", "es") == replacement
    assert store.load_job_keys() == [("d1", "es")]


def test_saved_job_state_is_a_snapshot(store):
    state = _job_state("d1")
    store.save_job_state("d1", "es", state)
    state["finals"]["9"] = "added later"

    loaded = store.get_job_state("d1", "es")
    assert "9" not in loaded["finals"]
    loaded["finals"]["8"] = "changed copy"
    assert "8" not in store.get_job_state("d1", "es")["finals"]


def test_save_job_state_rejects_non_dict_and_unserializable_states(store):
    with pytest.raises(TypeError):
        store.save_job_state("d1", "es", ["not", "a", "dict"])
    with pytest.raises(TypeError):
        store.save_job_state("d1", "es", {"finals": {object()}})
    assert store.get_job_state("d1", "es") is None
    assert store.load_job_keys() == []


def test_get_job_state_ignores_corrupt_or_non_object_rows(store):
    store.save_job_state("d1", "es", _job_state("d1"))
    store.save_job_state("d1", "fr", _job_state("d1", "fr"))
    conn = sqlite3.connect(str(store.store_path()))
    try:
        conn.execute("UPDATE translation_jobs SET state_json = '{not json' WHERE language_key = 'es'")
        conn.execute("UPDATE translation_jobs SET state_json = '[1, 2]' WHERE language_key = 'fr'")
        conn.commit()
    finally:
        conn.close()

    assert store.get_job_state("d1", "es") is None
    assert store.get_job_state("d1", "fr") is None
    # A corrupt row is replaced by the next save.
    store.save_job_state("d1", "es", _job_state("d1"))
    assert store.get_job_state("d1", "es") == _job_state("d1")


def test_load_job_keys_lists_every_state_sorted(store):
    store.save_job_state("d2", "zh-Hans", _job_state("d2", "zh-Hans"))
    store.save_job_state("d1", "fr", _job_state("d1", "fr"))
    store.save_job_state("d1", "es", _job_state("d1"))
    store.save_job_state("d1", "es", _job_state("d1", part=None))

    assert store.load_job_keys() == [("d1", "es"), ("d1", "fr"), ("d2", "zh-Hans")]


def test_pruned_plan_deletes_its_job_states(store):
    store.register_document(_document("d1"))
    store.register_document(_document("d9", doc_key="proof:manual:run:p9"))
    store.save_job_state("d1", "es", _job_state("d1"))
    store.save_job_state("d1", "fr", _job_state("d1", "fr"))
    store.save_job_state("d9", "es", _job_state("d9"))

    # The key moves to new content: the plan of d1 loses its only key.
    store.register_document(_document("d2"))

    assert store.get_job_state("d1", "es") is None
    assert store.get_job_state("d1", "fr") is None
    assert store.load_job_keys() == [("d9", "es")]
    assert store.get_job_state("d9", "es") == _job_state("d9")


def test_shared_plan_keeps_job_states_until_its_last_key_leaves(store):
    store.register_document(_document("d1", doc_key="final_answer:s1", kind="final_answer"))
    store.register_document(_document("d1", doc_key="final:a1:p1", kind="final_answer", origin="viewer"))
    store.save_job_state("d1", "es", _job_state("d1"))

    store.register_document(_document("d2", doc_key="final:a1:p1", origin="viewer"))
    assert store.get_job_state("d1", "es") == _job_state("d1")

    # Re-registering a key on the hash it already has never prunes.
    store.register_document(_document("d1", doc_key="final_answer:s1", kind="final_answer"))
    assert store.get_job_state("d1", "es") == _job_state("d1")

    store.register_document(_document("d2", doc_key="final_answer:s1"))
    assert store.get_job_state("d1", "es") is None
    assert store.load_job_keys() == []


def test_clear_all_drops_rows_but_keeps_schema(store):
    store.register_document(_document("d1"))
    store.save_translation(_translation("s1", "es", "Hola"))
    store.save_job_state("d1", "es", _job_state("d1"))

    store.clear_all()

    assert _row_counts(store) == {"documents": 0, "translations": 0}
    assert store.load_job_keys() == []
    assert store.get_job_state("d1", "es") is None
    assert store.load_documents() == []
    assert store.translated_hashes_by_language() == {}
    store.save_translation(_translation("s1", "es", "Hola"))
    store.save_job_state("d1", "es", _job_state("d1"))
    assert _row_counts(store)["translations"] == 1
    assert store.load_job_keys() == [("d1", "es")]
    conn = sqlite3.connect(str(store.store_path()))
    try:
        version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
    finally:
        conn.close()
    assert version == str(store_module.SCHEMA_VERSION)


def _table_names(path: Path) -> set:
    conn = sqlite3.connect(str(path))
    try:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        conn.close()


def test_schema_is_version_five_without_segment_sources(store):
    store.load_documents()
    assert store_module.SCHEMA_VERSION == 5
    tables = _table_names(store.store_path())
    assert tables == {
        "meta",
        "segment_translations",
        "document_plans",
        "document_keys",
        "translation_jobs",
        "translation_requests",
    }


def test_requests_are_saved_once_and_outlive_clear_all(store):
    store.save_requests([("paper:s:p1", "es"), ("paper:s:p1", "fr")], "2026-01-01T00:00:00+00:00")
    store.save_requests([("paper:s:p1", "es"), ("proof:s:p2", "es")], "2026-02-01T00:00:00+00:00")
    assert store.load_requests() == {"paper:s:p1": {"es", "fr"}, "proof:s:p2": {"es"}}
    store.save_translation(_translation("s1", "es", "Hola"))
    store.clear_all()
    assert store.translated_hashes_by_language() == {}
    assert store.load_requests() == {"paper:s:p1": {"es", "fr"}, "proof:s:p2": {"es"}}


def test_a_store_without_the_requests_table_gains_it_and_keeps_its_translations(store):
    store.save_translation(_translation("s1", "es", "Hola"))
    conn = sqlite3.connect(str(store.store_path()))
    try:
        conn.execute("DROP TABLE translation_requests")
        conn.commit()
    finally:
        conn.close()
    reopened = store_module.TranslationStore()
    assert reopened.load_requests() == {}
    assert reopened.translated_hashes_by_language() == {"es": {"s1"}}


@pytest.mark.parametrize(
    ("old_version", "legacy_table_sql"),
    [
        # The schema-1 documents table keyed documents by content hash only.
        ("1", "CREATE TABLE documents (doc_hash TEXT PRIMARY KEY, doc_key TEXT NOT NULL)"),
        # Schema 2 kept rolling rejection feedback per segment and language.
        (
            "2",
            "CREATE TABLE segment_feedback (source_hash TEXT NOT NULL, language_key TEXT NOT NULL, "
            "entries TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY (source_hash, language_key))",
        ),
        # Schema 3 also stored each unit's masked source text, which nothing read.
        (
            "3",
            "CREATE TABLE segment_sources (source_hash TEXT PRIMARY KEY, masked_text TEXT NOT NULL, "
            "placeholder_tokens TEXT NOT NULL, paragraph_count INTEGER NOT NULL, "
            "paragraph_markers TEXT NOT NULL, token_count INTEGER NOT NULL, created_at TEXT NOT NULL)",
        ),
    ],
)
def test_schema_version_mismatch_recreates_tables(store, old_version, legacy_table_sql):
    store.save_translation(_translation("s1", "es", "Hola"))
    store.register_document(_document("d1"))
    store.save_job_state("d1", "es", _job_state("d1"))
    path = store.store_path()
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (old_version,))
        conn.execute("DROP TABLE translation_jobs")
        conn.execute(legacy_table_sql)
        conn.commit()
    finally:
        conn.close()

    reopened = TranslationStore()
    assert reopened.load_documents() == []
    assert _row_counts(reopened) == {"documents": 0, "translations": 0}
    assert reopened.load_job_keys() == []
    conn = sqlite3.connect(str(path))
    try:
        version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
    finally:
        conn.close()
    assert version == str(store_module.SCHEMA_VERSION)
    tables = _table_names(path)
    assert "documents" not in tables
    assert "segment_feedback" not in tables
    assert "segment_sources" not in tables
    assert {"document_plans", "document_keys", "translation_jobs"} <= tables
    reopened.save_job_state("d1", "es", _job_state("d1"))
    assert reopened.get_job_state("d1", "es") == _job_state("d1")


def test_a_schema_four_store_is_recreated_without_unused_plan_columns(store):
    store.register_document(_document("d1"))
    path = store.store_path()
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("UPDATE meta SET value = '4' WHERE key = 'schema_version'")
        conn.execute("DROP TABLE document_plans")
        conn.execute(
            "CREATE TABLE document_plans (doc_hash TEXT PRIMARY KEY, profile TEXT NOT NULL, "
            "plan_json TEXT NOT NULL, segment_hashes TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        conn.commit()
    finally:
        conn.close()

    reopened = TranslationStore()
    reopened.register_document(_document("d1"))
    assert [record.doc_key for record in reopened.load_documents()] == ["paper:legacy:p1"]
    conn = sqlite3.connect(str(path))
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(document_plans)")}
    finally:
        conn.close()
    assert columns == {"doc_hash", "plan_json", "created_at"}


def test_deleted_store_file_is_recreated(store):
    store.save_translation(_translation("s1", "es", "Hola"))
    store.save_job_state("d1", "es", _job_state("d1"))
    store.store_path().unlink()

    assert store.load_documents() == []
    assert _row_counts(store) == {"documents": 0, "translations": 0}
    assert store.load_job_keys() == []
    store.save_translation(_translation("s1", "es", "Hola"))
    assert store.get_translations(["s1"], "es") == {"s1": "Hola"}
    store.save_job_state("d1", "es", _job_state("d1"))
    assert store.get_job_state("d1", "es") == _job_state("d1")


def test_store_path_follows_runtime_root_rebinding(store, tmp_path):
    store.save_translation(_translation("s1", "es", "Primero"))
    first_path = store.store_path()

    bind_runtime_roots(tmp_path / "second-data", tmp_path / "second-logs")

    second_path = store.store_path()
    assert second_path != first_path
    assert second_path == Path(system_config.data_dir) / TRANSLATIONS_DIR_NAME / STORE_FILE_NAME
    assert store.get_translations(["s1"], "es") == {}
    store.save_translation(_translation("s1", "es", "Segundo"))
    assert second_path.is_file()
    assert store.get_translations(["s1"], "es") == {"s1": "Segundo"}
    store.save_job_state("d1", "es", _job_state("d1"))
    assert store.load_job_keys() == [("d1", "es")]

    conn = sqlite3.connect(str(first_path))
    try:
        rows = conn.execute("SELECT translation FROM segment_translations").fetchall()
    finally:
        conn.close()
    assert rows == [("Primero",)]


def test_module_singleton_is_a_translation_store():
    assert isinstance(store_module.translation_store, TranslationStore)
