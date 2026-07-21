import importlib
from pathlib import Path

import pytest

from backend.shared.config import bind_runtime_roots, system_config


@pytest.fixture
def restore_runtime_roots():
    original_data = system_config.data_dir
    original_logs = system_config.logs_dir
    yield
    bind_runtime_roots(original_data, original_logs)


def test_bind_runtime_roots_rederives_complete_mutable_graph(tmp_path, restore_runtime_roots):
    data_root = tmp_path / "instance-data"
    logs_root = tmp_path / "instance-logs"

    before = system_config.runtime_root_generation
    identity = bind_runtime_roots(data_root, logs_root)

    assert identity == (
        str(data_root.resolve()),
        str(logs_root.resolve()),
        before + 1,
    )
    expected = {
        system_config.user_uploads_dir: data_root / "user_uploads",
        system_config.chroma_db_dir: data_root / "chroma_db",
        system_config.shared_training_file: data_root / "rag_shared_training.txt",
        system_config.compiler_outline_file: data_root / "compiler_outline.txt",
        system_config.compiler_paper_file: data_root / "compiler_paper.txt",
        system_config.auto_sessions_base_dir: data_root / "auto_sessions",
        system_config.lean4_workspace_dir: data_root / "lean4_workspace",
    }
    for actual, wanted in expected.items():
        assert Path(actual) == wanted.resolve()
        system_config.assert_path_in_data_root(actual)


def test_active_root_validator_rejects_stale_paths(tmp_path, restore_runtime_roots):
    bind_runtime_roots(tmp_path / "active")

    with pytest.raises(RuntimeError, match="Stale mutable path"):
        system_config.assert_path_in_data_root(tmp_path / "old" / "state.json")


def test_shared_training_rejects_stale_scoped_override(tmp_path, restore_runtime_roots):
    from backend.aggregator.memory.shared_training import SharedTrainingMemory

    bind_runtime_roots(tmp_path / "old")
    memory = SharedTrainingMemory()
    memory.file_path = tmp_path / "old" / "brainstorms" / "topic.txt"

    bind_runtime_roots(tmp_path / "new")
    with pytest.raises(RuntimeError, match="Stale shared-training override"):
        _ = memory.file_path


@pytest.mark.asyncio
async def test_manual_and_compiler_stores_follow_rebind(tmp_path, restore_runtime_roots):
    from backend.aggregator.core.coordinator import coordinator
    from backend.aggregator.memory.event_log import event_log
    from backend.aggregator.memory.shared_training import (
        load_manual_main_submitter_config,
        save_manual_main_submitter_config,
        shared_training_memory,
    )
    from backend.compiler.memory.outline_memory import outline_memory
    from backend.compiler.memory.paper_memory import paper_memory

    root_one = tmp_path / "one"
    root_two = tmp_path / "two"
    bind_runtime_roots(root_one)
    await shared_training_memory.initialize()
    await event_log.initialize()
    await paper_memory.initialize()
    await outline_memory.initialize()
    await save_manual_main_submitter_config({"model_id": "first"})
    assert coordinator.stats_file_path == root_one.resolve() / "aggregator_stats.json"

    bind_runtime_roots(root_two)
    await shared_training_memory.initialize()
    await event_log.initialize()
    await paper_memory.initialize()
    await outline_memory.initialize()
    await save_manual_main_submitter_config({"model_id": "second"})

    assert shared_training_memory.file_path == root_two.resolve() / "rag_shared_training.txt"
    assert event_log.file_path == root_two.resolve() / "aggregator_event_log.txt"
    assert paper_memory.file_path == root_two.resolve() / "compiler_paper.txt"
    assert outline_memory.file_path == root_two.resolve() / "compiler_outline.txt"
    assert coordinator.stats_file_path == root_two.resolve() / "aggregator_stats.json"
    assert await load_manual_main_submitter_config() == {"model_id": "second"}
    assert (root_one / "manual_main_submitter_config.json").exists()


def test_rag_manager_import_and_construction_are_side_effect_free(
    tmp_path, restore_runtime_roots
):
    bind_runtime_roots(tmp_path / "rag-data")
    module = importlib.import_module("backend.aggregator.core.rag_manager")
    manager = module.RAGManager()

    assert manager.is_initialized is False
    assert not Path(system_config.chroma_db_dir).exists()


@pytest.mark.asyncio
async def test_rag_manager_reopens_under_active_root(tmp_path, restore_runtime_roots):
    from backend.aggregator.core.rag_manager import RAGManager

    manager = RAGManager()
    first_root = tmp_path / "rag-one"
    second_root = tmp_path / "rag-two"
    try:
        bind_runtime_roots(first_root)
        await manager.ensure_initialized()
        first_identity = manager._root_identity
        assert (first_root / "chroma_db").exists()

        bind_runtime_roots(second_root)
        await manager.ensure_initialized()
        assert manager._root_identity != first_identity
        assert manager._root_identity == system_config.runtime_root_identity()
        assert (second_root / "chroma_db").exists()
    finally:
        await manager.reset()


def test_default_proof_stores_follow_rebind(tmp_path, restore_runtime_roots):
    from backend.autonomous.memory.proof_database import (
        manual_proof_database,
        proof_database,
    )

    bind_runtime_roots(tmp_path / "proof-data")

    assert proof_database._get_index_path().parent == (
        tmp_path / "proof-data" / "proofs"
    ).resolve()
    assert manual_proof_database._get_index_path().parent == (
        tmp_path / "proof-data" / "manual_proofs"
    ).resolve()
