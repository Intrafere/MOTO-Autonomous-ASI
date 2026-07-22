from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from backend.aggregator.core.rag_manager import RAGManager
from backend.shared.rag_lock import RAGOperationLock
from backend.shared.workflow_start_guard import WorkflowStartGuard


def test_workflow_guard_commits_and_releases_one_owner(monkeypatch):
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        "backend.shared.workflow_start_guard.sleep_inhibitor.acquire",
        lambda owner: calls.append(("acquire", owner)),
    )
    monkeypatch.setattr(
        "backend.shared.workflow_start_guard.sleep_inhibitor.release",
        lambda owner: calls.append(("release", owner)),
    )

    guard = WorkflowStartGuard()
    first_lease = guard.commit("manual_aggregator")
    assert guard.commit("manual_aggregator") == first_lease
    assert guard.release("manual_aggregator") is False
    guard.release(first_lease)

    assert guard.active_owner is None
    assert calls == [
        ("acquire", first_lease),
        ("release", first_lease),
    ]


def test_workflow_guard_rejects_different_committed_owner(monkeypatch):
    monkeypatch.setattr(
        "backend.shared.workflow_start_guard.sleep_inhibitor.acquire",
        lambda _owner: None,
    )
    guard = WorkflowStartGuard()
    guard.commit("manual_compiler_proof_only")

    try:
        guard.commit("autonomous")
    except RuntimeError as exc:
        assert "manual_compiler_proof_only" in str(exc)
    else:
        raise AssertionError("Expected a conflicting owner to be rejected")


def test_stale_lease_cannot_release_new_generation(monkeypatch):
    monkeypatch.setattr(
        "backend.shared.workflow_start_guard.sleep_inhibitor.acquire",
        lambda _owner: None,
    )
    monkeypatch.setattr(
        "backend.shared.workflow_start_guard.sleep_inhibitor.release",
        lambda _owner: None,
    )
    guard = WorkflowStartGuard()
    stale = guard.commit("autonomous")
    assert guard.release(stale) is True
    current = guard.commit("autonomous")

    assert current.generation > stale.generation
    assert guard.release(stale) is False
    assert guard.active_lease == current


def test_inhibitor_setup_failure_does_not_fail_logical_commit(monkeypatch):
    def fail(_owner):
        raise RuntimeError("simulated worker setup failure")

    monkeypatch.setattr(
        "backend.shared.workflow_start_guard.sleep_inhibitor.acquire",
        fail,
    )
    guard = WorkflowStartGuard()

    lease = guard.commit("leanoj")

    assert guard.active_lease == lease
    assert guard.active_owner == "leanoj"


def test_inhibitor_release_failure_does_not_fail_logical_release(
    monkeypatch, caplog
):
    def fail(_lease):
        raise RuntimeError("simulated worker release failure")

    monkeypatch.setattr(
        "backend.shared.workflow_start_guard.sleep_inhibitor.acquire",
        lambda _lease: None,
    )
    monkeypatch.setattr(
        "backend.shared.workflow_start_guard.sleep_inhibitor.release",
        fail,
    )
    guard = WorkflowStartGuard()
    lease = guard.commit("manual_aggregator")

    assert guard.release(lease) is True
    assert guard.active_owner is None
    assert "Unable to release desktop sleep inhibition for manual_aggregator" in caplog.text


@pytest.mark.asyncio
async def test_workflow_guard_serializes_start_and_stop_lifecycle_work():
    guard = WorkflowStartGuard()
    start_entered = asyncio.Event()
    allow_start_to_finish = asyncio.Event()
    stop_entered = asyncio.Event()

    async def run_start() -> None:
        async with guard.reserve():
            start_entered.set()
            await allow_start_to_finish.wait()

    async def run_stop() -> None:
        async with guard.reserve():
            stop_entered.set()

    start_task = asyncio.create_task(run_start())
    await start_entered.wait()
    stop_task = asyncio.create_task(run_stop())
    await asyncio.sleep(0)

    assert not stop_entered.is_set()

    allow_start_to_finish.set()
    await asyncio.gather(start_task, stop_task)

    assert stop_entered.is_set()


def test_release_all_invalidates_same_owner_generation(monkeypatch):
    monkeypatch.setattr(
        "backend.shared.workflow_start_guard.sleep_inhibitor.acquire",
        lambda _lease: None,
    )
    monkeypatch.setattr(
        "backend.shared.workflow_start_guard.sleep_inhibitor.release_all",
        lambda: None,
    )
    guard = WorkflowStartGuard()
    stale = guard.commit("leanoj")

    guard.release_all()
    current = guard.commit("leanoj")

    assert current.generation > stale.generation
    assert guard.release(stale) is False
    assert guard.active_lease == current


@pytest.mark.asyncio
async def test_cancelled_chroma_call_finishes_worker_before_releasing_lock(monkeypatch):
    manager = RAGManager()
    rag_lock = RAGOperationLock()
    worker_entered = threading.Event()
    allow_worker_to_finish = threading.Event()
    second_operation_entered = asyncio.Event()

    monkeypatch.setattr(
        "backend.aggregator.core.rag_manager.rag_operation_lock",
        rag_lock,
    )
    monkeypatch.setattr(manager, "_ensure_initialized_locked", lambda: None)

    def blocking_native_call() -> str:
        worker_entered.set()
        allow_worker_to_finish.wait()
        return "done"

    manager.collections[256] = SimpleNamespace(run=blocking_native_call)
    first = asyncio.create_task(
        manager._run_chroma_call("first", 256, "run")
    )
    await asyncio.to_thread(worker_entered.wait)
    first.cancel()

    async def run_second() -> None:
        await rag_lock.acquire("second")
        try:
            second_operation_entered.set()
        finally:
            rag_lock.release()

    second = asyncio.create_task(run_second())
    await asyncio.sleep(0)
    assert not second_operation_entered.is_set()

    allow_worker_to_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    assert second_operation_entered.is_set()
    assert rag_lock._acquisition_count == 0


@pytest.mark.asyncio
async def test_cancelled_waiter_cannot_release_current_rag_owner():
    rag_lock = RAGOperationLock()
    owner_release = asyncio.Event()
    waiter_started = asyncio.Event()

    async def owner() -> None:
        async with rag_lock.operation("owner"):
            waiter_started.set()
            await owner_release.wait()

    async def waiter() -> None:
        async with rag_lock.operation("waiter"):
            raise AssertionError("cancelled waiter entered the critical section")

    owner_task = asyncio.create_task(owner())
    await waiter_started.wait()
    waiter_task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    waiter_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_task

    assert rag_lock._lock.locked()
    owner_release.set()
    await owner_task
    assert not rag_lock._lock.locked()


@pytest.mark.asyncio
async def test_chroma_method_is_resolved_after_lifecycle_initialization(monkeypatch):
    manager = RAGManager()
    stale = SimpleNamespace(query=lambda **_kwargs: "stale")
    current = SimpleNamespace(query=lambda **_kwargs: "current")
    manager.collections[256] = stale

    def initialize() -> None:
        manager.collections[256] = current

    async def direct_native(_name, func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(manager, "_ensure_initialized_locked", initialize)
    monkeypatch.setattr(manager, "_await_native_worker", direct_native)

    result = await manager._run_chroma_call("query", 256, "query")

    assert result == "current"


@pytest.mark.asyncio
async def test_failed_chroma_upsert_does_not_commit_memory(monkeypatch):
    manager = RAGManager()
    existing = SimpleNamespace(chunk_id="existing")
    incoming = SimpleNamespace(
        chunk_id="incoming",
        text="new text",
        metadata={"source_file": "source.txt"},
    )
    manager.chunks_by_size[256] = [existing]
    manager.collections[256] = SimpleNamespace(
        upsert=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("write failed"))
    )
    monkeypatch.setattr(manager, "_ensure_initialized_locked", lambda: None)

    async def direct_native(_name, func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(manager, "_await_native_worker", direct_native)
    monkeypatch.setattr(
        "backend.aggregator.core.rag_manager.api_client_manager.get_embeddings",
        lambda _texts: asyncio.sleep(0, result=[[0.0, 1.0]]),
    )

    with pytest.raises(RuntimeError, match="write failed"):
        await manager._add_chunks([incoming], 256)

    assert manager.chunks_by_size[256] == [existing]


@pytest.mark.asyncio
async def test_failed_chunk_cap_delete_preserves_chunks_and_embeddings(monkeypatch):
    manager = RAGManager()
    first = SimpleNamespace(
        chunk_id="first",
        is_permanent=False,
        embedding=[1.0, 0.0],
    )
    second = SimpleNamespace(
        chunk_id="second",
        is_permanent=False,
        embedding=[0.0, 1.0],
    )
    manager.chunks_by_size[256] = [first, second]
    sentinel = object()
    manager.bm25_index[256] = sentinel
    monkeypatch.setattr(
        "backend.aggregator.core.rag_manager.rag_config.submitter_chunk_intervals",
        [256],
    )
    monkeypatch.setattr(
        "backend.aggregator.core.rag_manager.rag_config.max_chunks_per_size",
        1,
    )

    async def fail_delete(*_args, **_kwargs):
        raise RuntimeError("delete failed")

    monkeypatch.setattr(manager, "_run_chroma_call", fail_delete)

    with pytest.raises(RuntimeError, match="delete failed"):
        await manager._enforce_chunk_cap()

    assert manager.chunks_by_size[256] == [first, second]
    assert first.embedding == [1.0, 0.0]
    assert second.embedding == [0.0, 1.0]
    assert manager.bm25_index[256] is sentinel
