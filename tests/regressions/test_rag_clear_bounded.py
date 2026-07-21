import pytest

from backend.aggregator.core.rag_manager import CHROMA_DELETE_BATCH_SIZE, RAGManager


class _FakeCollection:
    def __init__(self, count: int) -> None:
        self.ids = [f"chunk-{index}" for index in range(count)]
        self.get_calls = []

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        limit = kwargs["limit"]
        return {"ids": self.ids[:limit]}

    def delete(self, *, ids):
        deleted = set(ids)
        self.ids = [item for item in self.ids if item not in deleted]


@pytest.mark.asyncio
async def test_source_delete_fetches_ids_in_bounded_batches(monkeypatch):
    manager = RAGManager()
    collection = _FakeCollection(CHROMA_DELETE_BATCH_SIZE + 1)

    async def direct_native(_name, func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(manager, "_ensure_initialized_locked", lambda: None)
    monkeypatch.setattr(manager, "_await_native_worker", direct_native)
    manager.collections[256] = collection
    deleted = await manager._delete_matching_ids(
        256,
        "source delete",
        where={"source_file": "source.txt"},
    )

    assert deleted == CHROMA_DELETE_BATCH_SIZE + 1
    assert collection.ids == []
    assert len(collection.get_calls) == 3
    assert all(
        call == {
            "limit": CHROMA_DELETE_BATCH_SIZE,
            "include": [],
            "where": {"source_file": "source.txt"},
        }
        for call in collection.get_calls
    )
