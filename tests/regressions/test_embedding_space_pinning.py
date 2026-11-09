"""A RAG index must hold vectors from exactly one embedding model.

LM Studio (nomic, 768 dimensions) used to fall back to OpenRouter
(text-embedding-3-small, 1536 dimensions) whenever one LM Studio embedding call
failed. Chroma then rejected every upsert into the existing index ("Collection
expecting embedding with dimension of 768, got 1536"), and even equal-length
vectors from two models would not be comparable.
"""
import uuid
from types import SimpleNamespace

import chromadb
import httpx
import pytest

from backend.aggregator.core import rag_manager as rag_module
from backend.aggregator.core.rag_manager import EMBEDDING_SPACE_METADATA_KEY, RAGManager
from backend.shared import api_client_manager as api_module
from backend.shared.api_client_manager import (
    PINNED_LM_STUDIO_EMBEDDING_ATTEMPTS,
    APIClientManager,
)
from backend.shared.config import rag_config, system_config
from backend.shared.models import DocumentChunk
from backend.shared.openrouter_client import OpenRouterClient

LM_STUDIO_SPACE = f"lm_studio:{rag_config.embedding_model}"
OPENROUTER_SPACE = f"openrouter:{OpenRouterClient.DEFAULT_EMBEDDING_MODEL}"


class _FakeOpenRouter:
    def __init__(self):
        self.calls = 0

    async def get_embeddings(self, texts, _model=None):
        self.calls += 1
        return [[0.5] * 1536 for _ in texts]


def _manager(monkeypatch, lm_studio_embeddings):
    monkeypatch.setattr(system_config, "generic_mode", False)
    monkeypatch.setattr(api_module.lm_studio_client, "get_embeddings", lm_studio_embeddings)
    manager = APIClientManager()
    manager._openrouter_client = _FakeOpenRouter()
    return manager


@pytest.mark.asyncio
async def test_unpinned_embeddings_prefer_lm_studio_and_report_the_space(monkeypatch):
    async def lm_studio(texts, _model=None, **_kwargs):
        return [[0.1] * 768 for _ in texts]

    manager = _manager(monkeypatch, lm_studio)

    vectors, space = await manager.get_embeddings_in_space(["a", "b"])

    assert space == LM_STUDIO_SPACE
    assert len(vectors) == 2 and len(vectors[0]) == 768
    assert manager._openrouter_client.calls == 0


@pytest.mark.asyncio
async def test_unpinned_embeddings_fall_back_to_openrouter_and_report_the_space(monkeypatch):
    async def lm_studio(_texts, _model=None, **_kwargs):
        raise httpx.ReadError("")

    manager = _manager(monkeypatch, lm_studio)

    vectors, space = await manager.get_embeddings_in_space(["a"])

    assert space == OPENROUTER_SPACE
    assert len(vectors[0]) == 1536


@pytest.mark.asyncio
async def test_lm_studio_index_retries_lm_studio_and_never_mixes_in_openrouter(monkeypatch):
    seen_attempts = []

    async def lm_studio(_texts, _model=None, **kwargs):
        seen_attempts.append(kwargs.get("max_attempts"))
        raise httpx.ReadError("")

    manager = _manager(monkeypatch, lm_studio)

    with pytest.raises(RuntimeError, match="not falling back to OpenRouter"):
        await manager.get_embeddings_in_space(["a"], embedding_space=LM_STUDIO_SPACE)

    assert seen_attempts == [PINNED_LM_STUDIO_EMBEDDING_ATTEMPTS]
    assert manager._openrouter_client.calls == 0


@pytest.mark.asyncio
async def test_openrouter_index_keeps_using_openrouter_when_lm_studio_returns(monkeypatch):
    async def lm_studio(_texts, _model=None, **_kwargs):
        raise AssertionError("an OpenRouter-built index must not receive LM Studio vectors")

    manager = _manager(monkeypatch, lm_studio)

    vectors, space = await manager.get_embeddings_in_space(["a"], embedding_space=OPENROUTER_SPACE)

    assert space == OPENROUTER_SPACE
    assert len(vectors[0]) == 1536


@pytest.mark.asyncio
async def test_pinned_lm_studio_backoff_retries_transport_errors(monkeypatch):
    client = api_module.lm_studio_client
    calls = {"count": 0}
    waits = []

    async def post(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": [{"index": 0, "embedding": [0.2] * 4}]},
        )

    async def record_wait(payload):
        waits.append(payload["retry_after_seconds"])

    monkeypatch.setattr(client, "client", SimpleNamespace(post=post))

    vectors = await client._get_embeddings_with_retry(
        ["a"],
        "nomic",
        retry_callback=record_wait,
        max_attempts=4,
    )

    assert vectors == [[0.2] * 4]
    assert waits == [client.RETRY_DELAY, client.RETRY_DELAY * 2]


class _FakeEmbeddingRoute:
    """Stands in for the API client manager: LM Studio first, then the pinned space."""

    def __init__(self):
        self.requested_spaces = []

    async def get_embeddings_in_space(self, texts, *, embedding_space=None, model=None):
        self.requested_spaces.append(embedding_space)
        return [[float(index + 1), 1.0, 0.0, 0.5] for index, _ in enumerate(texts)], (
            embedding_space or "lm_studio:test-model"
        )


def _chunk(index: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"source.txt::{index}::256",
        text=f"chunk text {index}",
        source_file="source.txt",
        position=index,
        chunk_size=256,
        metadata={"source_file": "source.txt"},
    )


@pytest.mark.asyncio
async def test_rag_index_pins_its_embedding_space_and_records_it_on_the_collection(monkeypatch):
    route = _FakeEmbeddingRoute()
    monkeypatch.setattr(rag_module, "api_client_manager", route)
    collection = chromadb.EphemeralClient().get_or_create_collection(
        name=f"chunks_256_{uuid.uuid4().hex}",
        metadata={"chunk_size": 256},
    )
    manager = RAGManager()
    manager.collections = {256: collection}

    async def opened():
        return None

    monkeypatch.setattr(manager, "ensure_initialized", opened)
    monkeypatch.setattr(manager, "_ensure_initialized_locked", lambda: None)

    await manager._add_chunks([_chunk(0)], 256)
    await manager._add_chunks([_chunk(1)], 256)
    await manager._vector_search(["chunk text"], 256)

    assert route.requested_spaces == [None, "lm_studio:test-model", "lm_studio:test-model"]
    assert collection.metadata[EMBEDDING_SPACE_METADATA_KEY] == "lm_studio:test-model"
    assert collection.metadata["chunk_size"] == 256

    reopened = RAGManager()
    reopened.collections = {256: collection}
    assert reopened._recorded_embedding_space_locked() == "lm_studio:test-model"

    manager._reset_memory_state()
    assert manager._embedding_space is None
