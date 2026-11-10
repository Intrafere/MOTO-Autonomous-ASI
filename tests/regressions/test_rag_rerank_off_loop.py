"""RAG Stage C (MMR reranking) must match the MMR definition and stay off the event loop.

Stage C used to run on the FastAPI event loop and recompute every
candidate-to-selected cosine pair at every step. With 240 candidates of 768-dim
embeddings that blocked the loop for 30 seconds or more per retrieval, long
enough for provider websocket keepalives to time out.
"""
import asyncio
import random
import threading
import time

import numpy as np
import pytest

from backend.aggregator.core.rag_manager import RAGManager
from backend.shared.config import rag_config
from backend.shared.models import DocumentChunk


def _plain_cosine(vec1, vec2):
    if not vec1 or not vec2:
        return 0.0
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def _exhaustive_mmr(candidates):
    """Reference MMR: recompute every candidate-to-selected similarity at every step."""
    selected = []
    remaining = list(candidates)
    while remaining and len(selected) < rag_config.hybrid_recall_top_k:
        best_idx = 0
        if selected:
            best_score = -float("inf")
            for idx, (chunk, relevance) in enumerate(remaining):
                diversity = 1.0 - min(
                    _plain_cosine(chunk.embedding, chosen.embedding) for chosen in selected
                )
                score = rag_config.mmr_lambda * relevance + (1 - rag_config.mmr_lambda) * diversity
                if score > best_score:
                    best_score = score
                    best_idx = idx
        selected.append(remaining.pop(best_idx)[0])
    final = []
    for chunk in selected:
        if not any(
            _plain_cosine(chunk.embedding, kept.embedding) > rag_config.similarity_threshold
            for kept in final
        ):
            final.append(chunk)
    return final


def _candidates(rng, count, dim):
    base = [rng.gauss(0, 1) for _ in range(dim)]
    chunks = []
    for index in range(count):
        roll = rng.random()
        if roll < 0.05:
            embedding = None
        elif roll < 0.08:
            embedding = []
        elif roll < 0.11:
            embedding = [0.0] * dim
        elif roll < 0.25 and chunks and chunks[-1].embedding:
            embedding = list(chunks[-1].embedding)
        elif roll < 0.40:
            embedding = [value + rng.gauss(0, 0.05) for value in base]
        else:
            embedding = [rng.gauss(0, 1) for _ in range(dim)]
        chunks.append(DocumentChunk(
            chunk_id=f"chunk-{index}",
            text=f"text {index}",
            source_file="source.txt",
            position=index,
            chunk_size=512,
            embedding=embedding,
        ))
    relevances = sorted(
        (rng.choice([1.0, 0.5, rng.random()]) for _ in range(count)),
        reverse=True,
    )
    return list(zip(chunks, relevances))


@pytest.mark.parametrize("top_k", [1, 3, 25, 120])
def test_stage_c_matches_exhaustive_mmr(monkeypatch, top_k):
    monkeypatch.setattr(rag_config, "hybrid_recall_top_k", top_k)
    manager = RAGManager()
    rng = random.Random(top_k)
    for count in (0, 1, 2, 9, 40):
        for _ in range(6):
            candidates = _candidates(rng, count, 16)
            expected = [chunk.chunk_id for chunk in _exhaustive_mmr(candidates)]
            actual = [chunk.chunk_id for chunk in manager._rerank_and_diversify(candidates)]
            assert actual == expected


@pytest.mark.asyncio
async def test_retrieve_runs_stage_c_off_the_event_loop(monkeypatch):
    manager = RAGManager()
    chunk = DocumentChunk(
        chunk_id="chunk-0",
        text="alpha beta gamma",
        source_file="source.txt",
        position=0,
        chunk_size=512,
        embedding=[1.0, 0.0],
    )
    loop_thread = threading.get_ident()
    stage_c_threads = []

    async def fake_hybrid_recall(*_args, **_kwargs):
        return [(chunk, 1.0)]

    def slow_rerank(candidates):
        stage_c_threads.append(threading.get_ident())
        time.sleep(0.5)
        return [candidates[0][0]]

    monkeypatch.setattr(manager, "_hybrid_recall", fake_hybrid_recall)
    monkeypatch.setattr(manager, "_rerank_and_diversify", slow_rerank)

    ticks = 0
    stop = asyncio.Event()

    async def heartbeat():
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    try:
        pack = await manager.retrieve("alpha beta gamma delta", chunk_size=512, max_tokens=1000)
    finally:
        stop.set()
        await beat

    assert stage_c_threads and stage_c_threads[0] != loop_thread
    assert ticks >= 10, "the event loop was blocked while Stage C ran"
    assert [item["text"] for item in pack.evidence] == ["alpha beta gamma"]
