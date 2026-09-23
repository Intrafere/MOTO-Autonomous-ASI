import asyncio
from datetime import datetime, timezone

import pytest

from backend.autonomous.memory.proof_competition_benchmarks import (
    BenchmarkRecord, BenchmarkRoute, ProofCompetitionBenchmarkStore, list_benchmark_reports,
)


def record(competitor="primary", **changes):
    values = dict(execution_id="execution", source_type="brainstorm", source_id="topic",
                  candidate_id="candidate", competitor_id=competitor,
                  competitor_order=0 if competitor == "primary" else 1,
                  route_revision="v1", configured_route=BenchmarkRoute(provider="test", model_id=competitor))
    values.update(changes)
    return BenchmarkRecord(**values)


@pytest.mark.asyncio
async def test_shared_starts_and_incomplete_not_losses(tmp_path):
    (tmp_path / "session").mkdir()
    store = ProofCompetitionBenchmarkStore("session", tmp_path)
    await store.ensure_report("report", run_id="run")
    now = datetime.now(timezone.utc)
    await store.record("report", record(started_at=now, outcome="exhausted", attempts_consumed=5))
    await store.record("report", record("secondary", started_at=now, outcome="verified", attempts_consumed=2, proof_id="proof"))
    await store.record("report", record(candidate_id="queued"))
    await store.record("report", record("secondary", candidate_id="queued", outcome="unavailable"))
    await store.record("report", record(candidate_id="interrupted", started_at=now, outcome="interrupted"))
    await store.record("report", record("secondary", candidate_id="interrupted", started_at=now, outcome="exhausted"))
    report = await store.get_report("report")
    assert report.attempted == 4
    assert report.unavailable == 1
    assert report.secondary_rescues == 1
    pair = report.pairwise[0]
    assert pair.shared_problem_count == 2
    assert pair.conditional_rescue_count == 1
    assert pair.conditional_rescue_opportunities == 1
    assert pair.conditional_rescue_rate == 1.0
    primary = next(score for score in (pair.left, pair.right) if score.competitor_id == "primary")
    assert primary.incomplete == 1 and primary.completed_losses == 1
    assert len(pair.shared_records) == 4
    assert all(row.input_tokens is None for row in report.records)

    # An interrupted shared opportunity must not dilute a model's solve rate.
    await store.record("report", record(candidate_id="another", started_at=now, outcome="exhausted"))
    await store.record("report", record("secondary", candidate_id="another", started_at=now, outcome="interrupted"))
    report = await store.get_report("report")
    secondary = report.pairwise[0].right
    assert secondary.incomplete == 1
    assert secondary.solve_rate == 0.5


@pytest.mark.asyncio
async def test_idempotency_concurrency_revision_and_persistence(tmp_path):
    (tmp_path / "session").mkdir()
    store = ProofCompetitionBenchmarkStore("session", tmp_path)
    await store.ensure_report("report", run_id="run")
    row = record()
    results = await asyncio.gather(*(store.record("report", row) for _ in range(10)))
    assert sum(results) == 1
    now = datetime.now(timezone.utc)
    started = row.model_copy(update=dict(revision=1, started_at=now, outcome="running"))
    assert await store.record("report", started)
    assert not await store.record("report", row)
    with pytest.raises(ValueError):
        await store.record("report", row.model_copy(update=dict(revision=2)))
    with pytest.raises(ValueError):
        await store.ensure_report("report", run_id="other")
    reopened = ProofCompetitionBenchmarkStore("session", tmp_path)
    assert (await reopened.get_report("report")).attempted == 1
    page = await list_benchmark_reports(base_dir=tmp_path, limit=1)
    assert page.total == 1 and page.reports[0].record_count == 1


@pytest.mark.asyncio
async def test_empty_comparison_and_identity_isolation(tmp_path):
    (tmp_path / "session").mkdir()
    store = ProofCompetitionBenchmarkStore("session", tmp_path)
    await store.ensure_report("disabled", run_id="run", enabled=False)
    assert not (tmp_path / "session" / "proof_competition.sqlite").exists()
    await store.ensure_report("report", run_id="run")
    await store.record("report", record())
    await store.record("report", record("secondary", outcome="skipped"))
    report = await store.get_report("report")
    assert report.pairwise[0].shared_problem_count == 0
    assert report.pairwise[0].left.solve_rate is None
    with pytest.raises(ValueError):
        ProofCompetitionBenchmarkStore("../escape", tmp_path)
    with pytest.raises(FileNotFoundError):
        await ProofCompetitionBenchmarkStore("missing", tmp_path).ensure_report("r", run_id="r")
    with pytest.raises(ValueError):
        await store.record("report", record(outcome="verified"))
    await store.record("report", record(route_revision="v2"))
    await store.record("report", record(execution_id="another-check"))
    assert (await store.get_report("report")).record_count == 4


@pytest.mark.asyncio
async def test_routes_are_typed_and_pagination_is_bounded(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from backend.api.routes import proof_competition_benchmarks as routes
    from backend.shared.config import system_config
    monkeypatch.setattr(system_config, "auto_sessions_base_dir", tmp_path)
    (tmp_path / "session").mkdir()
    store = ProofCompetitionBenchmarkStore("session", tmp_path)
    await store.ensure_report("report", run_id="run")
    app = FastAPI()
    app.include_router(routes.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get('/api/proof-competition/benchmarks')
        assert response.status_code == 200
        assert response.json()['total'] == 1
        assert response.headers['cache-control'] == 'no-store'
        assert (await client.get('/api/proof-competition/benchmarks?limit=101')).status_code == 422
        assert (await client.get('/api/proof-competition/benchmarks?session_id=..')).status_code == 400
        assert (await client.get('/api/proof-competition/benchmarks/session/missing')).status_code == 404
        assert (await client.get('/api/proof-competition/benchmarks/session/report')).json()['record_count'] == 0
