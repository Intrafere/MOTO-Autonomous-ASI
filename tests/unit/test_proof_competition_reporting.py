import pytest

from backend.autonomous.core.proof_competition_reporting import CompetitionReporter
from backend.autonomous.memory.proof_competition_benchmarks import ProofCompetitionBenchmarkStore


@pytest.mark.asyncio
async def test_reporter_persists_shared_attempts_and_registration(tmp_path):
    (tmp_path / 'session').mkdir()
    route = {'provider': 'openrouter', 'model_id': 'model', 'context_window': 1000, 'max_output_tokens': 100}
    reporter = CompetitionReporter(session_id='session', run_id='run', primary_route=route, secondaries=[route])
    reporter.store = ProofCompetitionBenchmarkStore('session', base_dir=tmp_path)
    event = {'run_id': 'run', 'source_type': 'brainstorm', 'source_id': 'topic', 'round': 1,
             'theorem_id': 'candidate', 'route_revision': 'revision', 'competitor_index': 0,
             'status': 'running', 'attempts_consumed': 0, 'attempt_started': True}
    await reporter(event)
    await reporter({**event, 'status': 'eligible', 'attempts_consumed': 1})
    resumed = (await reporter.store.get_report('run')).records[0]
    assert resumed.outcome == 'running'
    assert resumed.started_at is not None
    await reporter({**event, 'status': 'exhausted', 'attempts_consumed': 5})
    await reporter({**event, 'competitor_index': 1})
    await reporter({**event, 'competitor_index': 1, 'status': 'success', 'attempts_consumed': 1})
    await reporter.registered(winner={**event, 'candidate_fingerprint': 'candidate', 'competitor_index': 1}, proof_id='proof')
    report = await reporter.store.get_report('run')
    assert report.pairwise[0].shared_problem_count == 1
    assert report.secondary_rescues == 1
    assert next(row for row in report.records if row.competitor_order == 1).proof_id == 'proof'
    assert all(row.input_tokens is None and row.effective_routes == [] for row in report.records)
    await reporter({**event, 'theorem_id': 'preflight', 'status': 'eligible', 'attempt_started': False})
    await reporter({**event, 'theorem_id': 'preflight', 'status': 'unavailable', 'attempt_started': False})
    preflight = next(row for row in (await reporter.store.get_report('run')).records if row.candidate_id == 'preflight')
    assert preflight.started_at is None
    await reporter({**event, 'status': 'running'})
    assert next(row for row in (await reporter.store.get_report('run')).records if row.competitor_order == 0).outcome == 'exhausted'
