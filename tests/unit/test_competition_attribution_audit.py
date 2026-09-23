import pytest

from backend.autonomous.core.proof_competition_reporting import CompetitionReporter, observed_route
from backend.autonomous.memory.proof_competition_benchmarks import ProofCompetitionBenchmarkStore


@pytest.mark.asyncio
async def test_registration_is_exact_occurrence_and_routes_survive_replay(tmp_path):
    (tmp_path / 'session').mkdir()
    route = {'provider': 'openrouter', 'model_id': 'configured'}
    reporter = CompetitionReporter(session_id='session', run_id='run', primary_route=route, secondaries=[route])
    reporter.store = ProofCompetitionBenchmarkStore('session', base_dir=tmp_path)
    event = dict(execution_id='execution', source_type='paper', source_id='paper', round=1,
                 theorem_id='candidate', candidate_fingerprint='candidate', competitor_index=1,
                 route_revision='revision', status='running', attempt_started=True)
    observed = observed_route(dict(effective_provider='lm_studio', effective_model='local',
                                  configured_provider='openrouter', boosted=False, secret='never'))
    variants = [event, {**event, 'execution_id': 'other'}, {**event, 'source_id': 'other'},
                {**event, 'round': 2}, {**event, 'competitor_index': 0},
                {**event, 'route_revision': 'other'}]
    for item in variants:
        await reporter({**item, 'effective_routes': [observed]})
        await reporter({**item, 'status': 'success'})
    await reporter.registered(winner=event, proof_id='winner')
    report = await reporter.store.get_report('run')
    assert sum(row.proof_id == 'winner' for row in report.records) == 1
    winner = next(row for row in report.records if row.proof_id)
    assert winner.execution_id == 'execution' and winner.round_index == 1
    assert winner.effective_routes[0].model_id == 'local'
    assert winner.effective_routes[0].fallback
    assert 'secret' not in observed


@pytest.mark.asyncio
async def test_formalizer_forwards_successful_completion_route(monkeypatch):
    from backend.autonomous.agents.proof_formalization_agent import ProofFormalizationAgent
    from backend.shared.api_client_manager import api_client_manager
    metadata = dict(effective_provider='openrouter', effective_model='boost/model', boosted=True)
    async def complete(**kwargs):
        return {api_client_manager.CALL_METADATA_KEY: metadata}
    monkeypatch.setattr(api_client_manager, 'generate_completion', complete)
    agent = ProofFormalizationAgent(model_id='configured', context_window=1000, max_output_tokens=100, role_id='test')
    captured = []
    async def capture(value):
        captured.append(value)
    agent.completion_route_callback = capture
    await agent._generate_with_feedback_retries(prompt='test', retained_attempts=[],
                                                build_prompt=lambda _: 'test', max_input_tokens=900)
    assert captured == [metadata]


def test_boost_is_observed_not_configured_and_missing_metadata_stays_unknown():
    assert observed_route({}) is None
    route = observed_route(dict(effective_provider='openrouter', effective_model='boost/model',
                                configured_provider='openai_codex_oauth', boosted=True))
    assert route['boosted'] and not route['fallback']
    assert route['model_id'] == 'boost/model'
