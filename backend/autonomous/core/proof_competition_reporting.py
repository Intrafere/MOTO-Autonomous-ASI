"""Adapt safe competition snapshots to durable session benchmark records."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from backend.autonomous.memory.proof_competition_benchmarks import (
    BenchmarkRecord, BenchmarkRoute, ProofCompetitionBenchmarkStore,
)


def observed_route(metadata):
    """Allowlist successful-call routing facts; never infer an unobserved model."""
    provider, model = metadata.get('effective_provider'), metadata.get('effective_model')
    if not isinstance(provider, str) or not isinstance(model, str) or not provider or not model:
        return None
    boosted = metadata.get('boosted') is True
    return BenchmarkRoute(
        provider=provider[:100], model_id=model[:250], boosted=boosted,
        supercharge=metadata.get('supercharged') is True,
        fallback=not boosted and bool(metadata.get('configured_provider'))
            and provider != metadata['configured_provider'],
        host=str(metadata['openrouter_provider'])[:100] if metadata.get('openrouter_provider') else None,
        reasoning_effort=str(metadata['openrouter_reasoning_effort'])[:40]
            if metadata.get('openrouter_reasoning_effort') else None,
    ).model_dump(mode='json')


class CompetitionReporter:
    """Execution-local adapter; store snapshots supply recovery idempotency.

    Callers must supply a source-owned session and stable run/report identity.
    The adapter never sees proof code, prompts, or private feedback.
    """

    def __init__(self, *, session_id, run_id, primary_route, secondaries, context_revision=None):
        self.store = ProofCompetitionBenchmarkStore(session_id)
        self.report_id = str(run_id)
        self.run_id = str(run_id)
        self.routes = [primary_route, *secondaries]
        self.context_revision = context_revision
        self.rows = {}
        self.initialized = False
        self.lock = asyncio.Lock()

    async def __call__(self, event):
        async with self.lock:
            await self._record_event(event)

    async def _initialize(self):
        if not self.initialized:
            await self.store.ensure_report(self.report_id, run_id=self.run_id)
            report = await self.store.get_report(self.report_id)
            self.rows = {row.record_key: row for row in report.records}
            self.initialized = True

    async def _record_event(self, event):
        await self._initialize()
        index = int(event['competitor_index'])
        route = self.routes[index]
        route = route.model_dump(mode='json') if hasattr(route, 'model_dump') else dict(route)
        now = datetime.now(timezone.utc)
        status = event['status']
        outcome = {'success': 'verified', 'blocked': 'interrupted', 'eligible': 'queued'}.get(status, status)
        candidate_id = str(event.get('candidate_fingerprint') or event['theorem_id'])
        row = BenchmarkRecord(
            execution_id=str(event.get('execution_id') or event.get('run_id') or self.run_id),
            source_type=event['source_type'], source_id=event['source_id'],
            round_index=max(1, int(event.get('round') or 1)),
            candidate_id=candidate_id, competitor_id=f'competitor_{index}',
            competitor_order=index, route_revision=event['route_revision'],
            configured_route=BenchmarkRoute(
                provider=route.get('provider', 'unknown'), model_id=route.get('model_id', 'unknown'),
                host=route.get('openrouter_provider'), reasoning_effort=route.get('openrouter_reasoning_effort'),
                context_window=route.get('context_window'), max_output_tokens=route.get('max_output_tokens'),
                supercharge=bool(route.get('supercharge_enabled')),
            ),
            effective_routes=event.get('effective_routes') or [],
            outcome=outcome, attempts_consumed=int(event.get('attempts_consumed') or 0),
            interruption_kind=event.get('error_type'), context_revision=self.context_revision,
        )
        previous = self.rows.get(row.record_key)
        if previous:
            row.started_at = previous.started_at
            row.revision = previous.revision + 1
            row.proof_id = previous.proof_id
            row.effective_routes = (previous.effective_routes + [route for route in row.effective_routes
                                    if route not in previous.effective_routes])[:100]
            if previous.outcome in {'verified', 'exhausted', 'skipped'}:
                return
        if row.started_at is None and event.get('attempt_started') is True:
            row.started_at = now
        # Recovery re-enters eligibility with durable attempts already present.
        # Such work is resumed, not newly queued; queued records cannot own starts.
        if outcome == 'queued' and row.started_at is not None:
            outcome = row.outcome = 'running'
        if outcome not in {'queued', 'running'}:
            row.finished_at = now
            if row.started_at:
                row.elapsed_seconds = max(0, (now - row.started_at).total_seconds())
        await self.store.record(self.report_id, row)
        self.rows[row.record_key] = row

    async def registered(self, *, winner, proof_id):
        """Attach registration only to the exact successful execution occurrence."""
        async with self.lock:
            await self._initialize()
            for key, existing in list(self.rows.items()):
                identity = (existing.execution_id, existing.source_type, existing.source_id,
                            existing.round_index, existing.candidate_id,
                            existing.competitor_order, existing.route_revision)
                expected = (str(winner.get('execution_id') or winner.get('run_id') or self.run_id),
                            winner['source_type'], winner['source_id'],
                            max(1, int(winner.get('round') or 1)),
                            winner['candidate_fingerprint'], winner['competitor_index'],
                            winner['route_revision'])
                if identity != expected or existing.outcome != 'verified':
                    continue
                row = existing.model_copy(update={'proof_id': proof_id, 'revision': existing.revision + 1})
                await self.store.record(self.report_id, row)
                self.rows[key] = row
