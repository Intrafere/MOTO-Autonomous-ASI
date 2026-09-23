import React from 'react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';
import ProofCompetitionBenchmarkReport from './ProofCompetitionBenchmarkReport';

afterEach(cleanup);
const route = { provider: 'provider', model_id: 'model' };
const score = (id) => ({ competitor_key: id, competitor_id: id, configured_route: route, verified: 0, completed_losses: 0, incomplete: 0, solve_rate: null });
const report = {
  attempted: 0, verified: 0, secondary_rescues: 0, unavailable: 1, interrupted: 0,
  caveat: 'Conditional fallback; supporting context evolves. Unknown usage is not zero.',
  pairwise: [{ left: score('primary'), right: score('secondary'), shared_problem_count: 0, problem_keys: [] }],
  records: [{ execution_id: 'e', candidate_id: 'c', competitor_id: 'secondary', route_revision: 'v1',
    outcome: 'unavailable', source_type: 'paper', source_id: 'p', round_index: 1, attempts_consumed: 0,
    configured_route: route, effective_routes: [], support_ids: [], interruption_kind: 'provider_cooldown' }],
};

test('shows unknown usage, empty intersections and unavailable separately', async () => {
  const api = { list: vi.fn().mockResolvedValue({ reports: [{ session_id: 's', report_id: 'r', record_count: 1 }], total: 1 }), get: vi.fn().mockResolvedValue(report) };
  render(<ProofCompetitionBenchmarkReport api={api} />);
  expect(await screen.findByText('No shared attempted problems.')).toBeInTheDocument();
  expect(screen.queryByText(/0.0%/)).not.toBeInTheDocument();
  expect(screen.getByText('Tokens: input Unknown · output Unknown')).toBeInTheDocument();
  expect(screen.getByText(/not attributed to configured model/)).toBeInTheDocument();
  expect(screen.getByText('Error category: provider_cooldown')).toBeInTheDocument();
  expect(api.get).toHaveBeenCalledWith('s', 'r');
  fireEvent.mouseEnter(screen.getByRole('button', { name: 'Benchmark comparison limitations' }));
  expect(screen.getByText(/Context differences may be larger/)).toHaveTextContent('Model 1 with Model 3 or later');
});

test('does not display interrupted-only opportunities as zero percent', async () => {
  const interrupted = { ...report, pairwise: [{ ...report.pairwise[0], shared_problem_count: 1 }] };
  const api = { list: vi.fn().mockResolvedValue({ reports: [{ session_id: 's', report_id: 'r' }], total: 1 }), get: vi.fn().mockResolvedValue(interrupted) };
  render(<ProofCompetitionBenchmarkReport api={api} />);
  await screen.findByText('1 shared attempted problems');
  expect(screen.getAllByText(/No completed opportunities/)).toHaveLength(2);
  expect(screen.queryByText(/0.0%/)).not.toBeInTheDocument();
});

test('current mode requests session-filtered reports and exposes empty state', async () => {
  const api = { list: vi.fn().mockResolvedValue({ reports: [], total: 0 }), get: vi.fn() };
  render(<ProofCompetitionBenchmarkReport current api={api} />);
  expect(await screen.findByText(/No competition benchmark reports/)).toBeInTheDocument();
  expect(api.list).toHaveBeenCalledWith({ current: true, offset: 0, limit: 25 });
  expect(api.get).not.toHaveBeenCalled();
});

test('historical pagination requests the next bounded page', async () => {
  const api = { list: vi.fn().mockResolvedValue({ reports: [], total: 30 }), get: vi.fn() };
  render(<ProofCompetitionBenchmarkReport api={api} />);
  await screen.findByText('30 reports');
  fireEvent.click(screen.getByText('Next reports'));
  expect(api.list).toHaveBeenLastCalledWith({ current: false, offset: 25, limit: 25 });
});
