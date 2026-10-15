import React from 'react';
import { render, screen } from '@testing-library/react';
import ProofRunStatusControls from './ProofRunStatusControls';

const baseRun = {
  proof_run_id: 'proof-run-1',
  lifecycle_generation: 4,
  run_mode: 'loop_with_pruning',
  source_type: 'paper',
  source_id: 'paper-1',
  current_round: 3,
  pruning_status: 'validating',
};

test('shows source, round, status, and independent pruning state', () => {
  render(<ProofRunStatusControls run={{ ...baseRun, status: 'running' }} sourceLabel="Paper One" />);
  expect(screen.getByText('Paper One')).toBeInTheDocument();
  expect(screen.getByText('3')).toBeInTheDocument();
  expect(screen.getByText('Running · pruning validating')).toBeInTheDocument();
  expect(screen.getByText('validating')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Stop Loop' })).toBeEnabled();
});

test('shows automatic continuation and reconnect-safe no-discovery progress', () => {
  render(
    <ProofRunStatusControls
      run={{
        ...baseRun,
        status: 'running',
        pruning_status: 'idle',
        consecutive_no_discovery_rounds: 4,
        no_discovery_round_limit: 10,
      }}
      onStop={vi.fn()}
    />
  );
  expect(screen.getByText(/Rounds continue automatically until you press Stop/i))
    .toBeInTheDocument();
  expect(screen.getByText(/Current no-discovery streak: 4\/10/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Stop Loop' })).toBeEnabled();
});

test('preserves reconnect-safe no-discovery progress while provider-paused', () => {
  render(
    <ProofRunStatusControls
      run={{
        ...baseRun,
        status: 'provider_paused',
        pruning_status: 'idle',
        consecutive_no_discovery_rounds: 4,
        no_discovery_round_limit: 10,
      }}
      onStop={vi.fn()}
    />
  );

  expect(screen.getByText(/Continuous search is paused for provider credits/i))
    .toBeInTheDocument();
  expect(screen.getByText(/Current no-discovery streak: 4\/10/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Stop Loop' })).toBeEnabled();
});

test('allows an active one-round proof check to be stopped', () => {
  render(
    <ProofRunStatusControls
      run={{
        ...baseRun,
        run_mode: 'one_round',
        unbounded: false,
        status: 'running',
      }}
      onStop={vi.fn()}
    />
  );

  expect(screen.getByRole('button', { name: 'Stop Proof Check' })).toBeEnabled();
});

test('guides repair state to a new loop and disables stop while stopping', () => {
  const { rerender } = render(
    <ProofRunStatusControls
      run={{ ...baseRun, status: 'repair_required' }}
      onStop={vi.fn()}
    />
  );
  expect(screen.queryByRole('button', { name: 'Resume' })).not.toBeInTheDocument();
  expect(screen.getByText(/repair.*then start a new proof check/i)).toBeInTheDocument();
  rerender(
    <ProofRunStatusControls
      run={{ ...baseRun, status: 'stopping' }}
      onStop={vi.fn()}
    />
  );
  expect(screen.getByRole('button', { name: 'Stopping…' })).toBeDisabled();
});

test('shows the full safe terminal error detail when the backend provides it', () => {
  render(
    <ProofRunStatusControls
      run={{
        ...baseRun,
        status: 'error',
        terminal_reason: 'repair_required',
        last_error_summary: (
          'Prompt (51489 tokens) exceeds model context window (5632 tokens). '
          + 'Increase LM Studio Context Length (n_ctx).'
        ),
      }}
    />
  );

  expect(screen.getByRole('alert')).toHaveTextContent('51489 tokens');
  expect(screen.getByRole('alert')).toHaveTextContent('n_ctx');
  // Backend repair runs end as status "error"; the guidance follows the reason.
  expect(screen.getByText(/repair.*then start a new proof check/i)).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /Stop/ })).not.toBeInTheDocument();
});
