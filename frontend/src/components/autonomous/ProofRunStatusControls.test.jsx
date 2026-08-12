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

test('shows that a continuous run advances automatically until Stop', () => {
  render(
    <ProofRunStatusControls
      run={{
        ...baseRun,
        status: 'running',
        pruning_status: 'idle',
      }}
      onStop={vi.fn()}
    />
  );
  expect(screen.getByText(/Every completed round starts the next round automatically/i))
    .toBeInTheDocument();
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
  expect(screen.getByText(/repair.*then start a new proof loop/i)).toBeInTheDocument();
  rerender(
    <ProofRunStatusControls
      run={{ ...baseRun, status: 'stopping' }}
      onStop={vi.fn()}
    />
  );
  expect(screen.getByRole('button', { name: 'Stopping…' })).toBeDisabled();
});
