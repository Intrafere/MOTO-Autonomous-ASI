import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SolutionPathModal from './SolutionPathModal';
import { workflowAPI } from '../services/api';

vi.mock('../services/websocket', () => ({
  websocket: { on: vi.fn(), off: vi.fn() },
}));

vi.mock('../services/api', () => ({
  workflowAPI: {
    getSolutionPath: vi.fn(),
    resumeSolutionPathProposal: vi.fn(),
  },
}));

const repairSnapshot = {
  enabled: false,
  ownership: 'active',
  mode: 'compiler',
  run_id: 'manual-run',
  lifecycle_generation: 4,
  acceptance_count: 7,
  revision: null,
  steps: [],
  repairs: [{
    proposal_id: 'proposal-1',
    reason: 'context_overflow',
    detail: 'Increase the context window.',
    lifecycle_generation: 4,
  }],
  message: 'Solution path tracking is loaded; no approved plan is available yet.',
};

beforeEach(() => {
  vi.clearAllMocks();
});

test('routes compiler repair to Aggregator Main Submitter settings and presents no-plan metadata', async () => {
  const user = userEvent.setup();
  const onOpenSettings = vi.fn();
  render(
    <SolutionPathModal
      snapshot={repairSnapshot}
      onClose={vi.fn()}
      onSnapshotChange={vi.fn()}
      onOpenSettings={onOpenSettings}
    />
  );

  expect(screen.getByText('No approved route yet')).toBeInTheDocument();
  expect(screen.getByText(/7 accepted brainstorm ideas/i)).toBeInTheDocument();
  expect(screen.getByText('Run manual-run')).toBeInTheDocument();
  expect(screen.getByText('Generation 4')).toBeInTheDocument();

  await user.click(screen.getByRole('button', { name: /Open Main Submitter 1 settings/i }));
  expect(onOpenSettings).toHaveBeenCalledWith('aggregator');
});

test('refreshes a retry only within the same run and generation and announces success', async () => {
  const user = userEvent.setup();
  const onSnapshotChange = vi.fn();
  const refreshed = { ...repairSnapshot, repairs: [], queued_proposals: 1 };
  workflowAPI.resumeSolutionPathProposal.mockResolvedValue({ success: true });
  workflowAPI.getSolutionPath.mockResolvedValue(refreshed);

  render(
    <SolutionPathModal
      snapshot={repairSnapshot}
      onClose={vi.fn()}
      onSnapshotChange={onSnapshotChange}
      onOpenSettings={vi.fn()}
    />
  );
  const retryButton = screen.getByRole('button', { name: 'Retry review' });
  await user.click(retryButton);

  await waitFor(() => expect(onSnapshotChange).toHaveBeenCalledWith(refreshed));
  expect(screen.getByRole('status')).toHaveTextContent(/retry queued/i);
  await waitFor(() => expect(retryButton).toHaveFocus());
});

test('rejects a retry refresh from a different lifecycle generation', async () => {
  const user = userEvent.setup();
  const onSnapshotChange = vi.fn();
  workflowAPI.resumeSolutionPathProposal.mockResolvedValue({ success: true });
  workflowAPI.getSolutionPath.mockResolvedValue({
    ...repairSnapshot,
    lifecycle_generation: 5,
  });

  render(
    <SolutionPathModal
      snapshot={repairSnapshot}
      onClose={vi.fn()}
      onSnapshotChange={onSnapshotChange}
      onOpenSettings={vi.fn()}
    />
  );
  await user.click(screen.getByRole('button', { name: 'Retry review' }));

  expect(await screen.findByRole('alert')).toHaveTextContent(/run changed/i);
  expect(onSnapshotChange).not.toHaveBeenCalled();
});
