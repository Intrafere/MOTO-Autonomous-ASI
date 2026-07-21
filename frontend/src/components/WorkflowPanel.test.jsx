import React from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import WorkflowPanel from './WorkflowPanel';
import { proofSearchAPI, workflowAPI } from '../services/api';

const listeners = new Map();

vi.mock('../services/websocket', () => ({
  websocket: {
    on: vi.fn((event, callback) => {
      if (!listeners.has(event)) listeners.set(event, []);
      listeners.get(event).push(callback);
    }),
    off: vi.fn((event, callback) => {
      const callbacks = listeners.get(event) || [];
      const index = callbacks.indexOf(callback);
      if (index >= 0) callbacks.splice(index, 1);
    }),
  },
}));

vi.mock('../services/api', () => ({
  boostAPI: {
    getStatus: vi.fn().mockResolvedValue({
      success: true,
      status: {
        enabled: false,
        boost_next_count: 0,
        boosted_categories: [],
        boost_always_prefer: false,
      },
    }),
    getCategories: vi.fn().mockResolvedValue({ success: true, categories: [] }),
    setNextCount: vi.fn(),
    setAlwaysPrefer: vi.fn(),
    toggleCategory: vi.fn(),
  },
  workflowAPI: {
    getTokenStats: vi.fn().mockResolvedValue({
      success: true,
      total_input: 10,
      total_output: 20,
      by_model: {},
      elapsed_seconds: 30,
    }),
    getPredictions: vi.fn().mockResolvedValue({ success: true, mode: 'idle', tasks: [] }),
    getSolutionPath: vi.fn().mockResolvedValue({
      success: true,
      enabled: false,
      mode: 'idle',
      run_id: null,
      revision: null,
      steps: [],
      queued_proposals: 0,
      reviewing_proposals: 0,
    }),
  },
  proofSearchAPI: {
    getAssistantLatestPack: vi.fn(),
    getProof: vi.fn(),
  },
}));

const assistantPack = {
  enabled: true,
  has_pack: true,
  target_hash: 'target_hash',
  result_count: 1,
  max_result_count: 7,
  results: [
    {
      search_id: 'manual:proof_1',
      corpus: 'manual',
      corpus_scope: 'history',
      proof_id: 'proof_1',
      session_id: 'manual_run_1',
      theorem_name: 'Memory.Helper',
      theorem_statement: 'theorem helper : True',
      proof_description: 'A hydrated proof support.',
      novelty_tier: 'mathematical_discovery',
      novelty_reasoning: 'Useful discovery.',
      relevance_reason: 'Matches the active target.',
      lean_code: '',
      has_hydrated_code: true,
    },
  ],
};

beforeEach(() => {
  vi.clearAllMocks();
  listeners.clear();
  proofSearchAPI.getAssistantLatestPack.mockResolvedValue(assistantPack);
  proofSearchAPI.getProof.mockResolvedValue({
    ...assistantPack.results[0],
    proof_description: 'Hydrated proof history detail.',
    lean_code: 'theorem helper : True := by trivial',
  });
  window.localStorage.clear();
  window.localStorage.setItem('workflow_panel_collapsed', 'false');
});

test('renders Assistant Memory Bank proof tiles and opens hydrated proof preview', async () => {
  const user = userEvent.setup();
  const onOpenAssistantProof = vi.fn();

  render(
    <WorkflowPanel
      isRunning={false}
      onOpenBoostSettings={vi.fn()}
      onOpenAssistantProof={onOpenAssistantProof}
    />
  );

  expect(await screen.findByText('Assistant Memory Bank')).toBeInTheDocument();
  const tile = await screen.findByRole('button', { name: /Memory.Helper/i });
  expect(tile.className).toContain('assistant-proof-tile--gold');
  expect(screen.getByText('1/7')).toBeInTheDocument();

  await user.click(tile);

  expect(onOpenAssistantProof).toHaveBeenCalledWith(expect.objectContaining({ proof_id: 'proof_1' }));
  await waitFor(() => {
    expect(proofSearchAPI.getProof).toHaveBeenCalledWith('manual', 'proof_1', {
      searchId: 'manual:proof_1',
      runId: null,
      sessionId: 'manual_run_1',
    });
  });
  expect(await screen.findByText(/Hydrated proof history detail/i)).toBeInTheDocument();
  expect(screen.getByText(/theorem helper : True := by trivial/i)).toBeInTheDocument();
});

test('shows Assistant memory disabled reason from latest-pack status', async () => {
  proofSearchAPI.getAssistantLatestPack.mockResolvedValue({
    enabled: false,
    has_pack: false,
    results: [],
    disabled_reason: 'Session History Memory is disabled.',
  });

  render(
    <WorkflowPanel
      isRunning={false}
      onOpenBoostSettings={vi.fn()}
      onOpenAssistantProof={vi.fn()}
    />
  );

  expect(await screen.findByText('Session History Memory is disabled.')).toBeInTheDocument();
});

test('disabled latest-pack response clears stale Assistant proof tiles', async () => {
  proofSearchAPI.getAssistantLatestPack.mockResolvedValue(assistantPack);

  render(
    <WorkflowPanel
      isRunning={false}
      onOpenBoostSettings={vi.fn()}
      onOpenAssistantProof={vi.fn()}
      collapsed={false}
    />
  );

  expect(await screen.findByText('Memory.Helper')).toBeInTheDocument();
  act(() => {
    listeners.get('assistant_proof_pack_updated')?.forEach((callback) => callback({
      enabled: false,
      has_pack: false,
      results: [],
      disabled_reason: 'Session History Memory is disabled.',
    }));
  });
  await waitFor(() => {
    expect(screen.queryByText('Memory.Helper')).not.toBeInTheDocument();
  });
  expect(await screen.findByText('Session History Memory is disabled.')).toBeInTheDocument();
});

test('refreshes queued solution-path state and ignores stale revision events', async () => {
  workflowAPI.getSolutionPath
    .mockResolvedValueOnce({
      success: true,
      enabled: true,
      mode: 'autonomous',
      run_id: 'run-1',
      acceptance_count: 5,
      revision: 3,
      steps: [{ step_id: 'one', title: 'First', status: 'active' }],
      queued_proposals: 0,
      reviewing_proposals: 0,
    })
    .mockResolvedValueOnce({
      success: true,
      enabled: true,
      mode: 'autonomous',
      run_id: 'run-1',
      acceptance_count: 5,
      revision: 3,
      steps: [{ step_id: 'one', title: 'First', status: 'active' }],
      queued_proposals: 1,
      reviewing_proposals: 0,
    });

  render(
    <WorkflowPanel
      isRunning
      onOpenBoostSettings={vi.fn()}
      onOpenAssistantProof={vi.fn()}
      collapsed={false}
    />
  );
  expect(await screen.findByText(/1 steps · revision 3/i)).toBeInTheDocument();

  act(() => {
    listeners.get('solution_path_proposal_queued')?.forEach((callback) => callback({
      run_id: 'run-1',
      revision: 3,
    }));
  });
  await waitFor(() => expect(workflowAPI.getSolutionPath).toHaveBeenCalledTimes(2));
  expect(screen.getByText(/1 solution-path update\(s\) queued/i)).toBeInTheDocument();

  act(() => {
    listeners.get('solution_path_updated')?.forEach((callback) => callback({
      run_id: 'run-1',
      revision: 2,
    }));
  });
  expect(workflowAPI.getSolutionPath).toHaveBeenCalledTimes(2);
});

test('hides the solution path before five accepted ideas', async () => {
  workflowAPI.getSolutionPath.mockResolvedValue({
    success: true,
    enabled: false,
    ownership: 'active',
    mode: 'aggregator',
    run_id: 'run-pre-five',
    acceptance_count: 4,
    steps: [],
  });

  render(
    <WorkflowPanel
      isRunning
      onOpenBoostSettings={vi.fn()}
      onOpenAssistantProof={vi.fn()}
      collapsed={false}
    />
  );

  await waitFor(() => expect(workflowAPI.getSolutionPath).toHaveBeenCalled());
  expect(screen.queryByRole('button', { name: /open current solution path/i })).not.toBeInTheDocument();
});

test('ignores an older overlapping solution-path request', async () => {
  let resolveFirst;
  workflowAPI.getSolutionPath
    .mockImplementationOnce(() => new Promise((resolve) => { resolveFirst = resolve; }))
    .mockResolvedValueOnce({
      success: true,
      enabled: true,
      ownership: 'active',
      mode: 'autonomous',
      run_id: 'new-run',
      lifecycle_generation: 2,
      acceptance_count: 5,
      revision: 2,
      steps: [{ step_id: 'new', title: 'New route', status: 'active' }],
    });

  const { rerender } = render(
    <WorkflowPanel
      isRunning={false}
      onOpenBoostSettings={vi.fn()}
      onOpenAssistantProof={vi.fn()}
      collapsed={false}
    />
  );
  rerender(
    <WorkflowPanel
      isRunning
      onOpenBoostSettings={vi.fn()}
      onOpenAssistantProof={vi.fn()}
      collapsed={false}
    />
  );

  expect(await screen.findByText(/1 steps · revision 2/i)).toBeInTheDocument();
  await act(async () => {
    resolveFirst({
      success: true,
      enabled: false,
      ownership: 'resumable',
      mode: 'aggregator',
      run_id: 'old-run',
      acceptance_count: 5,
      steps: [],
    });
  });
  expect(screen.getByText(/1 steps · revision 2/i)).toBeInTheDocument();
});

