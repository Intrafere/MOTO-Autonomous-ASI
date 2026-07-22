import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import AutonomousResearchInterface from './AutonomousResearchInterface';

const baseProps = {
  isRunning: false,
  isStopping: false,
  anyWorkflowRunning: false,
  status: {},
  activity: [],
  onStart: vi.fn(),
  onStop: vi.fn(),
  onClear: vi.fn(),
  config: {
    allow_mathematical_proofs: false,
    allow_research_papers: true,
  },
  onConfigChange: vi.fn(),
  capabilities: { genericMode: true },
  api: {},
};

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  sessionStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

test('keeps large autonomous prompts editable when localStorage quota is exceeded', async () => {
  const quotaError = new DOMException('Quota exceeded', 'QuotaExceededError');
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
    throw quotaError;
  });

  render(<AutonomousResearchInterface {...baseProps} />);

  const promptInput = screen.getByLabelText('Research Goal');
  const largePrompt = 'resolve theorem\n'.repeat(10000);

  fireEvent.change(promptInput, { target: { value: largePrompt } });

  expect(promptInput).toHaveValue(largePrompt);
  await waitFor(() => {
    expect(Storage.prototype.setItem).toHaveBeenCalled();
  });
});

test('hydrates autonomous prompt synchronously from session storage fallback', () => {
  sessionStorage.setItem('autonomous_research_prompt', 'cached autonomous prompt');

  render(<AutonomousResearchInterface {...baseProps} />);

  expect(screen.getByLabelText('Research Goal')).toHaveValue('cached autonomous prompt');
});

test('keeps autonomous start and output controls available in the prompt composer', () => {
  render(
    <AutonomousResearchInterface
      {...baseProps}
      config={{
        ...baseProps.config,
        submitter_configs: [{ modelId: 'openai/gpt-5.5' }],
      }}
      capabilities={{ genericMode: false }}
      developerModeEnabled={true}
    />
  );

  expect(screen.getByLabelText('Research Goal')).toHaveAttribute(
    'placeholder',
    expect.stringContaining('S.T.E.M. research or solution objective')
  );
  expect(screen.getByRole('button', { name: 'Start Research' })).toBeEnabled();
  expect(screen.getByLabelText('Mathematical Proofs')).toBeInTheDocument();
  expect(screen.getByLabelText('Research Papers')).toBeInTheDocument();
  expect(screen.getByLabelText('Creativity Emphasis Boost')).toBeInTheDocument();
});

test('shows the red stop control immediately but locks it until start is confirmed', async () => {
  let resolveStart;
  const onStart = vi.fn(() => new Promise((resolve) => {
    resolveStart = resolve;
  }));
  sessionStorage.setItem('autonomous_research_prompt', 'research this safely');

  const { rerender } = render(
    <AutonomousResearchInterface
      {...baseProps}
      onStart={onStart}
      config={{
        ...baseProps.config,
        submitter_configs: [{ modelId: 'openai/gpt-5.5' }],
      }}
    />
  );

  fireEvent.click(screen.getByRole('button', { name: 'Start Research' }));

  await waitFor(() => {
    expect(onStart).toHaveBeenCalledTimes(1);
  });
  expect(screen.getByRole('button', { name: 'Starting...' })).toBeDisabled();

  rerender(
    <AutonomousResearchInterface
      {...baseProps}
      isRunning={true}
      onStart={onStart}
      config={{
        ...baseProps.config,
        submitter_configs: [{ modelId: 'openai/gpt-5.5' }],
      }}
    />
  );
  resolveStart(true);

  await waitFor(() => {
    expect(screen.getByRole('button', { name: 'Stop Research' })).toBeEnabled();
  });
});

test('reports pending start to the parent so settings can remain locked', async () => {
  let resolveStart;
  const onStart = vi.fn(() => new Promise((resolve) => {
    resolveStart = resolve;
  }));
  const onStartingChange = vi.fn();
  sessionStorage.setItem('autonomous_research_prompt', 'research this safely');

  render(
    <AutonomousResearchInterface
      {...baseProps}
      onStart={onStart}
      onStartingChange={onStartingChange}
      config={{
        ...baseProps.config,
        submitter_configs: [{ modelId: 'openai/gpt-5.5' }],
      }}
    />
  );

  fireEvent.click(screen.getByRole('button', { name: 'Start Research' }));
  expect(onStartingChange).toHaveBeenCalledWith(true);

  resolveStart(false);
  await waitFor(() => {
    expect(onStartingChange).toHaveBeenLastCalledWith(false);
  });
});

test('keeps autonomous prompt when confirmed clear is canceled by parent handler', async () => {
  const onClear = vi.fn().mockResolvedValue(false);
  sessionStorage.setItem('autonomous_research_prompt', 'prompt should stay');

  render(<AutonomousResearchInterface {...baseProps} onClear={onClear} />);

  const promptInput = screen.getByLabelText('Research Goal');
  expect(promptInput).toHaveValue('prompt should stay');

  fireEvent.click(screen.getByRole('button', { name: 'Clear Research Run' }));
  fireEvent.click(screen.getByRole('button', { name: 'Confirm Reset' }));

  await waitFor(() => {
    expect(onClear).toHaveBeenCalledTimes(1);
  });
  expect(promptInput).toHaveValue('prompt should stay');
  expect(localStorage.getItem('autonomous_research_prompt')).toBe('prompt should stay');
});

test('clears autonomous prompt after confirmed clear succeeds', async () => {
  const onClear = vi.fn().mockResolvedValue(true);
  sessionStorage.setItem('autonomous_research_prompt', 'prompt should clear');

  render(<AutonomousResearchInterface {...baseProps} onClear={onClear} />);

  const promptInput = screen.getByLabelText('Research Goal');
  expect(promptInput).toHaveValue('prompt should clear');

  fireEvent.click(screen.getByRole('button', { name: 'Clear Research Run' }));
  fireEvent.click(screen.getByRole('button', { name: 'Confirm Reset' }));

  await waitFor(() => {
    expect(promptInput).toHaveValue('');
  });
  expect(localStorage.getItem('autonomous_research_prompt')).toBeNull();
  expect(sessionStorage.getItem('autonomous_research_prompt')).toBeNull();
});
