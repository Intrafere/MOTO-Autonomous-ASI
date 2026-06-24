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

test('keeps autonomous prompt when confirmed clear is canceled by parent handler', async () => {
  const onClear = vi.fn().mockResolvedValue(false);
  sessionStorage.setItem('autonomous_research_prompt', 'prompt should stay');

  render(<AutonomousResearchInterface {...baseProps} onClear={onClear} />);

  const promptInput = screen.getByLabelText('Research Goal');
  expect(promptInput).toHaveValue('prompt should stay');

  fireEvent.click(screen.getByRole('button', { name: 'Clear All' }));
  fireEvent.click(screen.getByRole('button', { name: 'Confirm Clear' }));

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

  fireEvent.click(screen.getByRole('button', { name: 'Clear All' }));
  fireEvent.click(screen.getByRole('button', { name: 'Confirm Clear' }));

  await waitFor(() => {
    expect(promptInput).toHaveValue('');
  });
  expect(localStorage.getItem('autonomous_research_prompt')).toBeNull();
  expect(sessionStorage.getItem('autonomous_research_prompt')).toBeNull();
});
