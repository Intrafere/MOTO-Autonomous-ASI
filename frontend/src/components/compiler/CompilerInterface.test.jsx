import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, test, vi } from 'vitest';
import CompilerInterface from './CompilerInterface';
import { compilerAPI } from '../../services/api';

vi.mock('../../services/api', () => ({
  autonomousAPI: {
    getProofStatus: vi.fn(),
    updateProofSettings: vi.fn(),
  },
  compilerAPI: {
    getStatus: vi.fn(),
    getPrompt: vi.fn(),
    start: vi.fn(),
    stop: vi.fn(),
  },
}));

vi.mock('../../services/websocket', () => ({
  websocket: {
    on: vi.fn(),
    off: vi.fn(),
  },
}));

vi.mock('../TextFileUploader', () => ({
  default: () => null,
}));

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  sessionStorage.clear();
  compilerAPI.getStatus.mockResolvedValue({
    data: {
      is_running: false,
      current_mode: 'idle',
    },
  });
  compilerAPI.getPrompt.mockResolvedValue({
    data: {
      prompt: 'Write a focused paper from the aggregator database.',
    },
  });
});

test('keeps single paper writer start and output controls available in the prompt composer', async () => {
  render(
    <CompilerInterface
      activeTab="compiler-interface"
      capabilities={{ genericMode: false, lmStudioEnabled: true }}
      anyWorkflowRunning={false}
      onWorkflowRunningChange={vi.fn()}
      connectivityStatus={{
        skills: {
          agent_conversation_memory: { enabled: true },
        },
      }}
    />
  );

  expect(await screen.findByLabelText('Compiler-Directing Prompt:')).toBeInTheDocument();
  expect(
    screen.getByText(/one live rigorous research paper or solution report/i)
  ).toBeInTheDocument();
  expect(
    screen.getByText(/what kind of rigorous paper or solution report/i)
  ).toBeInTheDocument();
  await waitFor(() => {
    expect(screen.getByRole('button', { name: 'Start Writer' })).toBeEnabled();
  });
  expect(screen.getByLabelText('Mathematical Proofs')).toBeInTheDocument();
  expect(screen.getByLabelText('Research Papers')).toBeInTheDocument();
});
