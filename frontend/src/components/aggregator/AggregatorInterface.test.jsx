import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, test, vi } from 'vitest';
import AggregatorInterface from './AggregatorInterface';
import { api } from '../../services/api';

vi.mock('../../services/api', () => ({
  api: {
    getStatus: vi.fn(),
    startAggregator: vi.fn(),
    stopAggregator: vi.fn(),
    uploadFile: vi.fn(),
  },
}));

vi.mock('../TextFileUploader', () => ({
  default: () => null,
}));

const baseConfig = {
  userPrompt: 'Find useful mathematical ideas.',
  submitterConfigs: [
    {
      submitterId: 1,
      provider: 'openrouter',
      modelId: 'openai/gpt-5.5',
      openrouterProvider: null,
      openrouterReasoningEffort: 'auto',
      lmStudioFallbackId: null,
      contextWindow: 400000,
      maxOutputTokens: 85000,
    },
  ],
  validatorProvider: 'openrouter',
  validatorModel: 'minimax/minimax-m3',
  validatorOpenrouterProvider: 'AtlasCloud',
  validatorOpenrouterReasoningEffort: 'auto',
  validatorLmStudioFallback: null,
  validatorContextSize: 1048576,
  validatorMaxOutput: 209715,
  assistantProvider: 'openrouter',
  assistantModel: 'google/gemini-3.1-flash-lite',
  assistantOpenrouterProvider: null,
  assistantOpenrouterReasoningEffort: 'auto',
  assistantLmStudioFallback: null,
  assistantContextSize: 65536,
  assistantMaxOutput: 8192,
  uploadedFiles: [],
};

beforeEach(() => {
  vi.clearAllMocks();
  api.getStatus.mockResolvedValue({
    is_running: false,
    total_submissions: 0,
    total_acceptances: 0,
    total_rejections: 0,
  });
  api.startAggregator.mockResolvedValue({ status: 'started' });
});

test('does not inherit Validator host provider when Assistant host is Auto', async () => {
  render(
    <AggregatorInterface
      config={baseConfig}
      setConfig={vi.fn()}
      capabilities={{ lmStudioEnabled: true }}
      connectivityStatus={{
        skills: {
          agent_conversation_memory: { enabled: true },
        },
      }}
    />
  );

  fireEvent.click(await screen.findByRole('button', { name: /start aggregator/i }));

  await waitFor(() => {
    expect(api.startAggregator).toHaveBeenCalled();
  });
  const payload = api.startAggregator.mock.calls[0][0];
  expect(payload.validator_openrouter_provider).toBe('AtlasCloud');
  expect(payload.assistant_openrouter_provider).toBeNull();
});
