import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import TokenCountNote, { describeTokenCalibration } from './TokenCountNote';
import { workflowAPI } from '../services/api';
import {
  findTokenCalibration,
  refreshTokenCalibration,
  resetTokenCalibration,
  resetTokenCalibrationStoreForTests,
  setTokenCalibrationAdjustEnabled,
} from '../hooks/useTokenCalibration';

vi.mock('../services/api', () => ({
  workflowAPI: {
    getTokenCalibration: vi.fn(),
    setTokenCalibrationAdjustEnabled: vi.fn(),
    resetTokenCalibration: vi.fn(),
  },
}));

const countsMore = {
  provider: 'claude_api',
  model: 'claude-opus-5-5',
  state: 'counts_more',
  samples: 12,
  samples_required: 10,
  ratio: 1.25,
  applied: true,
  context_windows: [{ configured: 1000000, effective: 800000 }],
};
const measuring = { ...countsMore, model: 'claude-sonnet-5-5', state: 'measuring', samples: 4, ratio: null, applied: false };
const matches = { ...countsMore, provider: 'openai_codex_oauth', model: 'gpt-5.5', state: 'matches', ratio: 1.01, applied: false };

const snapshot = (overrides = {}) => ({
  success: true,
  adjust_enabled: true,
  samples_required: 10,
  minimum_sample_tokens: 8000,
  models: [countsMore, measuring, matches],
  ...overrides,
});

describe('TokenCountNote', () => {
  beforeEach(() => {
    resetTokenCalibrationStoreForTests();
    workflowAPI.getTokenCalibration.mockReset().mockResolvedValue(snapshot());
    workflowAPI.setTokenCalibrationAdjustEnabled.mockReset();
    workflowAPI.resetTokenCalibration.mockReset();
  });

  it('shows nothing for a model that has not been measured', async () => {
    const { container } = render(
      <TokenCountNote provider="openrouter" modelId="vendor/unmeasured" contextWindow={200000} />,
    );
    await waitFor(() => expect(workflowAPI.getTokenCalibration).toHaveBeenCalledTimes(1));
    expect(container).toBeEmptyDOMElement();
  });

  it('states the measured difference and the window MOTO budgets against', async () => {
    render(<TokenCountNote provider="claude_api" modelId="claude-opus-5-5" contextWindow={1000000} />);
    expect(
      await screen.findByText(/counts about 25% more tokens than MOTO's estimate\. MOTO fits prompts to 800,000 tokens\./),
    ).toBeInTheDocument();
  });

  it('reports progress while a model is still being measured', async () => {
    render(<TokenCountNote provider="claude_api" modelId="claude-sonnet-5-5" contextWindow={1000000} />);
    expect(await screen.findByText(/4 of 10 measurements/)).toBeInTheDocument();
  });

  it('says when a verified model matches', async () => {
    render(<TokenCountNote provider="openai_codex_oauth" modelId="gpt-5.5" contextWindow={400000} />);
    expect(await screen.findByText(/matches MOTO's estimate/)).toBeInTheDocument();
  });

  it('says when the adjustment is turned off', () => {
    expect(describeTokenCalibration(countsMore, { adjustEnabled: false, contextWindow: 1000000 })).toMatch(
      /The adjustment is turned off\./,
    );
  });

  it('shares one request between every note on the page', async () => {
    render(
      <>
        <TokenCountNote provider="claude_api" modelId="claude-opus-5-5" contextWindow={1000000} />
        <TokenCountNote provider="openai_codex_oauth" modelId="gpt-5.5" contextWindow={400000} />
      </>,
    );
    await screen.findByText(/matches MOTO's estimate/);
    expect(workflowAPI.getTokenCalibration).toHaveBeenCalledTimes(1);
  });

  it('keeps the last snapshot when a refresh fails', async () => {
    render(<TokenCountNote provider="claude_api" modelId="claude-opus-5-5" contextWindow={1000000} />);
    await screen.findByText(/counts about 25% more/);
    workflowAPI.getTokenCalibration.mockRejectedValueOnce(new Error('offline'));
    await refreshTokenCalibration();
    expect(screen.getByText(/counts about 25% more/)).toBeInTheDocument();
  });

  it('updates every note when the adjustment is toggled or a model is reset', async () => {
    render(<TokenCountNote provider="claude_api" modelId="claude-opus-5-5" contextWindow={1000000} />);
    await screen.findByText(/MOTO fits prompts to 800,000 tokens/);

    workflowAPI.setTokenCalibrationAdjustEnabled.mockResolvedValue(snapshot({ adjust_enabled: false }));
    await setTokenCalibrationAdjustEnabled(false);
    expect(await screen.findByText(/The adjustment is turned off\./)).toBeInTheDocument();
    expect(workflowAPI.setTokenCalibrationAdjustEnabled).toHaveBeenCalledWith(false);

    workflowAPI.resetTokenCalibration.mockResolvedValue(snapshot({ models: [] }));
    await resetTokenCalibration('claude_api', 'claude-opus-5-5');
    await waitFor(() => expect(screen.queryByText(/counts about 25% more/)).not.toBeInTheDocument());
  });

  it('matches loaded LM Studio sibling instances to one measurement', () => {
    const calibration = { models: [{ provider: 'lm_studio', model: 'qwen-3', state: 'matches' }] };
    expect(findTokenCalibration(calibration, 'lm_studio', 'qwen-3:2')).toBe(calibration.models[0]);
    expect(findTokenCalibration(calibration, 'lm_studio', '')).toBeNull();
  });
});
