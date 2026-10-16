import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import OpenRouterApiKeyModal from './OpenRouterApiKeyModal';
import { cloudAccessAPI, openRouterAPI } from '../services/api';

vi.mock('../services/api', () => ({
  cloudAccessAPI: {
    getOpenAICodexStatus: vi.fn(),
    getOpenAICodexModels: vi.fn(),
    getXAIGrokStatus: vi.fn(),
    getXAIGrokModels: vi.fn(),
    getProviderStatus: vi.fn(),
    getProviderModels: vi.fn(),
    getProviderUsage: vi.fn(),
    setProviderApiKey: vi.fn(),
    clearProviderApiKey: vi.fn(),
  },
  openRouterAPI: {
    getApiKeyStatus: vi.fn(),
  },
}));

const DESKTOP = { genericMode: false, lmStudioEnabled: true };

function renderModal(capabilities = DESKTOP) {
  const onCloudAccessChanged = vi.fn();
  render(
    <OpenRouterApiKeyModal
      isOpen
      onClose={vi.fn()}
      onCloudAccessChanged={onCloudAccessChanged}
      capabilities={capabilities}
    />
  );
  return { onCloudAccessChanged };
}

beforeEach(() => {
  vi.clearAllMocks();
  openRouterAPI.getApiKeyStatus.mockResolvedValue({ has_key: false });
  cloudAccessAPI.getOpenAICodexStatus.mockResolvedValue({ status: { configured: false } });
  cloudAccessAPI.getXAIGrokStatus.mockResolvedValue({ status: { configured: false } });
  cloudAccessAPI.getProviderStatus.mockResolvedValue({ status: { configured: false } });
  cloudAccessAPI.getProviderModels.mockResolvedValue({ models: [] });
  cloudAccessAPI.getProviderUsage.mockResolvedValue({ usage: { pools: [] } });
});

test('auto-selects a configured Kimi Code key and verifies its models', async () => {
  cloudAccessAPI.getProviderStatus.mockImplementation(async (providerId) => ({
    status: providerId === 'kimi_code'
      ? { configured: true, region: 'international', region_label: 'International (api.kimi.ai)' }
      : { configured: false },
  }));
  cloudAccessAPI.getProviderModels.mockResolvedValue({ models: [{ id: 'k3' }, { id: 'k3-256k' }] });
  renderModal();

  expect(await screen.findByText('SUCCESS! Kimi Code connected. 2 models available.')).toBeInTheDocument();
  expect(screen.getByRole('option', { name: 'Kimi Code (configured)' }).selected).toBe(true);
  expect(screen.getByText(/International \(api\.kimi\.ai\)/)).toBeInTheDocument();
  expect(cloudAccessAPI.getProviderModels).toHaveBeenCalledWith('kimi_code');
  expect(cloudAccessAPI.getProviderModels).not.toHaveBeenCalledWith('sakana_fugu');
});

test('every API-key provider renders through the shared section', async () => {
  renderModal();
  const select = await screen.findByRole('combobox');
  for (const [providerId, label] of [['sakana_fugu', 'Sakana Fugu'], ['kimi_code', 'Kimi Code']]) {
    fireEvent.change(select, { target: { value: providerId } });
    expect(await screen.findByLabelText(`${label} API Key`)).toHaveValue('');
    expect(screen.getByText(`${label} API key is not configured.`)).toBeInTheDocument();
  }
});

test('a provider the backend reports unavailable is not offered or queried', async () => {
  renderModal({ ...DESKTOP, kimiCodeAvailable: false });
  await waitFor(() => expect(cloudAccessAPI.getProviderStatus).toHaveBeenCalledWith('sakana_fugu'));
  expect(screen.queryByRole('option', { name: /Kimi Code/ })).toBeNull();
  expect(cloudAccessAPI.getProviderStatus).not.toHaveBeenCalledWith('kimi_code');
});
