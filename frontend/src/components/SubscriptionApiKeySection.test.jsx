import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import SubscriptionApiKeySection from './SubscriptionApiKeySection';
import { cloudAccessAPI } from '../services/api';

vi.mock('../services/api', () => ({
  cloudAccessAPI: {
    setProviderApiKey: vi.fn(),
    clearProviderApiKey: vi.fn(),
    getProviderUsage: vi.fn(),
  },
}));

const EMPTY_MODELS = { checking: false, count: null, error: '' };

function renderSection(props = {}) {
  const handlers = {
    verifyModels: vi.fn().mockResolvedValue(true),
    onStatusChange: vi.fn(),
    onModelsStatusChange: vi.fn(),
    onCloudAccessChanged: vi.fn(),
    onError: vi.fn(),
  };
  render(
    <SubscriptionApiKeySection
      providerId="kimi_code"
      genericMode={false}
      status={{ configured: false }}
      modelsStatus={EMPTY_MODELS}
      successBannerStyle={{}}
      getMessageBannerClass={() => 'test-result-banner'}
      {...handlers}
      {...props}
    />
  );
  return handlers;
}

beforeEach(() => {
  vi.clearAllMocks();
});

test('saves a Kimi Code key through the registry route and reports the change', async () => {
  cloudAccessAPI.setProviderApiKey.mockResolvedValue({
    status: { configured: true, region: 'international', region_label: 'International (api.kimi.ai)' },
    models: [{ id: 'k3' }],
  });
  const handlers = renderSection();

  const input = screen.getByLabelText('Kimi Code API Key');
  expect(input).toHaveAttribute('name', 'moto-kimi-code-api-key');
  expect(input).toHaveValue('');
  fireEvent.change(input, { target: { value: '  sk-kimi-test  ' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save Kimi Key' }));

  await waitFor(() => expect(cloudAccessAPI.setProviderApiKey).toHaveBeenCalledWith('kimi_code', 'sk-kimi-test'));
  await waitFor(() => expect(handlers.onCloudAccessChanged).toHaveBeenCalledWith(true, 'kimi_code', { modelsReady: true }));
  expect(handlers.onModelsStatusChange).toHaveBeenCalledWith({ checking: false, count: 1, error: '' });
  expect(await screen.findByText('Kimi Code API key saved and model list loaded.')).toBeInTheDocument();
  expect(input).toHaveValue('');
});

test('shows Kimi quota pools with reset times and refreshes only on request', async () => {
  cloudAccessAPI.getProviderUsage.mockResolvedValue({
    usage: { pools: [{ id: 'limit_5h', label: '5-hour window', used_ratio: 0.42, reset_at: 1792008000 }] },
  });
  renderSection({ status: { configured: true, region_label: 'China (api.kimi.com)' } });

  expect(await screen.findByText('42% used')).toBeInTheDocument();
  expect(screen.getByRole('progressbar', { name: '5-hour window quota used' })).toHaveAttribute('aria-valuenow', '42');
  expect(screen.getByText(/China \(api\.kimi\.com\)/)).toBeInTheDocument();
  expect(cloudAccessAPI.getProviderUsage).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  await waitFor(() => expect(cloudAccessAPI.getProviderUsage).toHaveBeenCalledTimes(2));
});

test('Sakana has no quota view and clears through its own route', async () => {
  cloudAccessAPI.clearProviderApiKey.mockResolvedValue({ success: true });
  const handlers = renderSection({ providerId: 'sakana_fugu', status: { configured: true } });

  expect(screen.queryByText('Subscription quota')).toBeNull();
  expect(cloudAccessAPI.getProviderUsage).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Clear Fugu Key' }));
  await waitFor(() => expect(cloudAccessAPI.clearProviderApiKey).toHaveBeenCalledWith('sakana_fugu'));
  expect(handlers.onStatusChange).toHaveBeenCalledWith({ configured: false });
  expect(handlers.onCloudAccessChanged).toHaveBeenCalledWith(false, 'sakana_fugu');
});

test('hosted mode disables the key input and explains desktop-only access', () => {
  renderSection({ genericMode: true });
  expect(screen.getByText(/desktop-only in this build/)).toBeInTheDocument();
  expect(screen.getByLabelText('Kimi Code API Key')).toBeDisabled();
  expect(cloudAccessAPI.getProviderUsage).not.toHaveBeenCalled();
});
