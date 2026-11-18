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

test('Amazon Bedrock saves the key with the region the user chose', async () => {
  cloudAccessAPI.setProviderApiKey.mockResolvedValue({
    status: { configured: true, region: 'eu-west-1', region_label: 'Europe (Ireland) - eu-west-1' },
    models: [{ id: 'anthropic.claude-opus-5-5' }],
  });
  const handlers = renderSection({
    providerId: 'amazon_bedrock',
    status: {
      configured: false,
      region_selection: 'user',
      default_region: 'us-east-1',
      region_options: [
        { id: 'us-east-1', label: 'US East (N. Virginia) - us-east-1' },
        { id: 'eu-west-1', label: 'Europe (Ireland) - eu-west-1' },
      ],
    },
  });

  const region = screen.getByLabelText('Amazon Bedrock Region');
  expect(region).toHaveValue('us-east-1');
  fireEvent.change(region, { target: { value: 'eu-west-1' } });
  fireEvent.change(screen.getByLabelText('Amazon Bedrock API Key'), { target: { value: 'bedrock-test-key' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save Bedrock Key' }));

  await waitFor(() => expect(cloudAccessAPI.setProviderApiKey)
    .toHaveBeenCalledWith('amazon_bedrock', 'bedrock-test-key', 'eu-west-1'));
  await waitFor(() => expect(handlers.onCloudAccessChanged)
    .toHaveBeenCalledWith(true, 'amazon_bedrock', { modelsReady: true }));
});

test('clearing an Amazon Bedrock key keeps the region choices for the next key', async () => {
  cloudAccessAPI.clearProviderApiKey.mockResolvedValue({ success: true });
  const regionOptions = [
    { id: 'us-east-1', label: 'US East (N. Virginia) - us-east-1' },
    { id: 'eu-west-1', label: 'Europe (Ireland) - eu-west-1' },
  ];
  const handlers = renderSection({
    providerId: 'amazon_bedrock',
    status: {
      configured: true,
      region: 'eu-west-1',
      region_selection: 'user',
      default_region: 'eu-west-1',
      region_options: regionOptions,
    },
  });

  fireEvent.click(screen.getByRole('button', { name: 'Clear Bedrock Key' }));
  await waitFor(() => expect(handlers.onStatusChange).toHaveBeenCalledWith({
    configured: false,
    region_selection: 'user',
    region_options: regionOptions,
    default_region: 'eu-west-1',
  }));
});

test('providers without a region choice show no region picker', () => {
  renderSection({ providerId: 'claude_api' });
  expect(screen.getByLabelText('Anthropic Claude API Key')).toBeInTheDocument();
  expect(screen.queryByLabelText('Anthropic Claude Region')).toBeNull();
});

test('hosted mode disables the key input and explains desktop-only access', () => {
  renderSection({ genericMode: true });
  expect(screen.getByText(/desktop-only in this build/)).toBeInTheDocument();
  expect(screen.getByLabelText('Kimi Code API Key')).toBeDisabled();
  expect(cloudAccessAPI.getProviderUsage).not.toHaveBeenCalled();
});
