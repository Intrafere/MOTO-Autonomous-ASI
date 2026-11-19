import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import VllmConnectivityModal, { parseVllmServerList } from './VllmConnectivityModal';
import { vllmAPI } from '../services/api';

vi.mock('../services/api', () => ({
  vllmAPI: {
    getStatus: vi.fn(),
    test: vi.fn(),
    saveConfig: vi.fn(),
    clear: vi.fn(),
  },
}));

const CHAT_SERVER = 'http://127.0.0.1:8001/v1';
const EMBED_SERVER = 'http://127.0.0.1:8002/v1';
const PROBE = {
  servers: [
    { url: CHAT_SERVER, reachable: true, error: '', models: [{ id: 'Qwen/Qwen2.5', kind: 'chat', context_length: 32768, server: CHAT_SERVER }] },
    { url: EMBED_SERVER, reachable: true, error: '', models: [{ id: 'nomic-embed', kind: 'embedding', context_length: 8192, server: EMBED_SERVER }] },
  ],
  chat_models: [{ id: 'Qwen/Qwen2.5', kind: 'chat', context_length: 32768, server: CHAT_SERVER }],
  embedding_models: [{ id: 'nomic-embed', kind: 'embedding', context_length: 8192, server: EMBED_SERVER }],
  embedding_model: 'nomic-embed',
  embedding_server: EMBED_SERVER,
};

function renderModal(props = {}) {
  const handlers = { onClose: vi.fn(), onChanged: vi.fn(), onRefresh: vi.fn() };
  render(
    <VllmConnectivityModal
      isOpen
      status={{ configured: false }}
      capabilities={{ genericMode: false, vllmAvailable: true }}
      {...handlers}
      {...props}
    />
  );
  return handlers;
}

beforeEach(() => {
  vi.clearAllMocks();
  vllmAPI.getStatus.mockResolvedValue({ status: { configured: false, servers: [], has_api_key: false } });
});

test('splits a pasted server list on new lines, spaces, and commas', () => {
  expect(parseVllmServerList(' http://a:8001\nhttp://b:8002, c:9000 ')).toEqual([
    'http://a:8001', 'http://b:8002', 'c:9000',
  ]);
  expect(parseVllmServerList('')).toEqual([]);
});

test('renders nothing while closed and never calls the backend', () => {
  render(<VllmConnectivityModal isOpen={false} capabilities={{}} onClose={vi.fn()} />);
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(vllmAPI.getStatus).not.toHaveBeenCalled();
});

test('hosted mode shows the desktop-only notice without contacting the backend', () => {
  renderModal({ capabilities: { genericMode: true } });
  expect(screen.getByText(/disabled in hosted\/generic mode/)).toBeInTheDocument();
  expect(vllmAPI.getStatus).not.toHaveBeenCalled();
  expect(screen.queryByLabelText(/Server addresses/)).not.toBeInTheDocument();
});

test('testing servers shows detected models and context windows without saving', async () => {
  vllmAPI.test.mockResolvedValue(PROBE);
  const { onChanged } = renderModal();
  await waitFor(() => expect(vllmAPI.getStatus).toHaveBeenCalled());

  fireEvent.change(screen.getByLabelText(/Server addresses/), {
    target: { value: 'http://127.0.0.1:8001\nhttp://127.0.0.1:8002' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Test Servers' }));

  await screen.findByText(/Nothing was saved yet/);
  expect(vllmAPI.test).toHaveBeenCalledWith({
    servers: ['http://127.0.0.1:8001', 'http://127.0.0.1:8002'],
    apiKey: '',
    embeddingModel: '',
  });
  expect(screen.getByText('Qwen/Qwen2.5')).toBeInTheDocument();
  expect(screen.getByText(/32,768 token context/)).toBeInTheDocument();
  expect(screen.getByRole('option', { name: 'nomic-embed' })).toBeInTheDocument();
  expect(vllmAPI.saveConfig).not.toHaveBeenCalled();
  expect(onChanged).not.toHaveBeenCalled();
});

test('saving sends the key once, clears it from the form, and reports the change', async () => {
  vllmAPI.saveConfig.mockResolvedValue({
    ...PROBE,
    status: { configured: true, servers: [CHAT_SERVER, EMBED_SERVER], has_api_key: true, embedding_model: 'nomic-embed' },
  });
  const { onChanged } = renderModal();
  await waitFor(() => expect(vllmAPI.getStatus).toHaveBeenCalled());

  fireEvent.change(screen.getByLabelText(/Server addresses/), { target: { value: '127.0.0.1:8001 127.0.0.1:8002' } });
  const keyInput = screen.getByLabelText(/API key \(optional\)/);
  expect(keyInput).toHaveAttribute('type', 'password');
  fireEvent.change(keyInput, { target: { value: 'secret-key' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save vLLM Setup' }));

  await screen.findByText(/vLLM setup saved/);
  expect(vllmAPI.saveConfig).toHaveBeenCalledWith({
    servers: ['127.0.0.1:8001', '127.0.0.1:8002'],
    apiKey: 'secret-key',
    embeddingModel: '',
    clearApiKey: false,
  });
  expect(keyInput).toHaveValue('');
  expect(screen.getByLabelText(/Server addresses/)).toHaveValue(`${CHAT_SERVER}\n${EMBED_SERVER}`);
  expect(onChanged).toHaveBeenCalledWith(true, expect.objectContaining({ embedding_model: 'nomic-embed' }));
});

test('a rejected save shows the backend reason and changes nothing', async () => {
  vllmAPI.saveConfig.mockRejectedValue(new Error('vLLM connection failed: http://127.0.0.1:8001/v1 did not answer'));
  const { onChanged } = renderModal();
  await waitFor(() => expect(vllmAPI.getStatus).toHaveBeenCalled());

  fireEvent.click(screen.getByRole('button', { name: 'Save vLLM Setup' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Enter at least one vLLM server address.');
  expect(vllmAPI.saveConfig).not.toHaveBeenCalled();

  fireEvent.change(screen.getByLabelText(/Server addresses/), { target: { value: '127.0.0.1:8001' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save vLLM Setup' }));
  await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('did not answer'));
  expect(onChanged).not.toHaveBeenCalled();
});

test('a saved setup is prefilled and removal needs a second confirmation', async () => {
  vllmAPI.getStatus.mockResolvedValue({
    status: { configured: true, servers: [CHAT_SERVER], has_api_key: true, embedding_model: '' },
  });
  vllmAPI.clear.mockResolvedValue({ success: true });
  const { onChanged } = renderModal({
    status: { configured: true, available: true, has_usable_chat_model: true, models: ['Qwen/Qwen2.5'] },
  });

  await waitFor(() => expect(screen.getByLabelText(/Server addresses/)).toHaveValue(CHAT_SERVER));
  expect(screen.getByText(/vLLM is active with 1 chat model/)).toBeInTheDocument();
  expect(screen.getByText(/A key is already saved/)).toBeInTheDocument();
  expect(screen.getByLabelText(/Remove the saved API key/)).not.toBeChecked();

  fireEvent.click(screen.getByRole('button', { name: 'Remove vLLM Setup' }));
  expect(vllmAPI.clear).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Confirm Remove' }));

  await screen.findByText('vLLM setup removed.');
  expect(vllmAPI.clear).toHaveBeenCalledTimes(1);
  expect(screen.getByLabelText(/Server addresses/)).toHaveValue('');
  expect(onChanged).toHaveBeenCalledWith(false, null);
});

test('environment-provided servers are shown read-only', async () => {
  vllmAPI.getStatus.mockResolvedValue({
    status: { configured: true, servers: [CHAT_SERVER], server_source: 'environment', has_api_key: false },
  });
  renderModal();
  await waitFor(() => expect(screen.getByLabelText(/Server addresses/)).toHaveValue(CHAT_SERVER));
  expect(screen.getByLabelText(/Server addresses/)).toBeDisabled();
  expect(screen.getByText(/MOTO_VLLM_BASE_URL/)).toBeInTheDocument();
});

test('a response that arrives after the window closed is ignored', async () => {
  let resolveTest;
  vllmAPI.test.mockReturnValue(new Promise((resolve) => { resolveTest = resolve; }));
  const handlers = { onClose: vi.fn(), onChanged: vi.fn(), onRefresh: vi.fn() };
  const { rerender } = render(
    <VllmConnectivityModal isOpen status={{}} capabilities={{ genericMode: false }} {...handlers} />
  );
  await waitFor(() => expect(vllmAPI.getStatus).toHaveBeenCalled());
  fireEvent.change(screen.getByLabelText(/Server addresses/), { target: { value: '127.0.0.1:8001' } });
  fireEvent.click(screen.getByRole('button', { name: 'Test Servers' }));

  rerender(<VllmConnectivityModal isOpen={false} status={{}} capabilities={{ genericMode: false }} {...handlers} />);
  resolveTest(PROBE);
  rerender(<VllmConnectivityModal isOpen status={{}} capabilities={{ genericMode: false }} {...handlers} />);
  await waitFor(() => expect(vllmAPI.getStatus).toHaveBeenCalledTimes(2));
  expect(screen.queryByText(/Nothing was saved yet/)).not.toBeInTheDocument();
});
