import { render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import CompilerLogs from './CompilerLogs';
import { normalizeActivityEvent } from '../../utils/activityNormalization';
import { formatProviderRetryCooldownMessage } from '../../utils/activityStyles';

vi.mock('../../services/api', () => ({
  compilerAPI: {
    getMetrics: vi.fn().mockResolvedValue({
      data: {
        construction: { acceptances: 0, rejections: 0, declines: 0, acceptance_rate: 0 },
        rigor: { acceptances: 0, rejections: 0, declines: 0, acceptance_rate: 0 },
        outline: { acceptances: 0, rejections: 0, declines: 0 },
        review: { acceptances: 0, rejections: 0, declines: 0 },
        minuscule_edit_count: 0,
        paper_word_count: 0,
        total_submissions: 0,
      },
    }),
    getStatus: vi.fn().mockResolvedValue({ data: { is_running: true, current_mode: 'construction' } }),
  },
}));

vi.mock('../../services/websocket', () => ({
  websocket: {
    on: vi.fn(() => () => {}),
    off: vi.fn(),
  },
}));

describe('CompilerLogs provider activity', () => {
  const previousFetch = globalThis.fetch;

  beforeEach(() => {
    localStorage.clear();
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: false });
  });

  afterEach(() => {
    globalThis.fetch = previousFetch;
  });

  test('renders a provider retry cooldown item as its message instead of raw JSON', async () => {
    const routeIdentity = '[Compiler · Writing Submitter · vendor/test-free:free via OpenRouter · host TestHost (pinned)]';
    const notification = {
      event_type: 'provider_retry_cooldown',
      notification_key: 'provider-retry:compiler_writer:test',
      workflow_mode: 'compiler',
      role_id: 'compiler_writer',
      model: 'vendor/test-free:free',
      provider: 'openrouter',
      provider_label: 'OpenRouter',
      host_routing: 'pinned',
      configured_host_provider: 'TestHost',
      effective_host_provider: 'TestHost',
      route_identity: routeIdentity,
      retry_attempt: 1,
      retry_after_seconds: 2,
      reason: 'invalid_response',
      error_summary: 'OpenRouter returned a successful response without completion choices.',
    };
    const message = formatProviderRetryCooldownMessage(notification);
    const providerItem = normalizeActivityEvent({
      event: 'provider_retry_cooldown',
      type: 'provider_retry_cooldown',
      timestamp: '2026-10-05T12:00:01Z',
      message,
      data: notification,
    });

    const { container } = render(<CompilerLogs providerActivity={[providerItem]} />);

    expect(message.startsWith(routeIdentity)).toBe(true);
    expect(await screen.findByText(message)).toBeInTheDocument();
    expect(container.textContent).not.toContain('"retry_attempt"');
    expect(container.textContent).not.toContain('"route_identity"');
  });
});
