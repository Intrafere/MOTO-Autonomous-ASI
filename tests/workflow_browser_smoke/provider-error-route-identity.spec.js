import { expect, test } from './fixtures.js';

const ROUTE_IDENTITY = '[Autonomous · Submitter 2 · vendor/test-free:free via OpenRouter · host auto → TestHost]';
const IDENTITY_PARTS = ['Autonomous', 'Submitter 2', 'vendor/test-free:free', 'OpenRouter', 'TestHost'];
const CORRELATION_ID = 'provider-retry-route-identity';
const ERROR_SUMMARY = 'OpenRouter returned a successful response without completion choices (upstream 429: rate limited by TestHost)';
const RECOVERY_EVENTS = '/api/cloud-access/provider-recovery/events';

// Shaped like the record the backend stores and broadcasts when an Autonomous
// Submitter 2 OpenRouter call routed to an auto-selected upstream host returns
// no completion choices. Synthetic model and host names only.
const storedRetryRecord = (attempt, waitSeconds, { withMessage = true } = {}) => {
  const eventId = `provider_retry_cooldown:${CORRELATION_ID}:${attempt}`;
  const record = {
    id: eventId,
    notification_key: eventId,
    event_id: eventId,
    event_type: 'provider_retry_cooldown',
    legacy_event_type: 'provider_retry_cooldown',
    contract_version: 1,
    created_at: `2026-10-05T12:0${attempt}:00.000Z`,
    correlation_id: CORRELATION_ID,
    workflow_mode: 'autonomous',
    provider: 'openrouter',
    provider_label: 'OpenRouter',
    role_id: 'aggregator_submitter_2',
    model: 'vendor/test-free:free',
    role_label: 'Submitter 2',
    route_kind: 'primary',
    configured_host_provider: '',
    effective_host_provider: 'TestHost',
    host_routing: 'auto',
    provider_request_id: 'gen-test-request',
    route_identity: ROUTE_IDENTITY,
    reason: 'invalid_response',
    error_summary: ERROR_SUMMARY,
    recovery_action: 'retry',
    retry_layer: 'adapter',
    retry_attempt: attempt,
    max_attempts: 3,
    retry_after_seconds: waitSeconds,
    persistence_status: 'durable',
    persistence_degraded: false,
  };
  if (withMessage) {
    record.message = `${ROUTE_IDENTITY} OpenRouter server-side error: ${ERROR_SUMMARY}. `
      + `OpenRouter API call will retry after a ${waitSeconds}-second cooldown (retry ${attempt}).`;
  }
  return record;
};

async function expectAttributedRetryRows(page, expectedRetries) {
  const rows = page.locator('.activity-item').filter({ hasText: 'vendor/test-free:free' });
  await expect(rows).toHaveCount(expectedRetries.length);
  for (const [index, { attempt, waitSeconds }] of expectedRetries.entries()) {
    const message = rows.nth(index).locator('.activity-message');
    for (const part of IDENTITY_PARTS) {
      await expect(message).toContainText(part);
    }
    await expect(message).toContainText(ERROR_SUMMARY);
    await expect(message).toContainText(`${waitSeconds}-second cooldown (retry ${attempt})`);
    // The backend-composed identity leads the line exactly once, whether the
    // backend supplied the message or the frontend formatter composed it.
    const text = (await message.textContent()).trim();
    expect(text.startsWith(ROUTE_IDENTITY)).toBe(true);
    expect(text.split(ROUTE_IDENTITY)).toHaveLength(2);
  }
}

test('provider error lines show mode role model provider and host across reload and hydration', async ({ page, mockApp }) => {
  await mockApp.open();

  const first = { attempt: 1, waitSeconds: 2 };
  const second = { attempt: 2, waitSeconds: 4 };
  const missed = { attempt: 3, waitSeconds: 8 };
  await mockApp.sendWebSocket(
    'provider_retry_cooldown',
    storedRetryRecord(first.attempt, first.waitSeconds),
    '2026-10-05T12:01:00.000Z',
  );
  // No backend message: the frontend formatter composes the line and prefixes it.
  await mockApp.sendWebSocket(
    'provider_retry_cooldown',
    storedRetryRecord(second.attempt, second.waitSeconds, { withMessage: false }),
    '2026-10-05T12:02:00.000Z',
  );
  await expectAttributedRetryRows(page, [first, second]);

  await page.reload();
  await page.getByRole('button', { name: 'I Have Read and Acknowledge This Disclaimer' }).click();
  await expectAttributedRetryRows(page, [first, second]);

  // Reconnect hydration replays both shown records plus one this page missed;
  // shown attempts must not duplicate and the missed one keeps its identity.
  mockApp.state.responses.set(`GET ${RECOVERY_EVENTS}`, {
    success: true,
    notifications: [
      storedRetryRecord(first.attempt, first.waitSeconds),
      storedRetryRecord(second.attempt, second.waitSeconds, { withMessage: false }),
      storedRetryRecord(missed.attempt, missed.waitSeconds),
    ],
  });
  const hydrationRequestsBefore = mockApp.requests('GET', RECOVERY_EVENTS).length;
  // Dropping the mocked socket drives the app's own reconnect, which hydrates.
  await page.evaluate(() => window.WebSocket.instances.at(-1).close());
  await expect.poll(
    () => mockApp.requests('GET', RECOVERY_EVENTS).length,
    { timeout: 15_000 },
  ).toBeGreaterThan(hydrationRequestsBefore);
  await expectAttributedRetryRows(page, [first, second, missed]);

  await mockApp.assertNoUnexpectedRequests();
});
