import { expect, test } from './fixtures.js';

test('provider retry attempts remain distinct, attributed, secret-safe, and reloadable', async ({ page, mockApp }) => {
  await mockApp.open();

  const correlationId = 'provider-retry:openrouter:agg_val:vendor-model';
  const base = {
    event_type: 'provider_retry_cooldown',
    correlation_id: correlationId,
    workflow_mode: 'autonomous',
    provider: 'openrouter',
    provider_label: 'OpenRouter',
    role_id: 'agg_val',
    model: 'vendor/model',
    retry_after_seconds: 60,
    reason: 'http_error',
    error_summary: 'HTTP 503 upstream unavailable',
  };
  await mockApp.sendWebSocket('provider_retry_cooldown', {
    ...base,
    event_id: `${correlationId}:1`,
    retry_attempt: 1,
  }, '2026-10-05T12:00:00.000Z');
  await mockApp.sendWebSocket('provider_retry_cooldown', {
    ...base,
    event_id: `${correlationId}:2`,
    retry_attempt: 2,
    retry_after_seconds: 120,
  }, '2026-10-05T12:01:00.000Z');

  const retryItems = page.locator('.activity-item').filter({ hasText: 'OpenRouter' });
  await expect(retryItems).toHaveCount(2);
  await expect(retryItems.nth(0)).toContainText('60-second cooldown');
  await expect(retryItems.nth(0)).toContainText('retry 1');
  await expect(retryItems.nth(1)).toContainText('120-second cooldown');
  await expect(retryItems.nth(1)).toContainText('retry 2');
  await expect(page.getByText(/Bearer|secret-value/)).toHaveCount(0);

  await page.reload();
  await page.getByRole('button', { name: 'I Have Read and Acknowledge This Disclaimer' }).click();
  await expect(page.locator('.activity-item').filter({ hasText: 'OpenRouter' })).toHaveCount(2);

  await mockApp.assertNoUnexpectedRequests();
});
