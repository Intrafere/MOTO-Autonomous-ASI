import { expect, test } from './fixtures.js';

test('overflow activity attributes routes, keeps proof overflow nonfatal, dedupes stop, and persists', async ({ page, mockApp }) => {
  await mockApp.open();

  const proofMessage = 'Proof candidate exceeded its direct-context budget';
  await mockApp.sendWebSocket('proof_context_overflow', {
    workflow_mode: 'autonomous',
    role_id: 'autonomous_proof_formalization',
    message: proofMessage,
    configured_model: 'configured-proof-model',
    configured_provider: 'openrouter',
    effective_model: 'rotated-proof-model',
    effective_provider: 'openrouter',
    effective_host_provider: 'Provider A',
  });

  await expect(page.getByText(new RegExp(`${proofMessage}.*Effective route: rotated-proof-model via openrouter, host Provider A.*Configured route: configured-proof-model via openrouter`))).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start Research' })).toBeVisible();

  const terminalMessage = 'Autonomous direct context exceeded the selected model window';
  await mockApp.sendWebSocket('context_overflow_error', {
    workflow_mode: 'autonomous',
    role_id: 'agg_sub1_topic',
    message: terminalMessage,
    configured_model: 'configured-topic-model',
    configured_provider: 'openrouter',
    effective_model: 'fallback-topic-model',
    effective_provider: 'lm_studio',
  }, '2026-07-14T12:01:00.000Z');
  await mockApp.sendWebSocket('auto_research_stopped', {
    reason: 'context_overflow',
    message: terminalMessage,
  }, '2026-07-14T12:01:01.000Z');

  const terminalActivity = page.locator('.activity-message').filter({ hasText: terminalMessage });
  await expect(terminalActivity).toHaveCount(1);
  await expect(terminalActivity).toContainText('Effective route: fallback-topic-model via lm_studio');
  await expect(terminalActivity).toContainText('Configured route: configured-topic-model via openrouter');

  await page.reload();
  await page.getByRole('button', { name: 'I Have Read and Acknowledge This Disclaimer' }).click();
  await expect(page.locator('.activity-message').filter({ hasText: proofMessage })).toHaveCount(1);
  await expect(page.locator('.activity-message').filter({ hasText: terminalMessage })).toHaveCount(1);

  await mockApp.assertNoUnexpectedRequests();
});
