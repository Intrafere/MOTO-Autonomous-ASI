import { expect, test } from './fixtures.js';

test('autonomous start sends hosted payload, locks controls, and stops cleanly', async ({ page, mockApp }) => {
  await mockApp.open();

  const prompt = page.getByLabel('Research Goal');
  await prompt.fill('Prove a deterministic browser smoke theorem.');
  await page.getByRole('button', { name: 'Start Research' }).click();

  await expect.poll(() => mockApp.requests('POST', '/api/auto-research/start').length).toBe(1);
  const startPayload = mockApp.requests('POST', '/api/auto-research/start')[0].body;
  expect(startPayload.user_research_prompt).toBe('Prove a deterministic browser smoke theorem.');
  expect(startPayload.allow_mathematical_proofs).toBe(false);
  expect(startPayload.allow_research_papers).toBe(true);
  expect(startPayload.submitter_configs.length).toBeGreaterThan(0);
  expect(startPayload.submitter_configs.every((role) => role.provider === 'openrouter')).toBe(true);
  expect(startPayload.submitter_configs.every((role) => role.lm_studio_fallback_id === null)).toBe(true);
  expect(startPayload.validator_provider).toBe('openrouter');
  expect(startPayload.writer_provider).toBe('openrouter');
  expect(startPayload.high_param_provider).toBe('openrouter');

  await expect(prompt).toBeDisabled();
  await expect(page.getByText('Running', { exact: true })).toBeVisible();
  const stopButton = page.getByRole('button', { name: 'Stop Research' });
  await expect(stopButton).toBeVisible();
  await stopButton.click();

  await expect.poll(() => mockApp.requests('POST', '/api/auto-research/stop').length).toBe(1);
  await expect(page.getByRole('button', { name: 'Start Research' })).toBeVisible();
  await expect(prompt).toBeEnabled();

  await mockApp.assertNoUnexpectedRequests();
});
