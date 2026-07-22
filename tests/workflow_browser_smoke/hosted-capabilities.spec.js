import { expect, test } from './fixtures.js';

test('hosted capabilities drive startup copy and hide desktop-only paths', async ({ page, mockApp }) => {
  mockApp.state.hasOpenRouterKey = false;
  await mockApp.open({ acknowledgeDisclaimer: false });

  await expect(page.getByText('This hosted deployment uses OpenRouter-only inference.')).toBeVisible();
  await expect(page.getByText('LM Studio is intentionally disabled in this environment.')).toBeVisible();

  await page.getByRole('button', { name: 'I Have Read and Acknowledge This Disclaimer' }).click();

  await expect(page.getByRole('heading', { name: 'Choose Your Startup Setup' })).toBeVisible();
  await expect(page.getByText('This hosted deployment needs an OpenRouter API key before you start.')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Enter OpenRouter Key' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'LM Studio Setup' })).toHaveCount(0);
  await expect(page.getByText('Cloud Provider Add-On')).toHaveCount(0);

  await mockApp.assertNoUnexpectedRequests();
});
