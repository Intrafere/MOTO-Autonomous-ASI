import { expect, test } from './fixtures.js';

test('mode switching is in-memory and LeanOJ remains developer gated', async ({ page, mockApp }) => {
  await mockApp.open();

  const modeSelect = page.locator('#app-mode-select');
  await expect(modeSelect).toHaveValue('autonomous');
  await expect(modeSelect.getByRole('option', { name: 'LeanOJ Proof Solver' })).toHaveCount(0);

  await modeSelect.selectOption('manual');
  await expect(page.getByText('Manual S.T.E.M. Writer')).toBeVisible();

  await page.keyboard.down('Shift');
  await page.keyboard.down('z');
  await page.keyboard.down('x');
  await page.keyboard.up('x');
  await page.keyboard.up('z');
  await page.keyboard.up('Shift');

  await expect(modeSelect.getByRole('option', { name: 'LeanOJ Proof Solver' })).toHaveCount(1);
  await modeSelect.selectOption('leanoj');
  await expect(page.getByText('Proof Solver Mode', { exact: true })).toBeVisible();

  await page.reload();
  await expect(page.getByRole('button', { name: 'I Have Read and Acknowledge This Disclaimer' })).toBeVisible();
  await page.getByRole('button', { name: 'I Have Read and Acknowledge This Disclaimer' }).click();
  await expect(page.locator('#app-mode-select')).toHaveValue('autonomous');
  await expect(page.getByRole('heading', { name: 'Autonomous Research' })).toBeVisible();

  await mockApp.assertNoUnexpectedRequests();
});
