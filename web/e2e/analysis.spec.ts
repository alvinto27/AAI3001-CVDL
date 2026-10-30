import { expect, shot, test, upload } from './fixtures';

test('a clean image gives a negative verdict', async ({ page }, info) => {
  await page.goto('/');
  await upload(page, 'clean-sample.png');
  await page.getByRole('button', { name: 'Analyse Image' }).click();
  await expect(page.getByRole('heading', { name: 'NO SIGNIFICANT ANOMALY DETECTED' })).toBeVisible();
  await shot(page, info, 'verdict-negative');
});

test('a stego image gives a positive verdict', async ({ page }, info) => {
  await page.goto('/');
  await upload(page, 'stego-sample.png');
  await page.getByRole('button', { name: 'Analyse Image' }).click();
  await expect(page.getByRole('heading', { name: /POSSIBLE HIDDEN DATA|SUSPICIOUS MODIFICATION DETECTED/ })).toBeVisible();
  await expect(page.getByRole('button', { name: /Inspect Suspicious Regions/ })).toBeEnabled();
  await shot(page, info, 'verdict-positive');
});

test('an unsupported file is rejected with an alert', async ({ page }, info) => {
  await page.goto('/');
  await upload(page, 'notes.png', Buffer.from('not an image at all'), 'image/png');
  await expect(page.getByRole('alert')).toContainText('Not a PNG, JPEG, or TIFF image.');
  await shot(page, info, 'error-unsupported');
});
