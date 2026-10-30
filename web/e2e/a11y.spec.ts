import { axeScan, test, upload } from './fixtures';

test('landing has no serious accessibility violations', async ({ page }, info) => {
  await page.goto('/');
  await page.getByText('Model ready').waitFor();
  await axeScan(page, info, 'landing');
});

test('staged image has no serious accessibility violations', async ({ page }, info) => {
  await page.goto('/');
  await upload(page, 'clean-sample.png');
  await page.getByRole('button', { name: 'Analyse Image' }).waitFor();
  await axeScan(page, info, 'staged');
});

test('results have no serious accessibility violations', async ({ page }, info) => {
  await page.goto('/');
  await upload(page, 'stego-sample.png');
  await page.getByRole('button', { name: 'Analyse Image' }).click();
  await page.getByRole('button', { name: /Inspect Suspicious Regions/ }).waitFor();
  await axeScan(page, info, 'results');
});
