import { expect, shot, test } from './fixtures';

test('landing page lays out cleanly', async ({ page }, info) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: /Inspect an image for hidden data/ })).toBeVisible();
  await expect(page.getByText('Example: real model output on a sample photo')).toBeVisible();
  await expect(page.getByRole('button', { name: /Drop an image here/ })).toBeVisible();
  for (const label of ['Formats', 'Pixels', 'Limits', 'Best input']) await expect(page.getByText(label, { exact: true })).toBeVisible();
  for (const cap of ['LSB inspection', 'Spatial anomaly localization', 'Forensic evidence review']) await expect(page.getByText(cap)).toBeVisible();
  await expect(page.getByText('Model ready')).toBeVisible();

  const { sw, cw, sh, ih } = await page.evaluate(() => ({
    sw: document.documentElement.scrollWidth,
    cw: document.documentElement.clientWidth,
    sh: document.querySelector('main > div')!.scrollHeight,
    ih: document.querySelector('main')!.clientHeight,
  }));
  expect(sw, 'no horizontal overflow').toBeLessThanOrEqual(cw);
  if (info.project.name === 'desktop') expect(sh, 'desktop workspace fits the viewport without scrolling').toBeLessThanOrEqual(ih + 1);
  await shot(page, info, 'landing');
});
