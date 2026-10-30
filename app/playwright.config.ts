import { defineConfig } from '@playwright/test';

// Tests never touch the real model: a mock API (same routes, seeded scores) is built into the app via VITE_API_URL.
const API_PORT = 8765;
const APP_PORT = 4173;

export default defineConfig({
  testDir: './e2e',
  outputDir: './audit/artifacts',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list'], ['json', { outputFile: 'audit/results.json' }], ['html', { outputFolder: 'audit/html', open: 'never' }]],
  use: {
    baseURL: `http://127.0.0.1:${APP_PORT}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'desktop', use: { viewport: { width: 1920, height: 1080 } } },
    { name: 'tablet', use: { viewport: { width: 820, height: 1180 }, hasTouch: true } },
    { name: 'mobile', use: { viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true } },
  ],
  webServer: [
    {
      command: 'node scripts/mock-api.mjs',
      url: `http://127.0.0.1:${API_PORT}/health`,
      env: { PORT: String(API_PORT), DELAY_MS: '100' },
      reuseExistingServer: false,
    },
    {
      command: `npx vite build && npx vite preview --port ${APP_PORT}`,
      url: `http://127.0.0.1:${APP_PORT}`,
      env: { VITE_API_URL: `http://127.0.0.1:${API_PORT}` },
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
});
