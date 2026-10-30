// Full design audit: Playwright (3 viewports + axe), then the report. A pass writes the stamp the Stop hook looks for.
import { spawnSync } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';

rmSync('audit', { recursive: true, force: true });
mkdirSync('audit', { recursive: true });

const tests = spawnSync('npx playwright test', { stdio: 'inherit', shell: true }).status ?? 1;
const report = spawnSync('node', ['scripts/audit-report.mjs'], { stdio: 'inherit' }).status ?? 1;
if (tests === 0 && report === 0) {
  mkdirSync('../.claude', { recursive: true });
  writeFileSync('../.claude/.design-audit-pass', String(Date.now()));
  console.log('Audit passed. Report: app/audit/report.md');
} else {
  console.log('Audit FAILED. Report: app/audit/report.md');
  process.exitCode = 1;
}
