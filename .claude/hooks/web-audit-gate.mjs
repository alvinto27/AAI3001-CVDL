// Stop: if frontend files changed since the last passing audit, ask once for /design-audit before finishing.
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = join(dirname(fileURLToPath(import.meta.url)), '..');
const input = JSON.parse(await new Promise((res) => {
  let s = '';
  process.stdin.on('data', (c) => (s += c));
  process.stdin.on('end', () => res(s || '{}'));
}));
if (input.stop_hook_active) process.exit(0);

const stamp = (name) => (existsSync(join(dir, name)) ? Number(readFileSync(join(dir, name), 'utf8')) || 0 : 0);
if (stamp('.ui-dirty') > stamp('.design-audit-pass')) {
  console.log(JSON.stringify({
    decision: 'block',
    reason: 'Frontend files changed since the last passing design audit. Run /design-audit (cd app && npm run audit), fix any failures, and report the result before finishing.',
  }));
}
