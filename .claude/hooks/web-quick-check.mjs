// PostToolUse (Edit|Write): after a frontend edit, mark the UI dirty and run the fast type-check. No browser.
import { spawnSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const input = JSON.parse(await new Promise((res) => {
  let s = '';
  process.stdin.on('data', (c) => (s += c));
  process.stdin.on('end', () => res(s || '{}'));
}));

const file = String(input.tool_input?.file_path ?? '').replace(/\\/g, '/');
if (!/\/web\/(src|e2e)\/.+\.(ts|tsx|css)$/.test(file)) process.exit(0);

mkdirSync(join(root, '.claude'), { recursive: true });
writeFileSync(join(root, '.claude', '.ui-dirty'), String(Date.now()));

if (!/\/web\/src\//.test(file)) process.exit(0);
const r = spawnSync('npx tsc --noEmit', { cwd: join(root, 'web'), shell: true, encoding: 'utf8', timeout: 60_000 });
if (r.status !== 0) {
  process.stderr.write(`Type-check failed after editing ${file}:\n${(r.stdout + r.stderr).slice(0, 2000)}\n`);
  process.exit(2);
}
