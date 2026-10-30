// Turns Playwright's JSON results, axe findings and screenshots into audit/report.md.
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';

const results = existsSync('audit/results.json') ? JSON.parse(readFileSync('audit/results.json', 'utf8')) : null;
const rows = [];
const walk = (suite) => {
  for (const spec of suite.specs ?? []) for (const t of spec.tests) rows.push({ project: t.projectName, title: spec.title, status: t.status });
  for (const s of suite.suites ?? []) walk(s);
};
for (const s of results?.suites ?? []) walk(s);

const failed = rows.filter((r) => r.status !== 'expected' && r.status !== 'skipped');
const axeFiles = existsSync('audit/axe') ? readdirSync('audit/axe').filter((f) => f.endsWith('.json')) : [];
const shots = existsSync('audit/screens') ? readdirSync('audit/screens').filter((f) => f.endsWith('.png')) : [];

let md = `# Design audit report\n\n${failed.length ? `**FAIL**: ${failed.length} of ${rows.length} checks failed.` : `**PASS**: ${rows.length} checks passed.`}\n\n`;
md += '## Checks\n\n| Viewport | Check | Result |\n| --- | --- | --- |\n';
for (const r of rows) md += `| ${r.project} | ${r.title} | ${r.status === 'expected' ? 'pass' : r.status} |\n`;
md += '\n## Accessibility findings (axe, WCAG A/AA)\n\nSerious and critical findings fail the run; the rest are advisory.\n\n';
let any = false;
for (const f of axeFiles) {
  for (const x of JSON.parse(readFileSync(`audit/axe/${f}`, 'utf8'))) {
    any = true;
    md += `- \`${f.replace('.json', '')}\` **${x.impact}** ${x.id}: ${x.help} (${x.nodes.slice(0, 3).join('; ')})\n`;
  }
}
if (!any) md += 'None.\n';
md += `\n## Screenshots\n\n${shots.map((s) => `- audit/screens/${s}`).join('\n') || 'None captured.'}\n`;
writeFileSync('audit/report.md', md);
console.log(md.split('\n').slice(0, 3).join('\n'));
process.exitCode = failed.length || !rows.length ? 1 : 0;
