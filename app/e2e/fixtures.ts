import { deflateSync } from 'node:zlib';
import { mkdirSync, writeFileSync } from 'node:fs';
import { test as base, expect, type Page, type TestInfo } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const CRC = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
const crc32 = (b: Buffer) => {
  let c = 0xffffffff;
  for (const x of b) c = CRC[(c ^ x) & 0xff]! ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};
const chunk = (type: string, data: Buffer) => {
  const body = Buffer.concat([Buffer.from(type), data]);
  const out = Buffer.alloc(body.length + 8);
  out.writeUInt32BE(data.length, 0);
  body.copy(out, 4);
  out.writeUInt32BE(crc32(body), body.length + 4);
  return out;
};

/** A valid 8-bit RGB PNG. The mock API picks its verdict from the file name, so pixels only need to be valid. */
export function pngBuffer(size = 512): Buffer {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  const rows = Buffer.alloc((size * 3 + 1) * size);
  for (let y = 0; y < size; y++) {
    const o = y * (size * 3 + 1);
    for (let x = 0; x < size * 3; x++) rows[o + 1 + x] = (x * 7 + y * 3) & 0xff;
  }
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(rows)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

export const upload = (page: Page, name: string, buffer: Buffer = pngBuffer(), mimeType = 'image/png') =>
  page.locator('input[type=file]').setInputFiles({ name, mimeType, buffer });

/** Full-page screenshot into audit/screens, one file per project and state. */
export async function shot(page: Page, info: TestInfo, state: string) {
  await page.waitForTimeout(450); // let entrance animations finish so the evidence is not captured mid-fade
  mkdirSync('audit/screens', { recursive: true });
  await page.screenshot({ path: `audit/screens/${info.project.name}-${state}.png`, fullPage: true });
}

/** WCAG A/AA scan. Saves every violation to audit/axe; fails the test on serious or critical ones only. */
export async function axeScan(page: Page, info: TestInfo, state: string) {
  const { violations } = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze();
  mkdirSync('audit/axe', { recursive: true });
  writeFileSync(
    `audit/axe/${info.project.name}-${state}.json`,
    JSON.stringify(violations.map((v) => ({ id: v.id, impact: v.impact, help: v.help, nodes: v.nodes.map((n) => n.target.join(' ')) })), null, 2),
  );
  const blocking = violations.filter((v) => v.impact === 'serious' || v.impact === 'critical');
  expect(blocking.map((v) => `${v.id} (${v.impact}): ${v.nodes.map((n) => n.target.join(' ')).join(', ')}`)).toEqual([]);
}

export const test = base;
export { expect };
