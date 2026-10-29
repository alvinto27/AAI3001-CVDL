// Stand-in for cvdl.pipeline.serve: same routes and response schema, synthetic scores.
// For UI development and screenshots only. Scenario comes from the upload's file name:
//   *clean*  -> not_detected          *stego*  -> diffuse detection
//   *border* -> score near cutoff     anything else -> localised lower-right cluster
import { createServer } from 'node:http';

const PORT = Number(process.env.PORT ?? 8000);
const CELL = 32;
const TILE = 256;
const CELL_CUTOFF = 0.1966136991977692;
const RULE_CUTOFF = -0.05636374639150742;
const TILE_CUTOFF = 0.5094263553619385;
const BUNDLE = { name: 'run18-rule-c-v1', version: 1, format_version: 1, weights_sha256: 'mock' };

const cors = {
  'access-control-allow-origin': '*',
  'access-control-allow-methods': 'GET,POST',
  'access-control-allow-headers': 'Content-Type',
};

const edges = (n) => {
  const out = [];
  for (let v = 0; v < n; v += CELL) out.push(v);
  out.push(n);
  return out;
};

function dimensions(buf) {
  const png = buf.indexOf(Buffer.from([0x89, 0x50, 0x4e, 0x47]));
  if (png >= 0) return { w: buf.readUInt32BE(png + 16), h: buf.readUInt32BE(png + 20), format: 'PNG' };
  const jpg = buf.indexOf(Buffer.from([0xff, 0xd8, 0xff]));
  if (jpg >= 0) {
    for (let i = jpg + 2; i < buf.length - 9; ) {
      if (buf[i] !== 0xff) { i++; continue; }
      const m = buf[i + 1];
      if (m >= 0xc0 && m <= 0xc3) return { h: buf.readUInt16BE(i + 5), w: buf.readUInt16BE(i + 7), format: 'JPEG' };
      i += 2 + buf.readUInt16BE(i + 2);
    }
  }
  return { w: 1024, h: 768, format: 'PNG' };
}

let seed = 7;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);

function analyse(name, buf) {
  seed = 7;
  const { w, h, format } = dimensions(buf);
  const xe = edges(w), ye = edges(h);
  const rows = ye.length - 1, cols = xe.length - 1;
  const scenario = /clean/i.test(name) ? 'clean' : /stego/i.test(name) ? 'stego' : /border/i.test(name) ? 'border' : 'local';

  const blobs = scenario === 'local'
    ? [{ cx: 0.78 * cols, cy: 0.74 * rows, rx: Math.max(2, cols * 0.09), ry: Math.max(2, rows * 0.1), a: 3.2 },
       { cx: 0.2 * cols, cy: 0.25 * rows, rx: Math.max(1.5, cols * 0.04), ry: Math.max(1.5, rows * 0.05), a: 1.8 }]
    : [];

  const values = [], flagged = [];
  for (let r = 0; r < rows; r++) {
    const vr = [], fr = [];
    for (let c = 0; c < cols; c++) {
      let v = -1.6 + (rand() - 0.5) * 0.9;
      if (scenario === 'stego') v = 0.5 + (rand() - 0.5) * 1.4;
      if (scenario === 'border') v = -0.3 + (rand() - 0.5) * 1.1;
      for (const b of blobs) {
        const d = ((c - b.cx) / b.rx) ** 2 + ((r - b.cy) / b.ry) ** 2;
        v += b.a * Math.exp(-d);
      }
      vr.push(v); fr.push(v >= CELL_CUTOFF);
    }
    values.push(vr); flagged.push(fr);
  }

  const ox = [], oy = [];
  for (let v = 0; v + TILE < w; v += TILE) ox.push(v);
  ox.push(Math.max(0, w - TILE));
  for (let v = 0; v + TILE < h; v += TILE) oy.push(v);
  oy.push(Math.max(0, h - TILE));
  const tiles = [];
  for (const y of oy) for (const x of ox) {
    const r0 = Math.floor(y / CELL), r1 = Math.min(rows, Math.ceil((y + TILE) / CELL));
    const c0 = Math.floor(x / CELL), c1 = Math.min(cols, Math.ceil((x + TILE) / CELL));
    let sum = 0, n = 0;
    for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) { sum += values[r][c]; n++; }
    const p = 1 / (1 + Math.exp(-(sum / n - CELL_CUTOFF) * 1.6));
    tiles.push({ box: [x, y, x + TILE, y + TILE], probability: p, flagged: p >= TILE_CUTOFF });
  }

  const flaggedShare = flagged.flat().filter(Boolean).length / (rows * cols);
  const logit = { clean: -3.3, stego: 2.4, border: 0.15, local: 1.9 }[scenario];
  const detected = logit >= RULE_CUTOFF;
  const warnings = format === 'JPEG' ? ['JPEG input: decoded RGB analysed. Compression can destroy RGB-LSB changes.'] : [];

  return {
    schema_version: 1, bundle: BUNDLE, device: 'cpu', width: w, height: h,
    input: { format, orientation: 1, warnings },
    photo_verdict: detected ? 'message_detected' : 'not_detected',
    photo_rule: { type: 'logistic_v1', logit, probability: 1 / (1 + Math.exp(-logit)), cutoff: RULE_CUTOFF, probability_note: 'sigmoid score, not a calibrated real-world probability' },
    features: { flagged_share: flaggedShare }, counts: { tiles: tiles.length },
    tile_cutoff: TILE_CUTOFF, tiles,
    heatmap: { kind: 'trained_head', score: 'logit', cell_size: CELL, shape: [rows, cols], x_edges: xe, y_edges: ye, cutoff: CELL_CUTOFF, values, flagged },
  };
}

createServer((req, res) => {
  if (req.method === 'OPTIONS') { res.writeHead(204, cors).end(); return; }
  if (req.url === '/health') {
    res.writeHead(200, { ...cors, 'content-type': 'application/json' });
    res.end(JSON.stringify({ ready: true, bundle: BUNDLE, device: 'cpu' }));
    return;
  }
  if (req.url === '/analyze' && req.method === 'POST') {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      const buf = Buffer.concat(chunks);
      const name = /filename="([^"]*)"/.exec(buf.subarray(0, 600).toString('latin1'))?.[1] ?? '';
      setTimeout(() => {
        res.writeHead(200, { ...cors, 'content-type': 'application/json' });
        res.end(JSON.stringify(analyse(name, buf)));
      }, Number(process.env.DELAY_MS ?? 1800));
    });
    return;
  }
  res.writeHead(404, cors).end();
}).listen(PORT, '127.0.0.1', () => console.log(`mock API on http://127.0.0.1:${PORT}`));
