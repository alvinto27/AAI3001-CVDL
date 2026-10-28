// LSB Inspector front end for the photo analysis API in cvdl/pipeline.
const API_URL = 'http://127.0.0.1:8000';
const HEALTH_INTERVAL_MS = 15000;
const MIN_SIDE = 256;
const MAX_PIXELS = 32000000;
const MAX_BYTES = 64 * 1024 * 1024;
// Longest side of the downloaded heatmap image, to keep large photos within browser canvas limits.
const DOWNLOAD_MAX_SIDE = 4096;

const $ = id => document.getElementById(id);
const sigmoid = x => 1 / (1 + Math.exp(-x));

let photoUrl = null;
let current = null; // { file, width, height, colourType, result }
let analysing = false;

// ---------- API status ----------

function setStatus(kind, title, text) {
  $('status').classList.toggle('off', kind === 'off');
  $('status').classList.toggle('checking', kind === 'checking');
  $('status-text').textContent = kind === 'ready' ? 'API ready' : kind === 'off' ? 'API offline' : 'Checking API…';
  $('offline-banner').hidden = kind !== 'off';
  if (kind === 'off') {
    $('offline-title').textContent = title;
    $('offline-text').innerHTML = text;
  }
}

async function checkHealth() {
  try {
    const response = await fetch(API_URL + '/health', { cache: 'no-store' });
    if (response.ok) setStatus('ready');
    else setStatus('off', 'The model isn\'t ready yet', 'The API is running but the model hasn\'t loaded. Wait a moment, then retry.');
    return response.ok;
  } catch {
    setStatus('off', 'Can\'t reach the analysis API', `Start it on this PC at <code>${API_URL.replace('http://', '')}</code>, then retry.`);
    return false;
  }
}

// ---------- File checks before upload ----------

function fileError(title, text) {
  $('file-error-title').textContent = title;
  $('file-error-text').textContent = text;
  $('file-error').hidden = false;
}

// Reads the PNG chunks before the image data and returns a reason the API would reject it, or the details it needs.
async function inspectPng(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const signature = [137, 80, 78, 71, 13, 10, 26, 10];
  if (bytes.length < 33 || signature.some((b, i) => bytes[i] !== b)) return { problem: 'It isn\'t a PNG image. Upload a PNG instead.' };
  const view = new DataView(bytes.buffer);
  const width = view.getUint32(16), height = view.getUint32(20);
  const bitDepth = bytes[24], colourType = bytes[25];
  if (colourType === 0 || colourType === 4) return { problem: 'It\'s a greyscale PNG. Upload an 8-bit RGB PNG instead.' };
  if (colourType === 3) return { problem: 'It uses a colour palette. Upload an 8-bit RGB PNG instead.' };
  if (bitDepth !== 8) return { problem: `It stores ${bitDepth}-bit colour. Upload an 8-bit RGB PNG instead.` };
  if (colourType !== 2 && colourType !== 6) return { problem: 'Its pixel format isn\'t supported. Upload an 8-bit RGB PNG instead.' };
  if (Math.min(width, height) < MIN_SIDE) return { problem: `It's ${width} × ${height} px. Both sides must be at least ${MIN_SIDE} px.` };
  if (width * height > MAX_PIXELS) return { problem: `It's ${(width * height / 1e6).toFixed(1)} megapixels. The limit is 32.` };

  let offset = 8;
  while (offset + 8 <= bytes.length) {
    const length = view.getUint32(offset);
    const type = String.fromCharCode(...bytes.subarray(offset + 4, offset + 8));
    if (type === 'IDAT' || type === 'IEND') break;
    if (type === 'acTL') return { problem: 'It\'s an animated PNG. Upload a still 8-bit RGB PNG instead.' };
    if (type === 'tRNS') return { problem: 'It marks a colour as transparent. Upload an 8-bit RGB PNG without transparency instead.' };
    offset += 12 + length;
  }
  return { width, height, colourType };
}

async function handleFile(file) {
  if (analysing) return;
  $('file-error').hidden = true;
  const title = `${file.name} can't be checked`;
  if (file.size > MAX_BYTES) return fileError(title, `It's ${(file.size / 1048576).toFixed(0)} MB. The limit is 64 MB.`);
  const info = await inspectPng(file);
  if (info.problem) return fileError(title, info.problem);
  current = { file, ...info, result: null };
  analyse();
}

// ---------- Analysis ----------

const API_ERRORS = {
  413: 'It\'s too large for the API. The limits are 64 MB and 32 megapixels.',
  415: 'The API can\'t read this image format. Upload an 8-bit RGB PNG.',
  422: 'The API couldn\'t read this image. It may be damaged.',
  503: 'The model isn\'t available right now. Check that the API has finished starting, then try again.'
};

async function analyse() {
  analysing = true;
  showResultView(true);
  const form = new FormData();
  form.append('photo', current.file);
  form.append('include_images', 'false');
  form.append('allow_rgba', current.colourType === 6 ? 'true' : 'false');
  const started = performance.now();
  let response;
  try {
    response = await fetch(API_URL + '/analyze', { method: 'POST', body: form });
  } catch {
    // The API can reject a busy request before reading the whole upload, which the browser reports as a network error.
    analysing = false;
    backToUpload();
    if (await checkHealth()) fileError('The upload didn\'t go through', 'The API may be checking another image. Try again in a few seconds.');
    return;
  }
  analysing = false;
  if (!response.ok) {
    backToUpload();
    if (response.status === 429) fileError('The API is busy', 'It\'s checking another image. Try again in a few seconds.');
    else fileError(`${current.file.name} can't be checked`, API_ERRORS[response.status] || 'Something went wrong in the API. Try again.');
    if (response.status === 503) checkHealth();
    return;
  }
  current.result = await response.json();
  current.seconds = (performance.now() - started) / 1000;
  showResult();
}

// ---------- Views ----------

function showResultView(loading) {
  $('upload').hidden = true;
  $('result').hidden = false;
  if (photoUrl) URL.revokeObjectURL(photoUrl);
  photoUrl = URL.createObjectURL(current.file);
  $('photo').src = photoUrl;
  const ratio = current.width / current.height;
  $('stack').style.aspectRatio = `${current.width} / ${current.height}`;
  $('stack').style.width = `min(100%, calc(70vh * ${ratio}))`;
  $('file-name').textContent = current.file.name;
  $('file-meta').textContent = `${current.width} × ${current.height} · ${formatSize(current.file.size)}`;

  $('veil').hidden = !loading;
  $('verdict').hidden = loading;
  $('verdict-skel').hidden = !loading;
  $('details').hidden = loading;
  $('actions').hidden = loading;
  $('controls').style.visibility = loading ? 'hidden' : 'visible';
  if (loading) {
    $('heat').style.opacity = 0;
    $('photo').style.filter = 'none';
  }
  window.scrollTo({ top: 0 });
}

function backToUpload() {
  $('result').hidden = true;
  $('upload').hidden = false;
  $('details').open = false;
  if (photoUrl) URL.revokeObjectURL(photoUrl);
  photoUrl = null;
  $('photo').removeAttribute('src');
}

function showResult() {
  const r = current.result;
  const detected = r.photo_verdict === 'message_detected';
  const score = r.photo_rule.probability;
  const cutoff = sigmoid(r.photo_rule.cutoff);

  $('verdict').className = 'card verdict ' + (detected ? 'flagged' : 'clear');
  $('tag').textContent = detected ? 'Flagged for review' : 'Nothing detected';
  $('v-title').textContent = detected ? 'Likely hidden data' : 'No hidden data found';
  $('v-sub').textContent = detected
    ? 'The pattern of least significant bits matches embedded data' + hotRegion(r.heatmap) + '.'
    : 'The least significant bits look like those of a normal photo.';
  $('v-note').textContent = detected
    ? 'A flag means the image needs a closer look. It isn\'t proof of a hidden message.'
    : 'Very short messages can still go unnoticed, so this isn\'t proof the image is clean.';
  $('score-val').textContent = score.toFixed(2);
  $('fill').style.width = (score * 100) + '%';
  $('cut').style.left = (cutoff * 100) + '%';
  $('cut-label').style.left = (cutoff * 100) + '%';

  const flaggedTiles = r.tiles.filter(t => t.flagged).length;
  const rows = [
    ['Tiles flagged', `${flaggedTiles} of ${r.tiles.length}`],
    ['Photo score', score.toFixed(2)],
    ['Cutoff', cutoff.toFixed(2)],
    ['Model', escapeHtml(`${r.bundle.name}, v${r.bundle.version}`)],
    ['Image', `${r.width} × ${r.height} ${current.colourType === 6 ? 'RGBA' : 'RGB'}`],
    ['Analysis time', `${current.seconds.toFixed(1)} s on ${escapeHtml(r.device.toUpperCase())}`]
  ];
  if (current.colourType === 6) rows.push(['Alpha channel', '<span>Not checked</span>', 'warn']);
  // The API's own alpha warning repeats the row above, so it is left out.
  for (const warning of r.input.warnings || []) {
    if (!/alpha/i.test(warning)) rows.push(['Note', escapeHtml(warning), 'warn']);
  }
  $('details-list').innerHTML = rows.map(([k, v, cls]) => `<dt>${k}</dt><dd class="${cls || ''}">${v}</dd>`).join('');

  showResultView(false);
  drawHeatmap(r.heatmap, r.width, r.height);
  setStrength(+$('strength').value);
}

// Names the part of the image where flagged cells sit, for the verdict sentence.
function hotRegion(heatmap) {
  let sumX = 0, sumY = 0, count = 0;
  heatmap.flagged.forEach((row, y) => row.forEach((on, x) => { if (on) { sumX += x; sumY += y; count++; } }));
  if (!count) return '';
  const [rows, cols] = heatmap.shape;
  const share = count / (rows * cols);
  if (share > .6) return ' across most of the image';
  const fy = (sumY / count + .5) / rows, fx = (sumX / count + .5) / cols;
  const v = fy < 1 / 3 ? 'top' : fy > 2 / 3 ? 'bottom' : 'middle';
  const h = fx < 1 / 3 ? 'left' : fx > 2 / 3 ? 'right' : '';
  const place = v === 'middle' && !h ? 'centre' : h ? `${v} ${h}` : v;
  return `, mostly in the ${place} of the image`;
}

// ---------- Heatmap ----------

const RAMP = [[83, 74, 183], [212, 83, 126], [239, 159, 39], [250, 236, 140]];

// Same display scale as the API's own heatmap image: sigmoid of the raw cell value minus the cell cutoff.
// Cells well below the cutoff fade to transparent so only suspicious areas are coloured.
function cellColour(value, cutoff) {
  const d = sigmoid(value - cutoff);
  const t = Math.min(.999, d) * (RAMP.length - 1), i = Math.floor(t), f = t - i;
  const [r, g, b] = RAMP[i].map((a, k) => Math.round(a + (RAMP[i + 1][k] - a) * f));
  const alpha = Math.min(.92, Math.max(0, (d - .2) / .4));
  return `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
}

function paintCells(ctx, heatmap, scale) {
  const { values, x_edges: xs, y_edges: ys, cutoff } = heatmap;
  ctx.imageSmoothingEnabled = false;
  values.forEach((row, y) => row.forEach((value, x) => {
    const x0 = Math.round(xs[x] * scale), x1 = Math.round(xs[x + 1] * scale);
    const y0 = Math.round(ys[y] * scale), y1 = Math.round(ys[y + 1] * scale);
    ctx.fillStyle = cellColour(value, cutoff);
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  }));
}

function drawHeatmap(heatmap, width, height) {
  // Capping the canvas at 2048 px keeps it light for 32 MP photos while each 32 px cell stays at least 9 px wide.
  const scale = Math.min(1, 2048 / Math.max(width, height));
  const canvas = $('heat');
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  paintCells(ctx, heatmap, scale);
}

function setStrength(value) {
  $('heat').style.opacity = value / 100;
  $('photo').style.filter = `brightness(${1 - .45 * value / 100}) saturate(${1 - .5 * value / 100})`;
  $('strength-out').textContent = value + '%';
}

async function downloadHeatmap() {
  const { width, height, heatmap } = current.result;
  const scale = Math.min(1, DOWNLOAD_MAX_SIDE / Math.max(width, height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext('2d');
  const strength = +$('strength').value / 100;
  ctx.filter = `brightness(${1 - .45 * strength}) saturate(${1 - .5 * strength})`;
  ctx.drawImage($('photo'), 0, 0, canvas.width, canvas.height);
  ctx.filter = 'none';
  ctx.globalAlpha = strength;
  paintCells(ctx, heatmap, scale);
  canvas.toBlob(blob => {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = current.file.name.replace(/\.png$/i, '') + '-heatmap.png';
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }, 'image/png');
}

// ---------- Helpers ----------

function formatSize(bytes) {
  return bytes >= 1048576 ? (bytes / 1048576).toFixed(1) + ' MB' : Math.round(bytes / 1024) + ' KB';
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', '\'': '&#39;' })[c]);
}

// ---------- Events ----------

$('choose').addEventListener('click', () => $('file-input').click());
$('file-input').addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
  e.target.value = '';
});

let dragDepth = 0;
function setDragging(on) {
  $('drop').classList.toggle('over', on);
  $('drop-title').textContent = on ? 'Release to check this image' : 'Drop a PNG here';
  $('drop-sub').textContent = on ? 'One PNG at a time' : 'or choose a file from your computer';
  $('choose').hidden = on;
}
// Listen on the whole page so a drop outside the box doesn't make the browser open the file.
document.addEventListener('dragenter', e => {
  e.preventDefault();
  if (!$('upload').hidden && ++dragDepth === 1) setDragging(true);
});
document.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; setDragging(false); } });
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => {
  e.preventDefault();
  dragDepth = 0;
  setDragging(false);
  if ($('upload').hidden) return;
  const files = e.dataTransfer.files;
  if (files.length > 1) return fileError('One image at a time', 'Drop a single PNG.');
  if (files[0]) handleFile(files[0]);
});

$('strength').addEventListener('input', e => setStrength(+e.target.value));
$('retry').addEventListener('click', () => { setStatus('checking'); checkHealth(); });
$('new-top').addEventListener('click', () => { if (!analysing) backToUpload(); });
$('again').addEventListener('click', backToUpload);
$('download').addEventListener('click', downloadHeatmap);

checkHealth();
setInterval(checkHealth, HEALTH_INTERVAL_MS);
