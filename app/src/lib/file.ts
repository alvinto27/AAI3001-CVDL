import type { FileMeta } from '../types/analysis';

export const MIN_SIDE = 256;
export const MAX_PIXELS = 32_000_000;
export const MAX_BYTES = 64 * 1024 * 1024;

export class FileProblem extends Error {}

const PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];

/** Reads PNG chunks ahead of the pixel data and rejects what the service would. */
function inspectPng(bytes: Uint8Array): FileMeta {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const width = view.getUint32(16);
  const height = view.getUint32(20);
  const bitDepth = bytes[24];
  const colourType = bytes[25];
  if (colourType === 0 || colourType === 4) throw new FileProblem('Greyscale PNG. Upload an 8-bit RGB image.');
  if (colourType === 3) throw new FileProblem('Palette PNG. Upload an 8-bit RGB image.');
  if (bitDepth !== 8) throw new FileProblem(`${bitDepth}-bit PNG. Upload an 8-bit RGB image.`);

  let offset = 8;
  while (offset + 8 <= bytes.length) {
    const length = view.getUint32(offset);
    const type = String.fromCharCode(...bytes.subarray(offset + 4, offset + 8));
    if (type === 'IDAT' || type === 'IEND') break;
    if (type === 'acTL') throw new FileProblem('Animated PNG. Upload a still image.');
    if (type === 'tRNS') throw new FileProblem('PNG with colour-key transparency. Upload one without.');
    offset += 12 + length;
  }
  return { width, height, hasAlpha: colourType === 6, format: 'PNG', warnings: [] };
}

async function readDimensions(file: File): Promise<{ width: number; height: number }> {
  const bitmap = await createImageBitmap(file).catch(() => null);
  if (!bitmap) throw new FileProblem('The browser could not decode this file.');
  const { width, height } = bitmap;
  bitmap.close();
  return { width, height };
}

/** Throws FileProblem with a one-sentence reason when the file cannot be analysed. */
export async function preflight(file: File): Promise<FileMeta> {
  if (file.size > MAX_BYTES) throw new FileProblem(`${(file.size / 1048576).toFixed(0)} MB exceeds the 64 MB limit.`);
  const head = new Uint8Array(await file.slice(0, 1024 * 1024).arrayBuffer());

  let meta: FileMeta;
  if (head.length >= 33 && PNG_SIGNATURE.every((b, i) => head[i] === b)) {
    meta = inspectPng(head);
  } else if (head[0] === 0xff && head[1] === 0xd8) {
    meta = { ...(await readDimensions(file)), hasAlpha: false, format: 'JPEG', warnings: ['JPEG compression can destroy earlier LSB changes. A negative result is weaker evidence.'] };
  } else if ((head[0] === 0x49 && head[1] === 0x49) || (head[0] === 0x4d && head[1] === 0x4d)) {
    meta = { ...(await readDimensions(file).catch(() => ({ width: 0, height: 0 }))), hasAlpha: false, format: 'TIFF', warnings: [] };
  } else {
    throw new FileProblem('Not a PNG, JPEG, or TIFF image.');
  }

  if (meta.width && Math.min(meta.width, meta.height) < MIN_SIDE) {
    throw new FileProblem(`${meta.width} × ${meta.height} px. Both sides must be at least ${MIN_SIDE} px.`);
  }
  if (meta.width * meta.height > MAX_PIXELS) {
    throw new FileProblem(`${((meta.width * meta.height) / 1e6).toFixed(1)} megapixels exceeds the 32 MP limit.`);
  }
  return meta;
}

export const formatBytes = (n: number) => (n >= 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);
