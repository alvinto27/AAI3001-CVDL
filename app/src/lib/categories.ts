import type { CategoryId } from '../types/analysis';

// Registry of anomaly categories. Add a row when the engine learns a new one;
// components read labels from here and never switch on ids.
export const CATEGORIES: Record<CategoryId, { label: string; family: 'hidden-data' | 'manipulation' | 'unknown' }> = {
  lsb_steganography: { label: 'LSB steganography', family: 'hidden-data' },
  object_removal: { label: 'Object removal / inpainting', family: 'manipulation' },
  generative_fill: { label: 'Generative fill', family: 'manipulation' },
  copy_move: { label: 'Copy-move manipulation', family: 'manipulation' },
  local_reconstruction: { label: 'Local reconstruction', family: 'manipulation' },
  recompression: { label: 'Recompression / processing', family: 'manipulation' },
  unknown_manipulation: { label: 'Unknown manipulation', family: 'manipulation' },
  unclassified: { label: 'Unclassified anomaly', family: 'unknown' },
};

export const FAMILY_LABEL = {
  'hidden-data': 'Steganographic anomaly',
  manipulation: 'Image-manipulation anomaly',
  unknown: 'Unclassified anomaly',
} as const;

export function asCategoryId(value: string | undefined): CategoryId {
  return value && value in CATEGORIES ? (value as CategoryId) : 'unclassified';
}
