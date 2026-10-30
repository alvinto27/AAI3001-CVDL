// Encodes the EffectCraft renders (build/composited/<clip>/c####.png) to looping H.264 MP4s in out/.
//   FFMPEG=path/to/ffmpeg node preview-video/encode.mjs
import { execFileSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const FFMPEG = process.env.FFMPEG ?? 'ffmpeg';
mkdirSync(join(HERE, 'out'), { recursive: true });

for (const clip of ['original', 'heatmap']) {
  execFileSync(FFMPEG, [
    '-v', 'error', '-y',
    '-framerate', '30', '-i', join(HERE, 'build/composited', clip, 'c%04d.png'),
    '-frames:v', '300',
    '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
    '-c:v', 'libx264', '-preset', 'veryslow', '-crf', '20', '-profile:v', 'high',
    // One keyframe a second keeps seeking and loop restarts cheap.
    '-g', '30', '-keyint_min', '30', '-sc_threshold', '0',
    '-bsf:v', 'h264_metadata=video_full_range_flag=0:colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1',
    '-an', '-movflags', '+faststart',
    join(HERE, 'out', `inspection-${clip}.mp4`),
  ], { stdio: 'inherit' });
}
