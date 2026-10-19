// Build fixtures/cropped/: tight crops of the original golden photos (mosaic fills the frame,
// only a thin strip of white mat left), the way people often crop or screenshot a photo.
// Crop box = the pipeline's detected quad on the original, expanded by --margin (fraction).
//
//   npx tsx scripts/make-cropped-set.ts [--margin 0.03] [--set cropped]
//
// The committed sets were generated once (cropped: margin 0.01, cropped3: margin 0.03) and are
// frozen test data. Don't regenerate them as part of pipeline work: the crop box comes from
// the pipeline itself, so regenerating would move the test set along with the code under test.
import { copyFileSync, mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import sharp from 'sharp';
import { loadOpenCv } from '../src/pipeline/cv';
import { runPipelineDetailed } from '../src/pipeline/run';
import type { RgbaImage } from '../src/pipeline/types';

async function decodeImage(path: string): Promise<RgbaImage> {
  const { data, info } = await sharp(path).rotate().ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length) };
}

const i = process.argv.indexOf('--margin');
const margin = i > 0 ? Number(process.argv[i + 1]) : 0.03;
const j = process.argv.indexOf('--set');
const set = j > 0 ? process.argv[j + 1] : 'cropped';
const root = process.cwd();
for (const d of ['images', 'labels']) mkdirSync(join(root, 'fixtures', set, d), { recursive: true });
const cv = await loadOpenCv();
for (const f of readdirSync(join(root, 'fixtures/images')).filter((n) => n.endsWith('.jpg')).sort()) {
  const src = join(root, 'fixtures/images', f);
  const img = await decodeImage(src);
  const det = await runPipelineDetailed(cv, img, { classifier: 'heuristic', countBaseplate: true });
  const q = det.result.grid?.corners;
  if (!q) {
    console.log(`${f}: no quad, skipped`);
    continue;
  }
  const xs = q.map((p) => p[0]);
  const ys = q.map((p) => p[1]);
  const w = Math.max(...xs) - Math.min(...xs);
  const h = Math.max(...ys) - Math.min(...ys);
  const left = Math.max(0, Math.round(Math.min(...xs) - w * margin));
  const top = Math.max(0, Math.round(Math.min(...ys) - h * margin));
  const right = Math.min(img.width, Math.round(Math.max(...xs) + w * margin));
  const bottom = Math.min(img.height, Math.round(Math.max(...ys) + h * margin));
  await sharp(src).extract({ left, top, width: right - left, height: bottom - top }).jpeg({ quality: 92 }).toFile(join(root, 'fixtures', set, 'images', f));
  // Same visible window as the original, so the original label applies unchanged.
  copyFileSync(join(root, 'fixtures/labels', f.replace(/\.jpg$/, '.json')), join(root, 'fixtures', set, 'labels', f.replace(/\.jpg$/, '.json')));
  console.log(`${f}: ${right - left}x${bottom - top}`);
}
