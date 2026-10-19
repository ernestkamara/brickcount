// Draft a ground-truth label from the pipeline's prediction, for hand correction.
//
//   npx tsx scripts/label-draft.ts fixtures/images/09_new.jpg [--out fixtures/labels/09_new.json] [--force]
//                                   [--grid 15x16] [--classifier heuristic|cnn]
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { basename, dirname, extname, join } from 'node:path';
import sharp from 'sharp';
import { loadOpenCv } from '../src/pipeline/cv';
import { runPipeline } from '../src/pipeline/run';
import type { FixtureLabel, ScanOptions, SeamClassifierName } from '../src/pipeline/types';

const args = process.argv.slice(2);
const arg = (n: string) => { const i = args.indexOf(`--${n}`); return i >= 0 ? args[i + 1] : undefined; };
const image = args.find((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--') && args[i - 1] !== '--force'));
if (!image) {
  console.error('usage: tsx scripts/label-draft.ts <image> [--out path] [--force] [--grid CxR] [--classifier heuristic|cnn]');
  process.exit(1);
}
const stem = basename(image, extname(image));
const out = arg('out') ?? join('fixtures/labels', `${stem}.json`);
const grid = arg('grid')?.split('x').map(Number);

async function main() {
  if (existsSync(out) && !args.includes('--force')) {
    console.error(`${out} exists — refusing to overwrite a (possibly hand-corrected) label. Use --force.`);
    process.exit(1);
  }
  const { data, info } = await sharp(image!).rotate().ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const img = { width: info.width, height: info.height, data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length) };
  const cv = await loadOpenCv();
  const opts: ScanOptions = {
    classifier: (arg('classifier') as SeamClassifierName) ?? 'heuristic',
    countBaseplate: true,
    gridOverride: grid && grid.length === 2 ? { cols: grid[0], rows: grid[1] } : undefined,
  };
  const r = await runPipeline(cv, img, opts);
  if (r.status !== 'ok' || !r.grid || !r.pieces) {
    console.error(`pipeline ${r.status}: ${r.message ?? ''}`);
    process.exit(2);
  }
  const low = r.pieces.filter((p) => p.confidence < 0.6 || p.split);
  const label: FixtureLabel = {
    image: basename(image!),
    grid: { cols: r.grid.cols, rows: r.grid.rows },
    corners: r.grid.corners.map(([x, y]) => [Math.round(x), Math.round(y)]) as FixtureLabel['corners'],
    pieces: r.pieces.map(({ x, y, w, h, color }) => ({ x, y, w, h, color })),
    notes: `DRAFT from pipeline (${r.classifier}); ${r.pieceCount} pieces. Check grid size and these low-confidence/split pieces: ${low.map((p) => `${p.size}@(${p.x},${p.y})`).join(' ') || 'none'}.`,
  };
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify(label, null, 1) + '\n');
  console.log(`wrote ${out}: grid ${label.grid.cols}x${label.grid.rows}, ${label.pieces.length} pieces, ${low.length} to check`);
}

main().catch((e) => { console.error(e); process.exit(1); });
