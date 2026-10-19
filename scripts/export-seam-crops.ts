// Export labelled seam crops from the real photos, for fine-tuning the seam CNN (ml/finetune.py).
//
//   npx tsx scripts/export-seam-crops.ts [--sets real,closeup] [--out ml/data/real] [--blur 0.5]
//
// For each labelled photo the real pipeline runs up to the grid fit, using a capturing classifier,
// so the rectified image, grid and cells are exactly what inference sees. Crops are cut with
// src/pipeline/seams/crop.ts after the same pre-blur as CnnSeamClassifier (default 0.5 crop px).
// The fitted grid is aligned to the label by the (dx,dy) in {-1,0,1}^2 that maximises cell-colour
// agreement. Photos whose alignment is ambiguous or poor are skipped and listed (summary.json).
//
// Output per photo, in <out>/<set>__<stem>.*:
//   .crops.f32  float32 [N,3,16,32] (exactly the CNN input)
//   .warped.png the blurred rectified image (for jittered re-crops in Python)
//   .json       pairs (pred col,row,dir; label col,row; seam; sameColour; glare) + grid + alignment
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, extname, join } from 'node:path';
import sharp from 'sharp';
import { loadOpenCv } from '../src/pipeline/cv';
import { gaussianBlur } from '../src/pipeline/image';
import { runPipelineDetailed } from '../src/pipeline/run';
import { DEFAULT_BLUR_CROP_PX } from '../src/pipeline/seams/cnn';
import { extractGapCrops } from '../src/pipeline/seams/crop';
import type { SeamClassifier, SeamContext } from '../src/pipeline/seams/index';
import type { FixtureLabel, PairDecision, RgbaImage, StudPair } from '../src/pipeline/types';

const args = process.argv.slice(2);
const arg = (name: string, def?: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : def;
};
const root = process.cwd();
const sets = arg('sets', 'real,closeup')!.split(',');
const outDir = join(root, arg('out', 'ml/data/real')!);
const blur = Number(arg('blur', String(DEFAULT_BLUR_CROP_PX)));

async function decodeImage(path: string): Promise<RgbaImage> {
  const { data, info } = await sharp(path).rotate().ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length) };
}

class Capture implements SeamClassifier {
  readonly name = 'cnn' as const;
  ctx: SeamContext | null = null;
  async classify(ctx: SeamContext, pairs: StudPair[]): Promise<PairDecision[]> {
    this.ctx = ctx;
    return pairs.map((p) => ({ ...p, seam: false, confidence: 0 }));
  }
}

async function main() {
  mkdirSync(outDir, { recursive: true });
  const cv = await loadOpenCv();
  const skipped: { photo: string; reason: string }[] = [];
  const summary: Record<string, unknown>[] = [];
  for (const set of sets) {
    const setDir = set === 'real' ? 'fixtures' : `fixtures/${set}`;
    const imgDir = join(root, setDir, 'images');
    const files = readdirSync(imgDir).filter((f) => /\.(jpe?g|png)$/i.test(f)).sort();
    for (const f of files) {
      const stem = basename(f, extname(f));
      const labelPath = join(root, setDir, 'labels', `${stem}.json`);
      if (!existsSync(labelPath)) continue;
      const label: FixtureLabel = JSON.parse(readFileSync(labelPath, 'utf8'));
      const cap = new Capture();
      const det = await runPipelineDetailed(cv, await decodeImage(join(imgDir, f)), { classifier: 'cnn', countBaseplate: true }, { classifier: cap });
      const ctx = cap.ctx;
      const photo = `${set}/${stem}`;
      if (det.result.status !== 'ok' || !ctx) { skipped.push({ photo, reason: `pipeline ${det.result.status}` }); continue; }
      const { cols, rows } = ctx.grid;
      const L = label.grid;
      // label owner / colour maps
      const owner = new Int32Array(L.cols * L.rows).fill(-1);
      const colour: string[] = new Array(L.cols * L.rows).fill('');
      label.pieces.forEach((p, i) => {
        for (let y = p.y; y < p.y + p.h; y++) for (let x = p.x; x < p.x + p.w; x++) {
          if (x < L.cols && y < L.rows) { owner[y * L.cols + x] = i; colour[y * L.cols + x] = p.color; }
        }
      });
      // alignment by cell colour agreement: label (c+dx, r+dy) <-> pred (c, r)
      const scores: { dx: number; dy: number; agree: number; n: number }[] = [];
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        let agree = 0, n = 0;
        for (const cell of ctx.cells) {
          const lc = cell.col + dx, lr = cell.row + dy;
          if (lc < 0 || lr < 0 || lc >= L.cols || lr >= L.rows) continue;
          n++;
          if (colour[lr * L.cols + lc] === cell.colorId) agree++;
        }
        scores.push({ dx, dy, agree: n ? agree / n : 0, n });
      }
      scores.sort((a, b) => b.agree - a.agree);
      const [best, second] = scores;
      const sizeMatch = cols === L.cols && rows === L.rows;
      const margin = best.agree - second.agree;
      // accept a clear winner: high agreement, or moderate agreement with a large margin (close-ups have
      // weaker colour classification but an unambiguous shift)
      const ok = best.n >= 0.85 * L.cols * L.rows && ((best.agree >= 0.8 && margin >= 0.08) || (best.agree >= 0.65 && margin >= 0.15));
      const align = { dx: best.dx, dy: best.dy, agree: +best.agree.toFixed(3), second: +second.agree.toFixed(3), sizeMatch, pred: `${cols}x${rows}`, label: `${L.cols}x${L.rows}` };
      if (!ok || (sizeMatch && (best.dx || best.dy) && margin < 0.15)) {
        skipped.push({ photo, reason: `ambiguous/poor alignment ${JSON.stringify(align)}` });
        continue;
      }
      const src = blur > 0 ? gaussianBlur(ctx.warped, (blur * ctx.warped.width) / cols / 16) : ctx.warped;
      const pairs: StudPair[] = [];
      const meta: Record<string, unknown>[] = [];
      const cell = (c: number, r: number) => ctx.cells[r * cols + c];
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        for (const dir of ['h', 'v'] as const) {
          const c2 = dir === 'h' ? c + 1 : c, r2 = dir === 'h' ? r : r + 1;
          if (c2 >= cols || r2 >= rows) continue;
          const la = [c + best.dx, r + best.dy], lb = [c2 + best.dx, r2 + best.dy];
          if ([la, lb].some(([x, y]) => x < 0 || y < 0 || x >= L.cols || y >= L.rows)) continue;
          const oa = owner[la[1] * L.cols + la[0]], ob = owner[lb[1] * L.cols + lb[0]];
          if (oa < 0 || ob < 0) continue;
          pairs.push({ col: c, row: r, dir });
          meta.push({
            col: c, row: r, dir, lcol: la[0], lrow: la[1],
            seam: oa !== ob ? 1 : 0,
            sameColour: colour[la[1] * L.cols + la[0]] === colour[lb[1] * L.cols + lb[0]] ? 1 : 0,
            glare: +Math.max(cell(c, r).glare, cell(c2, r2).glare).toFixed(3),
          });
        }
      }
      const crops = extractGapCrops(src, cols, rows, pairs);
      const base = join(outDir, `${set}__${stem}`);
      writeFileSync(`${base}.crops.f32`, Buffer.from(crops.buffer, crops.byteOffset, crops.byteLength));
      await sharp(Buffer.from(src.data.buffer, src.data.byteOffset, src.data.length), { raw: { width: src.width, height: src.height, channels: 4 } })
        .removeAlpha().png().toFile(`${base}.warped.png`);
      const seams = meta.filter((m) => m.seam).length;
      const same = meta.filter((m) => m.sameColour).length;
      writeFileSync(`${base}.json`, JSON.stringify({ set, stem, mosaic: stem, cols, rows, width: src.width, height: src.height, blurCropPx: blur, align, pairs: meta }));
      summary.push({ photo, n: meta.length, seams, same, align });
      console.log(`${photo.padEnd(22)} grid ${cols}x${rows} label ${L.cols}x${L.rows} shift ${best.dx},${best.dy} agree ${align.agree} (2nd ${align.second})  pairs ${meta.length} seam ${seams} sameColour ${same}`);
    }
  }
  writeFileSync(join(outDir, 'summary.json'), JSON.stringify({ summary, skipped }, null, 1));
  for (const s of skipped) console.log(`SKIPPED ${s.photo}: ${s.reason}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
