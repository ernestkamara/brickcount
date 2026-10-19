// Evaluate the pipeline on the golden photos (or the synthetic set).
//
//   npx tsx scripts/eval.ts [--set real|synthetic] [--classifier heuristic|cnn] [--annotate]
//                           [--only <stem>[,<stem>]] [--limit N] [--out eval-report.json] [--blur 0.5 (cnn pre-blur)]
//
// real:      fixtures/images/*      labels fixtures/labels/<stem>.json
// synthetic: fixtures/synthetic/images/*  labels fixtures/synthetic/labels/<stem>.json
// Missing labels -> prediction only. --annotate writes eval-out/<stem>.png (rectified image,
// green = correct piece, red = wrong predicted piece, blue dashed = missed true piece).
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, extname, join } from 'node:path';
import sharp from 'sharp';
import { loadOpenCv } from '../src/pipeline/cv';
import { runPipelineDetailed } from '../src/pipeline/run';
import { CnnSeamClassifier, type OrtLike } from '../src/pipeline/seams/cnn';
import type { SeamClassifier } from '../src/pipeline/seams/index';
import type { FixtureLabel, PieceRect, RgbaImage, ScanOptions, SeamClassifierName } from '../src/pipeline/types';

const args = process.argv.slice(2);
const arg = (name: string, def?: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : def;
};
const flag = (name: string) => args.includes(`--${name}`);

const set = arg('set', 'real')!;
const classifierName = (arg('classifier', 'heuristic') as SeamClassifierName);
const annotate = flag('annotate');
const only = arg('only')?.split(',');
const limit = Number(arg('limit', '0'));
const outPath = arg('out', 'eval-report.json')!;
const root = process.cwd();
const imgDir = set === 'synthetic' ? join(root, 'fixtures/synthetic/images') : join(root, 'fixtures/images');
const labelDir = set === 'synthetic' ? join(root, 'fixtures/synthetic/labels') : join(root, 'fixtures/labels');

export async function decodeImage(path: string): Promise<RgbaImage> {
  const { data, info } = await sharp(path).rotate().ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length) };
}

async function makeCnn(): Promise<SeamClassifier | null> {
  const modelPath = join(root, 'public/models/seam.onnx');
  if (!existsSync(modelPath)) {
    console.log(`[cnn] ${modelPath} not found — skipping CNN eval.`);
    return null;
  }
  try {
    const ort = (await import('onnxruntime-web')) as unknown as OrtLike & { env: { wasm: { numThreads: number } } };
    ort.env.wasm.numThreads = 1;
    const cnn = new CnnSeamClassifier({ ort, model: new Uint8Array(readFileSync(modelPath)), blurCropPx: arg('blur') !== undefined ? Number(arg('blur')) : undefined });
    await cnn.load();
    if (!cnn.ready) { console.log(`[cnn] ${cnn.fallbackReason} — skipping.`); return null; }
    return cnn;
  } catch (e) {
    console.log(`[cnn] could not load onnxruntime in Node (${(e as Error).message}) — skipping.`);
    return null;
  }
}

const rectKey = (p: PieceRect) => `${p.x},${p.y},${p.w},${p.h}`;

interface PhotoReport {
  image: string;
  status: string;
  message?: string;
  grid?: { cols: number; rows: number; confidence: number };
  trueGrid?: { cols: number; rows: number };
  gridDetected?: boolean;
  predPieces: number;
  truePieces?: number;
  predStuds: number;
  trueStuds?: number;
  matched?: number;
  matchedColour?: number;
  precision?: number;
  recall?: number;
  cellColourAcc?: number;
  /** 'ok' | 'grid_mismatch' | 'grid_failed' | 'error' | 'unlabelled' */
  category?: string;
  /** Max distance of predicted to labelled corners, in label pitches (labels with corners only). */
  cornerErrPitch?: number;
  /** Label flagged LOW CONFIDENCE in its notes. */
  lowConfidenceLabel?: boolean;
  /** Best (dx,dy) shift of the prediction onto the label (≠0 when the grid is off by a row/col) and P/R under it. */
  aligned?: { dx: number; dy: number; matched: number; precision: number; recall: number };
  lowConfidence: number;
  ms: number;
  timings: Record<string, number>;
  bySize?: Record<string, number>;
  debug?: unknown;
}

async function main() {
  if (!existsSync(imgDir)) { console.log(`No images dir ${imgDir}`); return; }
  let files = readdirSync(imgDir).filter((f) => /\.(jpe?g|png|webp)$/i.test(f)).sort();
  if (only) files = files.filter((f) => only.includes(basename(f, extname(f))));
  if (limit > 0) files = files.slice(0, limit);
  if (files.length === 0) { console.log(`No images in ${imgDir}`); return; }

  const cv = await loadOpenCv();
  let classifier: SeamClassifier | undefined;
  if (classifierName === 'cnn') {
    const c = await makeCnn();
    if (!c) return;
    classifier = c;
  }
  if (annotate) mkdirSync(join(root, 'eval-out'), { recursive: true });

  const reports: PhotoReport[] = [];
  for (const f of files) {
    const stem = basename(f, extname(f));
    const img = await decodeImage(join(imgDir, f));
    const opts: ScanOptions = { classifier: classifierName, countBaseplate: true };
    const t = performance.now();
    const det = await runPipelineDetailed(cv, img, opts, { classifier });
    const ms = Math.round(performance.now() - t);
    const r = det.result;
    const labelPath = join(labelDir, `${stem}.json`);
    const label: FixtureLabel | null = existsSync(labelPath) ? JSON.parse(readFileSync(labelPath, 'utf8')) : null;
    const pred = r.pieces ?? [];
    const rep: PhotoReport = {
      image: f,
      status: r.status,
      message: r.message,
      grid: r.grid ? { cols: r.grid.cols, rows: r.grid.rows, confidence: +r.grid.confidence.toFixed(2) } : undefined,
      predPieces: r.pieceCount,
      predStuds: r.studCount,
      lowConfidence: r.lowConfidenceCount,
      ms,
      timings: r.timingsMs,
      bySize: r.breakdown?.bySize,
      debug: det.debug,
    };
    let trueKeys = new Set<string>();
    if (label) {
      const truth = label.pieces;
      trueKeys = new Set(truth.map(rectKey));
      const trueByKey = new Map(truth.map((p) => [rectKey(p), p]));
      rep.trueGrid = label.grid;
      rep.gridDetected = r.status === 'ok' && r.grid?.cols === label.grid.cols && r.grid?.rows === label.grid.rows;
      rep.truePieces = truth.length;
      rep.trueStuds = truth.reduce((s, p) => s + p.w * p.h, 0);
      rep.lowConfidenceLabel = /low confidence/i.test(label.notes ?? '');
      if (label.corners && r.grid) {
        const pitch = Math.hypot(label.corners[1][0] - label.corners[0][0], label.corners[1][1] - label.corners[0][1]) / label.grid.cols;
        rep.cornerErrPitch = +(Math.max(...label.corners.map((c, i) => Math.hypot(c[0] - r.grid!.corners[i][0], c[1] - r.grid!.corners[i][1]))) / pitch).toFixed(2);
      }
      rep.category = r.status !== 'ok' ? r.status : rep.gridDetected ? 'ok' : 'grid_mismatch';
      // shift-tolerant matching: diagnoses "pieces right, grid off by one edge row/col"
      let best = { dx: 0, dy: 0, matched: -1 };
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const m = pred.filter((p) => trueKeys.has(rectKey({ ...p, x: p.x + dx, y: p.y + dy }))).length;
        if (m > best.matched || (m === best.matched && dx === 0 && dy === 0)) best = { dx, dy, matched: m };
      }
      rep.aligned = { ...best, precision: pred.length ? +(best.matched / pred.length).toFixed(3) : 0, recall: +(best.matched / truth.length).toFixed(3) };
      const matched = pred.filter((p) => trueKeys.has(rectKey(p)));
      rep.matched = matched.length;
      rep.matchedColour = matched.filter((p) => trueByKey.get(rectKey(p))!.color === p.color).length;
      rep.precision = pred.length ? +(matched.length / pred.length).toFixed(3) : 0;
      rep.recall = truth.length ? +(matched.length / truth.length).toFixed(3) : 0;
      if (r.cells && rep.gridDetected) {
        const colourAt = new Map<string, string>();
        for (const p of truth) for (let y = p.y; y < p.y + p.h; y++) for (let x = p.x; x < p.x + p.w; x++) colourAt.set(`${x},${y}`, p.color);
        const ok = r.cells.filter((c) => colourAt.get(`${c.col},${c.row}`) === c.colorId).length;
        rep.cellColourAcc = +(ok / r.cells.length).toFixed(3);
      }
    }
    if (!label) rep.category = r.status === 'ok' ? 'unlabelled' : r.status;
    reports.push(rep);
    const al = rep.aligned && (rep.aligned.dx || rep.aligned.dy) ? ` [shift ${rep.aligned.dx},${rep.aligned.dy}: P ${rep.aligned.precision} R ${rep.aligned.recall}]` : '';
    const lab = label
      ? `true ${label.grid.cols}x${label.grid.rows} ${rep.truePieces} pcs/${rep.trueStuds} studs, matched ${rep.matched} (P ${rep.precision} R ${rep.recall})${al} colour ${rep.cellColourAcc ?? '-'}${rep.category !== 'ok' ? ` ** ${rep.category.toUpperCase()} **` : ''}${rep.lowConfidenceLabel ? ' (label low-confidence)' : ''}${rep.cornerErrPitch !== undefined ? ` cornerErr ${rep.cornerErrPitch}p` : ''}`
      : '(no label)';
    console.log(
      `${stem.padEnd(14)} ${r.status.padEnd(11)} grid ${r.grid ? `${r.grid.cols}x${r.grid.rows}@${r.grid.confidence.toFixed(2)}` : '-'}  pred ${r.pieceCount} pcs/${r.studCount} studs  ${lab}  ${ms} ms`,
    );
    if (r.message) console.log(`   ${r.message}`);

    if (annotate && det.warped && r.grid) {
      const S = det.warped.width, H = det.warped.height;
      const cw = S / r.grid.cols, ch = H / r.grid.rows;
      let svg = '';
      for (const p of pred) {
        const good = !label || trueKeys.has(rectKey(p));
        svg += `<rect x="${p.x * cw + 3}" y="${p.y * ch + 3}" width="${p.w * cw - 6}" height="${p.h * ch - 6}" fill="none" stroke="${good ? '#00e000' : '#ff2020'}" stroke-width="${good ? 2 : 4}"/>`;
        if (p.confidence < 0.6) svg += `<circle cx="${(p.x + 0.2) * cw}" cy="${(p.y + 0.2) * ch}" r="5" fill="orange"/>`;
      }
      if (label) {
        const predKeys = new Set(pred.map(rectKey));
        for (const p of label.pieces) if (!predKeys.has(rectKey(p))) svg += `<rect x="${p.x * cw + 8}" y="${p.y * ch + 8}" width="${p.w * cw - 16}" height="${p.h * ch - 16}" fill="none" stroke="#2050ff" stroke-width="3" stroke-dasharray="8 5"/>`;
      }
      await sharp(Buffer.from(det.warped.data.buffer, det.warped.data.byteOffset, det.warped.data.length), { raw: { width: S, height: H, channels: 4 } })
        .composite([{ input: Buffer.from(`<svg width="${S}" height="${H}">${svg}</svg>`) }])
        .png()
        .toFile(join(root, 'eval-out', `${stem}.png`));
    }
  }

  const labelled = reports.filter((r) => r.truePieces !== undefined);
  const gridOk = labelled.filter((r) => r.category === 'ok');
  const mean = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
  const pr = (rs: PhotoReport[]) => {
    const m = rs.reduce((s, r) => s + (r.matched ?? 0), 0);
    const p = rs.reduce((s, r) => s + r.predPieces, 0);
    const t = rs.reduce((s, r) => s + (r.truePieces ?? 0), 0);
    return { precision: p ? m / p : 0, recall: t ? m / t : 0 };
  };
  const categories: Record<string, string[]> = {};
  for (const r of reports) (categories[r.category ?? 'unknown'] ??= []).push(r.image);
  const summary = {
    set,
    classifier: classifierName,
    photos: reports.length,
    labelled: labelled.length,
    categories,
    exactCountRate: mean(labelled.map((r) => (r.predPieces === r.truePieces ? 1 : 0))),
    meanAbsPctCountError: mean(labelled.map((r) => (Math.abs(r.predPieces - r.truePieces!) / r.truePieces!) * 100)),
    gridDetectedRate: mean(labelled.map((r) => (r.gridDetected ? 1 : 0))),
    piecePrecision: pr(labelled).precision,
    pieceRecall: pr(labelled).recall,
    gridOkPiecePrecision: pr(gridOk).precision,
    gridOkPieceRecall: pr(gridOk).recall,
    shiftAlignedPrecision: (() => { const m = labelled.reduce((s, r) => s + (r.aligned?.matched ?? 0), 0); const p = labelled.reduce((s, r) => s + r.predPieces, 0); return p ? m / p : 0; })(),
    meanCellColourAcc: mean(labelled.filter((r) => r.cellColourAcc !== undefined).map((r) => r.cellColourAcc!)),
    lowConfidenceLabels: labelled.filter((r) => r.lowConfidenceLabel).map((r) => r.image),
    meanMs: mean(reports.map((r) => r.ms)),
  };
  console.log('\nSummary:', JSON.stringify(summary, (_, v) => (typeof v === 'number' ? +v.toFixed(3) : v), 2));
  writeFileSync(join(root, outPath), JSON.stringify({ summary, photos: reports }, null, 2));
  console.log(`wrote ${outPath}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
