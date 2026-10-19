// Pipeline entry: quad -> grid -> glare -> colour -> seams -> pieces.
//
// Resolution strategy: photos are large and the mosaic may cover only ~40% of the width,
// so the quad is detected on a small copy (ScanOptions.maxSide = detection size, default
// 1024) and every warp samples the full-resolution input (or an exact 2x2-binned copy when
// the mosaic is much bigger than needed). The rectified image has a fixed pitch (≤64 px).
//
// Coordinates: ScanOptions.corners and Grid.corners are in pixels of the image passed in.
//
// countBaseplate: true (default) counts every stud region as pieces. false treats exposed
// baseplate as not-a-piece: split, seamless regions (>16 studs) of the dominant border
// colour are dropped from pieces/counts (a real plate never produces such a region; an
// uncovered baseplate does).
import type { CV } from './cv';
import { classifyCells } from './colour';
import { glareMask } from './glare';
import { fitGrid, type GridFitResult } from './grid';
import { crop, halve } from './image';
import { DEFAULT_DETECT_SIDE, detectQuad } from './preprocess';
import { CnnSeamClassifier } from './seams/cnn';
import { allPairs, HeuristicSeamClassifier } from './seams/heuristic';
import type { SeamClassifier, SeamContext } from './seams/index';
import type {
  GridSize, PairDecision, Piece, Point, Quad, RgbaImage, ScanOptions, ScanResult, StudCell,
} from './types';
import { breakdown, groupPieces, LOW_CONFIDENCE, type GroupResult } from './validate';

export type ProgressFn = (stage: string, frac: number) => void;

export interface PipelineHooks {
  onProgress?: ProgressFn;
  /** Inject a classifier instance (e.g. a CnnSeamClassifier with a loaded session). */
  classifier?: SeamClassifier;
}

export interface PipelineDetail {
  result: ScanResult;
  warped?: RgbaImage;
  glare?: Uint8Array;
  debug: Record<string, unknown>;
}

/** Mosaic side (px) we want to sample from: keep >= ~1.2x the rectified size (seams are thin). */
const WANT_SRC_SIDE = 1200;

let defaultCnn: CnnSeamClassifier | null = null;
const defaultHeuristic = new HeuristicSeamClassifier();

export function getClassifier(name: ScanOptions['classifier']): SeamClassifier {
  if (name === 'cnn') return (defaultCnn ??= new CnnSeamClassifier());
  return defaultHeuristic;
}

/** Full pipeline entry. Pure w.r.t. inputs; caller supplies an initialised cv (OpenCV.js). */
export async function runPipeline(cv: unknown, img: RgbaImage, opts: ScanOptions, hooks: PipelineHooks = {}): Promise<ScanResult> {
  return (await runPipelineDetailed(cv, img, opts, hooks)).result;
}

export async function runPipelineDetailed(
  cv: unknown, img: RgbaImage, opts: ScanOptions, hooks: PipelineHooks = {},
): Promise<PipelineDetail> {
  const timingsMs: Record<string, number> = {};
  const debug: Record<string, unknown> = {};
  const progress = (s: string, f: number) => { try { hooks.onProgress?.(s, f); } catch { /* ignore */ } };
  let t = performance.now();
  const lap = (name: string) => { const n = performance.now(); timingsMs[name] = Math.round(n - t); t = n; };
  const t0 = t;
  const base = { pieceCount: 0, studCount: 0, lowConfidenceCount: 0, timingsMs, classifier: opts.classifier };
  try {
    // ---- 1. quad
    progress('quad', 0);
    let quad: Quad;
    if (opts.corners) {
      quad = opts.corners;
      debug.quadMethod = 'override';
    } else {
      const q = detectQuad(cv as CV, img, opts.maxSide ?? DEFAULT_DETECT_SIDE);
      quad = q.quad;
      debug.quadMethod = q.method;
      debug.quadConfidence = q.confidence;
    }
    lap('quad');

    // ---- 2. grid (sample from full-res or a binned copy)
    progress('grid', 0.15);
    const side = Math.max(
      Math.hypot(quad[1][0] - quad[0][0], quad[1][1] - quad[0][1]),
      Math.hypot(quad[3][0] - quad[0][0], quad[3][1] - quad[0][1]),
    );
    // Work on a crop around the quad (room for the grid fitter's mat margin), binned 2x2
    // while the mosaic stays >= WANT_SRC_SIDE px.
    const xs = quad.map((p) => p[0]), ys = quad.map((p) => p[1]);
    const pad = side * 0.3;
    const cx0 = Math.max(0, Math.floor(Math.min(...xs) - pad)), cy0 = Math.max(0, Math.floor(Math.min(...ys) - pad));
    const cx1 = Math.min(img.width, Math.ceil(Math.max(...xs) + pad)), cy1 = Math.min(img.height, Math.ceil(Math.max(...ys) + pad));
    let work = cx0 === 0 && cy0 === 0 && cx1 === img.width && cy1 === img.height ? img : crop(img, cx0, cy0, cx1 - cx0, cy1 - cy0);
    let f = 1;
    while (side / (f * 2) >= WANT_SRC_SIDE) { work = halve(work); f *= 2; }
    const toWork = (p: Point): Point => [(p[0] - cx0 + 0.5) / f - 0.5, (p[1] - cy0 + 0.5) / f - 0.5];
    const toSrc = (p: Point): Point => [(p[0] + 0.5) * f - 0.5 + cx0, (p[1] + 0.5) * f - 0.5 + cy0];
    lap('pyramid');
    const fit: GridFitResult = fitGrid(work, quad.map(toWork) as Quad, { gridOverride: opts.gridOverride });
    const grid = { ...fit.grid, corners: fit.grid.corners.map(toSrc) as Quad };
    debug.grid = fit.debug;
    debug.mat = fit.matRgb;
    lap('grid');
    if (!fit.ok) {
      timingsMs.total = Math.round(performance.now() - t0);
      return {
        result: { ...base, status: 'grid_failed', grid, message: `Could not find the stud grid (confidence ${grid.confidence.toFixed(2)}). Adjust the corners or set the grid size.` },
        warped: fit.warped, debug,
      };
    }

    // ---- 3. glare
    progress('glare', 0.4);
    const warped = fit.warped;
    const glare = glareMask(warped, { radius: Math.round(grid.pitch * 1.5) });
    lap('glare');

    // ---- 4. colour
    progress('colour', 0.5);
    const cells = classifyCells(warped, grid, glare, { matRgb: fit.matRgb });
    lap('colour');

    // ---- 5. seams
    progress('seams', 0.65);
    const classifier = hooks.classifier ?? getClassifier(opts.classifier);
    const ctx: SeamContext = { warped, grid, cells, glareMask: glare };
    const pairs = await classifier.classify(ctx, allPairs(grid.cols, grid.rows));
    let used = classifier.name;
    let message: string | undefined;
    if (classifier instanceof CnnSeamClassifier && classifier.fallbackReason) {
      used = 'heuristic';
      message = classifier.fallbackReason;
    }
    if (classifier instanceof HeuristicSeamClassifier) debug.seamStats = classifier.lastStats;
    lap('seams');

    // ---- 6. pieces
    progress('pieces', 0.9);
    let g = groupPieces(cells, grid, pairs);
    if (opts.countBaseplate === false) g = dropBaseplate(g, cells, grid);
    lap('pieces');
    timingsMs.total = Math.round(performance.now() - t0);
    progress('done', 1);
    return {
      result: {
        status: 'ok',
        message,
        grid,
        cells,
        pairs,
        pieces: g.pieces,
        pieceCount: g.pieceCount,
        studCount: g.studCount,
        breakdown: breakdown(g.pieces),
        lowConfidenceCount: g.lowConfidenceCount,
        timingsMs,
        classifier: used,
      },
      warped,
      glare,
      debug,
    };
  } catch (e) {
    timingsMs.total = Math.round(performance.now() - t0);
    return { result: { ...base, status: 'error', message: e instanceof Error ? e.message : String(e) }, debug };
  }
}

/** Re-derive pieces after manual seam edits (no image work). */
export function regroup(cells: StudCell[], grid: GridSize, pairs: PairDecision[]): GroupResult {
  return groupPieces(cells, grid, pairs);
}

function dropBaseplate(g: GroupResult, cells: StudCell[], grid: GridSize): GroupResult {
  const border = new Map<string, number>();
  for (const c of cells) {
    if (c.col === 0 || c.row === 0 || c.col === grid.cols - 1 || c.row === grid.rows - 1) border.set(c.colorId, (border.get(c.colorId) ?? 0) + 1);
  }
  const dominant = [...border.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
  // Regions are recovered by flood-filling split pieces of the dominant colour.
  const split = g.pieces.filter((p) => p.split && p.color === dominant);
  const studs = split.reduce((s, p) => s + p.w * p.h, 0);
  if (!dominant || studs <= 16) return g;
  const keep: Piece[] = g.pieces.filter((p) => !(p.split && p.color === dominant));
  return {
    pieces: keep,
    pieceCount: keep.length,
    studCount: keep.reduce((s, p) => s + p.w * p.h, 0),
    lowConfidenceCount: keep.filter((p) => p.confidence < LOW_CONFIDENCE).length,
  };
}
