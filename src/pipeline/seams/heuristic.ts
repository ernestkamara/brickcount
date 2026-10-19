// Heuristic seam classifier.
//
// - Different palette colour -> seam (confidence from the Lab ΔE between the two cells).
// - Same colour -> measure the gap strip at the cell boundary: locate the line (darkest
//   column within ±12% pitch of the nominal boundary), then per row along the boundary:
//   darkness vs. the flanks, brightness (seams often catch a highlight edge), second
//   derivative across the line ("curvature": a thin line, not a shading ramp), flank
//   asymmetry; aggregate with low percentiles/medians over rows (a real seam spans the whole
//   length). A small logistic model (fitted on the golden photos, leave-one-photo-out pair
//   accuracy 0.895 vs 0.80 for a dark-line threshold) turns two of them into P(seam).
// - Pairs mostly under glare fall back to colour logic (same colour -> no seam) with low
//   confidence; partly glared pairs get their confidence capped.
import { deltaE76 } from '../colour';
import { clamp01, luma, otsu } from '../image';

import type { PairDecision, StudPair } from '../types';
import type { SeamClassifier, SeamContext } from './index';

export const SEAM_THR_DEFAULT = 0.14;
export const SEAM_THR_MIN = 0.07;
export const SEAM_THR_MAX = 0.3;

export interface GapMeasure {
  /** Relative darkness of the boundary line, ~0 (no line) .. 1 (black line). */
  score: number;
  /** Extra features (diagnostics / learned combination). */
  f?: { dark50: number; bright35: number; curv50: number; offset: number; flankDiff: number };
  /** Fraction of boundary rows that were glare-free. */
  clean: number;
}

const lumaCache = new WeakMap<object, Float32Array>();
function getLuma(ctx: SeamContext): Float32Array {
  let l = lumaCache.get(ctx.warped);
  if (!l) { l = luma(ctx.warped); lumaCache.set(ctx.warped, l); }
  return l;
}

/** Dark-line energy across the gap between a pair of studs. */
export function measureGap(
  L: Float32Array, w: number, h: number, cols: number, rows: number, pair: StudPair, glare?: Uint8Array,
): GapMeasure {
  const cw = w / cols, ch = h / rows;
  // Express everything in a local frame: u = across the gap, v = along it.
  const horiz = pair.dir === 'h';
  const P = horiz ? cw : ch; // pitch across the gap
  const Q = horiz ? ch : cw; // pitch along the gap
  const b = horiz ? (pair.col + 1) * cw : (pair.row + 1) * ch; // boundary coordinate (across)
  const v0 = (horiz ? pair.row : pair.col) * Q;
  const at = (u: number, v: number): number => {
    const x = horiz ? u : v, y = horiz ? v : u;
    const xi = Math.min(w - 1, Math.max(0, Math.round(x - 0.5)));
    const yi = Math.min(h - 1, Math.max(0, Math.round(y - 0.5)));
    return yi * w + xi;
  };
  const search = Math.max(2, Math.round(P * 0.12));
  const flankIn = Math.max(search + 1, Math.round(P * 0.13));
  const flankOut = Math.max(flankIn + 2, Math.round(P * 0.19));
  // Along-gap extent: middle 70% of the cell; edge cells of the visible grid may be partly
  // hidden by the frame, so only use their inner half.
  const along = horiz ? pair.row : pair.col;
  const nAlong = horiz ? rows : cols;
  let fa = 0.15, fb = 0.85;
  if (along === 0) fa = 0.45;
  if (along === nAlong - 1) fb = 0.55;
  const vs = Math.max(0, Math.round(v0 + Q * fa)), ve = Math.min(horiz ? h : w, Math.round(v0 + Q * fb));
  // 1) locate the line: mean profile over the strip
  const prof: number[] = [];
  for (let du = -search; du <= search; du++) {
    let s = 0, n = 0;
    for (let v = vs; v < ve; v += 1) { s += L[at(b + du, v + 0.5)]; n++; }
    prof.push(n ? s / n : 0);
  }
  let bi = 0;
  for (let i = 1; i < prof.length; i++) if (prof[i] < prof[bi]) bi = i;
  const uLine = b + (bi - search);
  // 2) per-row darkness at the line relative to the flanks
  const ds: number[] = [], bs: number[] = [], cs: number[] = [], fd: number[] = [];
  let total = 0;
  for (let v = vs; v < ve; v++) {
    total++;
    const vc = v + 0.5;
    if (glare) {
      let g = 0, nn = 0;
      for (let du = -flankOut; du <= flankOut; du += 2) { nn++; if (glare[at(uLine + du, vc)]) g++; }
      if (g > nn * 0.25) continue;
    }
    let mn = Infinity;
    for (let du = -1; du <= 1; du++) mn = Math.min(mn, L[at(uLine + du, vc)]);
    let fl = 0, fr = 0, nf = 0;
    for (let du = flankIn; du <= flankOut; du++) { fl += L[at(uLine - du, vc)]; fr += L[at(uLine + du, vc)]; nf++; }
    const ref = Math.min(fl, fr) / nf;
    const refMax = Math.max(fl, fr) / nf;
    ds.push((ref - mn) / Math.max(25, ref));
    let mx = -Infinity;
    for (let du = -1; du <= 1; du++) mx = Math.max(mx, L[at(uLine + du, vc)]);
    bs.push((mx - refMax) / Math.max(25, refMax));
    const c = L[at(uLine - 2, vc)] + L[at(uLine + 2, vc)] - 2 * L[at(uLine, vc)];
    cs.push(c / Math.max(25, ref));
    fd.push(Math.abs(fl - fr) / nf / Math.max(25, ref));
  }
  if (ds.length === 0) return { score: 0, clean: 0 };
  const q = (a: number[], p: number) => { a.sort((x, y) => x - y); return a[Math.floor((a.length - 1) * p)]; };
  const score = q(ds, 0.35);
  return {
    score: Math.max(0, score),
    clean: ds.length / Math.max(1, total),
    f: { dark50: q(ds, 0.5), bright35: q(bs, 0.35), curv50: q(cs, 0.5), offset: Math.abs(bi - search) / P, flankDiff: q(fd, 0.5) },
  };
}

/** Optional diagnostic: Otsu split of a score list (used by tests/eval tooling). */
export function adaptiveThreshold(scores: number[]): { thr: number; sep: number } {
  const { thr, sep } = otsu(scores);
  return { thr: Math.min(SEAM_THR_MAX, Math.max(SEAM_THR_MIN, sep > 0.55 ? thr : SEAM_THR_DEFAULT)), sep };
}

/**
 * Logistic model over standardised gap features [score (dark line), curv50 (second
 * derivative across the line)]. Fitted on the 6 golden photos with a matching grid:
 * leave-one-photo-out same-colour pair accuracy 0.895 (6 features gave 0.90 but did not
 * transfer to clean synthetic images).
 */
export const SEAM_MODEL = {
  mu: [0.138, 0.3414],
  sd: [0.1567, 0.5218],
  w: [0.224, 2.281],
  b: -0.351,
};

/** P(seam) for one gap measurement. */
export function seamProbability(m: GapMeasure): number {
  if (!m.f) return m.score >= SEAM_THR_DEFAULT ? 0.75 : 0.25;
  const x = [m.score, m.f.curv50];
  let z = SEAM_MODEL.b;
  for (let i = 0; i < x.length; i++) z += SEAM_MODEL.w[i] * ((x[i] - SEAM_MODEL.mu[i]) / SEAM_MODEL.sd[i]);
  return 1 / (1 + Math.exp(-z));
}

export class HeuristicSeamClassifier implements SeamClassifier {
  readonly name = 'heuristic' as const;
  /** Last image's diagnostics. */
  lastStats: { pairs: number; same: number; seamsSame: number; glareFallback: number } | null = null;

  async classify(ctx: SeamContext, pairs: StudPair[]): Promise<PairDecision[]> {
    return this.classifySync(ctx, pairs);
  }

  classifySync(ctx: SeamContext, pairs: StudPair[]): PairDecision[] {
    const { cols, rows } = ctx.grid;
    const { width: w, height: h } = ctx.warped;
    const L = getLuma(ctx);
    const cell = (c: number, r: number) => ctx.cells[r * cols + c];
    const stats = { pairs: pairs.length, same: 0, seamsSame: 0, glareFallback: 0 };
    const out = pairs.map((p): PairDecision => {
      const a = cell(p.col, p.row);
      const b = p.dir === 'h' ? cell(p.col + 1, p.row) : cell(p.col, p.row + 1);
      if (!a || !b) return { ...p, seam: true, confidence: 0 };
      if (a.colorId !== b.colorId) {
        const dE = deltaE76(a.lab, b.lab);
        return { ...p, seam: true, confidence: clamp01(0.5 + dE / 30) };
      }
      stats.same++;
      const m = measureGap(L, w, h, cols, rows, p, ctx.glareMask);
      if (m.clean < 0.3) {
        stats.glareFallback++;
        return { ...p, seam: false, confidence: 0.3 };
      }
      const pr = seamProbability(m);
      const seam = pr > 0.5;
      if (seam) stats.seamsSame++;
      let conf = seam ? pr : 1 - pr;
      if (m.clean < 0.7) conf = Math.min(conf, 0.3 + 0.5 * m.clean);
      return { ...p, seam, confidence: conf };
    });
    this.lastStats = stats;
    return out;
  }
}

/** Every right ('h') and lower ('v') neighbour pair, row-major, h before v per cell. */
export function allPairs(cols: number, rows: number): StudPair[] {
  const out: StudPair[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (c + 1 < cols) out.push({ col: c, row: r, dir: 'h' });
      if (r + 1 < rows) out.push({ col: c, row: r, dir: 'v' });
    }
  }
  return out;
}
