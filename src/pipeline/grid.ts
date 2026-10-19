// Stud-grid fitting on a rectified image.
//
// 1. Warp the detected quad (plus a margin of mat) to a square.
// 2. Per axis, build a gradient-projection profile and find the stud pitch by a harmonic
//    Fourier score (sum of |F(k/T)|, k=1..3) over candidate periods T.
// 3. The fundamental's phase gives the cell-boundary lattice; the number of cells and the
//    first cell come from the run of "textured" (stud) strips around the centre vs the
//    smooth mat outside (robust to the mat covering part of the outer row and to the quad
//    being a little off).
// 4. Map the fitted grid boundary back to the source, re-warp and refine per half-band so
//    residual keystone is removed, then produce the final warp at a fixed pitch.
import { applyH, clamp01, luma, median, warpQuad } from './image';
import type { Grid, GridSize, Point, Quad, RgbaImage } from './types';

export const MIN_STUDS = 8;
export const MAX_STUDS = 48;
/** Edge cell kept only if its stud centre is at least this many pitches inside the visible window. */
export const CENTRE_MARGIN = 0;
/** Edge cell must be at least this fraction visible (stud centre visible) to count. */
export const EDGE_VISIBLE = 0.5;

export interface AxisFit {
  /** First grid line (cell boundary), px. */
  start: number;
  period: number;
  n: number;
  /** Harmonic peak strength relative to the median over candidate periods. */
  peak: number;
  /** (inside - outside) / inside strip texture. */
  contrast: number;
}

export interface GridFitResult {
  ok: boolean;
  grid: Grid;
  warped: RgbaImage;
  /** warped -> source homography. */
  toSrc: number[];
  /** Median mat colour (white reference) sampled around the mosaic, sRGB. */
  matRgb: [number, number, number] | null;
  debug: Record<string, number | string>;
}

export interface FitGridOptions {
  gridOverride?: GridSize;
  /** Rectified pitch for the final warp (px). Default: min(64, source pitch). */
  pitch?: number;
}

/** Gradient profiles of a grey image: px[x] = mean_y |dI/dx| over y in [y0,y1), etc. */
export function gradientProfiles(
  g: Float32Array,
  w: number,
  h: number,
  band: { x0: number; x1: number; y0: number; y1: number },
): { px: Float64Array; py: Float64Array; ex: Float64Array; ey: Float64Array } {
  const px = new Float64Array(w); // |gx| summed over band rows
  const py = new Float64Array(h); // |gy| summed over band cols
  const ex = new Float64Array(w); // |gx|+|gy| over band rows (texture per column)
  const ey = new Float64Array(h);
  const { x0, x1, y0, y1 } = band;
  for (let y = 1; y < h - 1; y++) {
    const inY = y >= y0 && y < y1;
    const row = y * w;
    for (let x = 1; x < w - 1; x++) {
      const i = row + x;
      const dx = Math.abs(g[i + 1] - g[i - 1]);
      const dy = Math.abs(g[i + w] - g[i - w]);
      const inX = x >= x0 && x < x1;
      if (inY) { px[x] += dx; ex[x] += dx + dy; }
      if (inX) { py[y] += dy; ey[y] += dx + dy; }
    }
  }
  const nY = Math.max(1, y1 - y0), nX = Math.max(1, x1 - x0);
  for (let x = 0; x < w; x++) { px[x] /= nY; ex[x] /= nY; }
  for (let y = 0; y < h; y++) { py[y] /= nX; ey[y] /= nX; }
  return { px, py, ex, ey };
}

/** Complex Fourier coefficient of (p - mean) over [a,b) at angular frequency w. */
function fourier(p: Float64Array, a: number, b: number, mean: number, w: number): [number, number] {
  let re = 0, im = 0;
  // incremental rotation
  let c = Math.cos(w * a), s = -Math.sin(w * a);
  const dc = Math.cos(w), ds = -Math.sin(w);
  for (let x = a; x < b; x++) {
    const v = p[x] - mean;
    re += v * c;
    im += v * s;
    const nc = c * dc - s * ds;
    s = c * ds + s * dc;
    c = nc;
  }
  return [re, im];
}

function harmonicScore(p: Float64Array, a: number, b: number, mean: number, T: number): number {
  let s = 0;
  for (let k = 1; k <= 3; k++) {
    const [re, im] = fourier(p, a, b, mean, (2 * Math.PI * k) / T);
    s += Math.hypot(re, im) / (k === 1 ? 1 : 1.5);
  }
  return s;
}

/** Find best period in [tMin, tMax] over profile range [a,b). */
export function findPeriod(p: Float64Array, a: number, b: number, tMin: number, tMax: number): { T: number; peak: number } {
  let mean = 0;
  for (let x = a; x < b; x++) mean += p[x];
  mean /= Math.max(1, b - a);
  const scores: number[] = [];
  const Ts: number[] = [];
  const step = 1.004;
  for (let T = tMin; T <= tMax; T *= step) {
    Ts.push(T);
    scores.push(harmonicScore(p, a, b, mean, T));
  }
  let bi = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[bi]) bi = i;
  // parabolic refinement in log-T
  let T = Ts[bi];
  if (bi > 0 && bi < scores.length - 1) {
    const y0 = scores[bi - 1], y1 = scores[bi], y2 = scores[bi + 1];
    const d = y0 - 2 * y1 + y2;
    if (d < 0) T = Ts[bi] * Math.pow(step, (0.5 * (y0 - y2)) / d);
  }
  const med = median(scores) || 1e-9;
  return { T, peak: scores[bi] / med };
}

/** Grid-line phase (position of a cell boundary mod T) from the fundamental. */
export function findPhase(p: Float64Array, a: number, b: number, T: number): number {
  let mean = 0;
  for (let x = a; x < b; x++) mean += p[x];
  mean /= Math.max(1, b - a);
  const w = (2 * Math.PI) / T;
  const [re, im] = fourier(p, a, b, mean, w);
  // p ~ A cos(w (x - x0))  =>  F = A/2 * N * e^{-i w x0}
  let x0 = (-Math.atan2(im, re) / w) % T;
  if (x0 < 0) x0 += T;
  return x0;
}

function stripScores(e: Float64Array, origin: number, T: number): { k0: number; s: number[] } {
  // strips [origin + kT, origin + (k+1)T) fully inside [0, len)
  const len = e.length;
  const k0 = Math.ceil((0 - origin) / T);
  const s: number[] = [];
  for (let k = k0; origin + (k + 1) * T <= len; k++) {
    const a = Math.round(origin + k * T), b = Math.round(origin + (k + 1) * T);
    // trim strip edges slightly so a misplaced line doesn't leak
    const ta = a + Math.round(T * 0.1), tb = b - Math.round(T * 0.1);
    let acc = 0;
    for (let x = ta; x < tb; x++) acc += e[x];
    s.push(acc / Math.max(1, tb - ta));
  }
  return { k0, s };
}

/** Period, peak strength and fundamental phase for one axis. */
export interface AxisPeriod {
  T: number;
  peak: number;
  /** position of the fundamental's maximum, mod T (either cell boundary or cell centre). */
  phase: number;
}

export function axisPeriod(p: Float64Array, inner: [number, number], nFixed?: number): AxisPeriod {
  const [a, b] = inner;
  const span = b - a;
  let tMin = span / (MAX_STUDS + 2), tMax = span / (MIN_STUDS - 1.5);
  if (nFixed) { tMin = (span / nFixed) * 0.85; tMax = (span / nFixed) * 1.15; }
  // analyse a slightly shrunken inner range (outer row may be partly under the mat)
  const ia = Math.round(a + span * 0.04), ib = Math.round(b - span * 0.04);
  const { T, peak } = findPeriod(p, ia, ib, tMin, tMax);
  return { T, peak, phase: findPhase(p, ia, ib, T) };
}

/**
 * Given the boundary lattice (origin + kT), find the run of textured strips (cells) around
 * the centre. `inner` = [a,b) is where the quad says the mosaic is; outside is mat.
 */
export function axisRun(
  e: Float64Array,
  inner: [number, number],
  origin: number,
  T: number,
  nFixed?: number,
): { start: number; n: number; contrast: number; visLo: number; visHi: number } | null {
  const [a, b] = inner;
  const { k0, s } = stripScores(e, origin, T);
  if (s.length < 3) return null;
  const insideIdx: number[] = [], outsideIdx: number[] = [];
  for (let i = 0; i < s.length; i++) {
    const lo = origin + (k0 + i) * T, hi = lo + T;
    if (lo >= a + T * 0.5 && hi <= b - T * 0.5) insideIdx.push(i);
    else if (hi <= a - T * 0.2 || lo >= b + T * 0.2) outsideIdx.push(i);
  }
  if (insideIdx.length === 0) return null;
  const I = median(insideIdx.map((i) => s[i]));
  const O = outsideIdx.length ? median(outsideIdx.map((i) => s[i])) : Math.min(...s);
  const thr = O + 0.3 * (I - O);
  const centre = (a + b) / 2;
  const ci = Math.min(s.length - 1, Math.max(0, Math.floor((centre - origin) / T) - k0));
  let lo = ci, hi = ci;
  if (nFixed) {
    let bestSum = -Infinity;
    for (let st = Math.max(0, ci - nFixed + 1); st <= ci && st + nFixed <= s.length; st++) {
      let sum = 0;
      for (let i = st; i < st + nFixed; i++) sum += Math.min(s[i], I);
      if (sum > bestSum) { bestSum = sum; lo = st; hi = st + nFixed - 1; }
    }
    if (bestSum === -Infinity) return null;
  } else {
    // A cell belongs to the visible grid if it looks like studs AND its stud centre lies
    // inside the visible window [a,b] (the frame may hide part of the outer row).
    const centreOk = (i: number) => {
      const c = origin + (k0 + i + 0.5) * T;
      return c >= a + T * CENTRE_MARGIN && c <= b - T * CENTRE_MARGIN;
    };
    while (lo > 0 && s[lo - 1] > thr && centreOk(lo - 1)) lo--;
    while (hi < s.length - 1 && s[hi + 1] > thr && centreOk(hi + 1)) hi++;
  }
  // visible fraction of the first/last cells (diagnostics)
  const visLo = (origin + (k0 + lo + 1) * T - a) / T;
  const visHi = (b - (origin + (k0 + hi) * T)) / T;
  return { start: origin + (k0 + lo) * T, n: hi - lo + 1, contrast: (I - O) / Math.max(1e-9, I), visLo, visHi };
}

/** Gradient-magnitude image (|gx|+|gy|). */
export function gradMag(g: Float32Array, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      out[i] = Math.abs(g[i + 1] - g[i - 1]) + Math.abs(g[i + w] - g[i - w]);
    }
  }
  return out;
}

/**
 * Stud-centre score for a lattice hypothesis: mean gradient on a ring of radius ~0.3 pitch
 * (the stud outline) minus mean gradient at the ring's half radius, sampled at the
 * hypothesised cell centres. Higher = cell centres sit on studs.
 */
export function ringScore(
  G: Float32Array, w: number, h: number,
  ox: number, Tx: number, oy: number, Ty: number,
  inner: [number, number, number, number],
): number {
  const [xa, xb, ya, yb] = inner;
  const N = 16;
  const cosT: number[] = [], sinT: number[] = [];
  for (let i = 0; i < N; i++) { cosT.push(Math.cos((2 * Math.PI * i) / N)); sinT.push(Math.sin((2 * Math.PI * i) / N)); }
  let acc = 0, cnt = 0;
  const kx0 = Math.ceil((xa + Tx - ox) / Tx), ky0 = Math.ceil((ya + Ty - oy) / Ty);
  for (let ky = ky0; oy + (ky + 1) * Ty <= yb - Ty; ky++) {
    const cy = oy + (ky + 0.5) * Ty;
    for (let kx = kx0; ox + (kx + 1) * Tx <= xb - Tx; kx++) {
      const cx = ox + (kx + 0.5) * Tx;
      let ring = 0, mid = 0;
      for (let i = 0; i < N; i++) {
        const rx = Math.round(cx + cosT[i] * 0.3 * Tx), ry = Math.round(cy + sinT[i] * 0.3 * Ty);
        const mx = Math.round(cx + cosT[i] * 0.15 * Tx), my = Math.round(cy + sinT[i] * 0.15 * Ty);
        if (rx < 0 || ry < 0 || rx >= w || ry >= h) continue;
        ring += G[ry * w + rx];
        mid += G[my * w + mx];
      }
      acc += (ring - mid) / N;
      cnt++;
    }
  }
  return cnt ? acc / cnt : 0;
}

function wrapHalf(d: number, T: number): number {
  let r = d % T;
  if (r > T / 2) r -= T;
  if (r < -T / 2) r += T;
  return r;
}

/**
 * Refine start/period with n fixed using a band profile `p`, relative to the full profile
 * `pAll` (so we never need to know whether the fundamental peaks on boundaries or centres).
 */
export function refineAxis(
  p: Float64Array, pAll: Float64Array, start: number, period: number, n: number,
): { start: number; period: number } {
  const a = Math.max(0, Math.round(start + period * 0.3));
  const b = Math.min(p.length, Math.round(start + n * period - period * 0.3));
  if (b - a < period * 3) return { start, period };
  const g = findPeriod(pAll, a, b, period * 0.97, period * 1.03);
  const dG = wrapHalf(findPhase(pAll, a, b, g.T) - start, g.T);
  const { T } = findPeriod(p, a, b, g.T * 0.98, g.T * 1.02);
  // phase relative to the centre of the band range so period errors don't skew it
  const mid = (a + b) / 2;
  const dB = wrapHalf(findPhase(p, a, b, T) - start, T);
  const shift = wrapHalf(dB - dG, T);
  if (Math.abs(shift) > T * 0.3) return { start, period: g.T };
  // the lattice is anchored where the phase is measured best (band centre): keep the line
  // nearest the centre fixed when changing the period.
  const s0 = start + shift;
  const kMid = Math.round((mid - s0) / g.T);
  const anchor = s0 + kMid * g.T;
  return { start: anchor - kMid * T, period: T };
}

/** Position of the strongest edge (frame inner edge) in profile p within [a,b]. */
export function frameEdge(p: Float64Array, a: number, b: number): number {
  let bi = -1, bv = -Infinity;
  for (let x = Math.max(1, Math.round(a)); x <= Math.min(p.length - 2, Math.round(b)); x++) {
    const v = p[x - 1] + p[x] + p[x + 1];
    if (v > bv) { bv = v; bi = x; }
  }
  return bi;
}

function dist(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function lineIntersect(p1: Point, p2: Point, p3: Point, p4: Point): Point {
  const d = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]);
  if (Math.abs(d) < 1e-9) return p1;
  const t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / d;
  return [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])];
}

function sampleMat(warped: RgbaImage, ix: [number, number], iy: [number, number], m: number): [number, number, number] | null {
  const { width: w, height: h, data } = warped;
  const rs: number[] = [], gs: number[] = [], bs: number[] = [];
  for (let y = 0; y < h; y += 3) {
    for (let x = 0; x < w; x += 3) {
      const outside = x < ix[0] - m * 0.2 || x > ix[1] + m * 0.2 || y < iy[0] - m * 0.2 || y > iy[1] + m * 0.2;
      const near = x > ix[0] - m * 0.9 && x < ix[1] + m * 0.9 && y > iy[0] - m * 0.9 && y < iy[1] + m * 0.9;
      if (!outside || !near) continue;
      const k = (y * w + x) * 4;
      const r = data[k], g = data[k + 1], bl = data[k + 2];
      const mx = Math.max(r, g, bl), mn = Math.min(r, g, bl);
      if (mx < 120 || mx > 252 || mx - mn > 40) continue;
      rs.push(r); gs.push(g); bs.push(bl);
    }
  }
  if (rs.length < 50) return null;
  return [median(rs), median(gs), median(bs)];
}

/** Fit the stud grid for `quad` (source px) on the full-resolution source image. */
export function fitGrid(src: RgbaImage, quad: Quad, opts: FitGridOptions = {}): GridFitResult {
  const debug: Record<string, number | string> = {};
  // ---- pass 1: coarse, with mat margin; aspect follows the quad (visible window may be 15x16)
  const qw = (dist(quad[0], quad[1]) + dist(quad[3], quad[2])) / 2;
  const qh = (dist(quad[0], quad[3]) + dist(quad[1], quad[2])) / 2;
  const sc = 800 / Math.max(qw, qh, 1);
  const innerW = Math.round(qw * sc), innerH = Math.round(qh * sc);
  const m1 = 120;
  const W1 = innerW + 2 * m1, H1 = innerH + 2 * m1;
  const w1 = warpQuad(src, quad, W1, H1, m1, 1);
  const g1 = luma(w1.img);
  const pr1 = gradientProfiles(g1, W1, H1, { x0: m1, x1: W1 - m1, y0: m1, y1: H1 - m1 });
  const innerX: [number, number] = [m1, W1 - m1];
  const innerY: [number, number] = [m1, H1 - m1];
  const ax = axisPeriod(pr1.px, innerX, opts.gridOverride?.cols);
  const ay = axisPeriod(pr1.py, innerY, opts.gridOverride?.rows);
  const matRgb = sampleMat(w1.img, innerX, innerY, m1);
  // Phase: the fundamental peaks either on cell boundaries or on stud centres depending on
  // lighting; pick the lattice whose cell centres sit on stud outlines.
  const G1 = gradMag(g1, W1, H1);
  const box: [number, number, number, number] = [m1, W1 - m1, m1, H1 - m1];
  let bestR = -Infinity, ox = ax.phase, oy = ay.phase;
  for (const hx of [0, 0.5]) {
    for (const hy of [0, 0.5]) {
      const r = ringScore(G1, W1, H1, ax.phase + hx * ax.T, ax.T, ay.phase + hy * ay.T, ay.T, box);
      if (r > bestR) { bestR = r; ox = ax.phase + hx * ax.T; oy = ay.phase + hy * ay.T; }
    }
  }
  const fx = axisRun(pr1.ex, innerX, ox, ax.T, opts.gridOverride?.cols);
  const fy = axisRun(pr1.ey, innerY, oy, ay.T, opts.gridOverride?.rows);
  if (!fx || !fy) return failed(src, quad, matRgb, { reason: 'no periodic structure' });
  debug.p1 = `x:${fx.n}@${ax.T.toFixed(1)} pk${ax.peak.toFixed(1)} c${fx.contrast.toFixed(2)} y:${fy.n}@${ay.T.toFixed(1)} pk${ay.peak.toFixed(1)} c${fy.contrast.toFixed(2)}`;

  // Visible grid = studs whose centre is visible: locate the frame's inner edge (strongest
  // straight edge near each border) and drop an edge cell that is less than half visible.
  if (!opts.gridOverride) {
    const trim = (f: { start: number; n: number }, p: Float64Array, T: number, tag: string) => {
      const lo = frameEdge(p, f.start - 0.4 * T, f.start + 0.9 * T);
      const hiB = f.start + f.n * T;
      const hi = frameEdge(p, hiB - 0.9 * T, hiB + 0.4 * T);
      const vLo = (f.start + T - lo) / T, vHi = (hi - (hiB - T)) / T;
      debug['vis' + tag] = `${vLo.toFixed(2)} ${vHi.toFixed(2)}`;
      if (vLo < EDGE_VISIBLE && f.n > MIN_STUDS) { f.start += T; f.n -= 1; }
      if (vHi < EDGE_VISIBLE && f.n > MIN_STUDS) f.n -= 1;
    };
    trim(fx, pr1.px, ax.T, 'X');
    trim(fy, pr1.py, ay.T, 'Y');
  }
  const cols = fx.n, rows = fy.n;
  if (cols < 2 || rows < 2) return failed(src, quad, matRgb, { reason: 'grid too small' });
  const quad1: Quad = [
    applyH(w1.toSrc, [fx.start, fy.start]),
    applyH(w1.toSrc, [fx.start + cols * ax.T, fy.start]),
    applyH(w1.toSrc, [fx.start + cols * ax.T, fy.start + rows * ay.T]),
    applyH(w1.toSrc, [fx.start, fy.start + rows * ay.T]),
  ];

  // ---- pass 2: refine with n fixed, per half-band (removes residual keystone)
  const P2 = 40;
  const m2 = P2;
  const S2w = cols * P2 + 2 * m2, S2h = rows * P2 + 2 * m2;
  const w2 = warpQuad(src, quad1, S2w, S2h, m2, 1);
  const g2 = luma(w2.img);
  const all2 = gradientProfiles(g2, S2w, S2h, { x0: m2, x1: S2w - m2, y0: m2, y1: S2h - m2 });
  const halves = (lo: number, hi: number): [number, number][] => [[lo, (lo + hi) / 2], [(lo + hi) / 2, hi]];
  const xs: { y: number; start: number; end: number }[] = [];
  for (const [y0, y1] of halves(m2, S2h - m2)) {
    const pr = gradientProfiles(g2, S2w, S2h, { x0: m2, x1: S2w - m2, y0: Math.round(y0), y1: Math.round(y1) });
    const r = refineAxis(pr.px, all2.px, m2, P2, cols);
    xs.push({ y: (y0 + y1) / 2, start: r.start, end: r.start + cols * r.period });
  }
  const ys: { x: number; start: number; end: number }[] = [];
  for (const [x0, x1] of halves(m2, S2w - m2)) {
    const pr = gradientProfiles(g2, S2w, S2h, { x0: Math.round(x0), x1: Math.round(x1), y0: m2, y1: S2h - m2 });
    const r = refineAxis(pr.py, all2.py, m2, P2, rows);
    ys.push({ x: (x0 + x1) / 2, start: r.start, end: r.start + rows * r.period });
  }
  const left: [Point, Point] = [[xs[0].start, xs[0].y], [xs[1].start, xs[1].y]];
  const right: [Point, Point] = [[xs[0].end, xs[0].y], [xs[1].end, xs[1].y]];
  const top: [Point, Point] = [[ys[0].x, ys[0].start], [ys[1].x, ys[1].start]];
  const bottom: [Point, Point] = [[ys[0].x, ys[0].end], [ys[1].x, ys[1].end]];
  const c2: Point[] = [
    lineIntersect(left[0], left[1], top[0], top[1]),
    lineIntersect(right[0], right[1], top[0], top[1]),
    lineIntersect(right[0], right[1], bottom[0], bottom[1]),
    lineIntersect(left[0], left[1], bottom[0], bottom[1]),
  ];
  const exp: Point[] = [[m2, m2], [S2w - m2, m2], [S2w - m2, S2h - m2], [m2, S2h - m2]];
  const maxDev = Math.max(...c2.map((c, i) => Math.hypot(c[0] - exp[i][0], c[1] - exp[i][1])));
  debug.refineDev = +(maxDev / P2).toFixed(2);
  const refined = maxDev < P2 * 0.8;
  const finalQuad: Quad = (refined ? c2 : exp).map((c) => applyH(w2.toSrc, c)) as Quad;

  // ---- final warp: square cells, cols*pitch x rows*pitch
  const srcPitch = (dist(finalQuad[0], finalQuad[1]) / cols + dist(finalQuad[0], finalQuad[3]) / rows) / 2;
  const pitch = opts.pitch ?? Math.max(24, Math.min(64, Math.floor(srcPitch)));
  const width = cols * pitch, height = rows * pitch;
  const wf = warpQuad(src, finalQuad, width, height, 0);

  // ---- confidence
  const peak = Math.min(ax.peak, ay.peak);
  const contrast = Math.min(fx.contrast, fy.contrast);
  // periods relative to the cell counts: a square mosaic in a square warp has equal pitch
  // cells are square in the world; the pass-1 warp preserved the quad's aspect
  const periodAgree = 1 - Math.min(1, Math.abs(ax.T - ay.T) / Math.max(ax.T, ay.T) / 0.08);
  const inRange = cols >= MIN_STUDS && cols <= MAX_STUDS && rows >= MIN_STUDS && rows <= MAX_STUDS;
  let confidence = clamp01((peak - 1.5) / 3) * 0.4 + clamp01(contrast / 0.5) * 0.3 + periodAgree * 0.3;
  // visible windows are often 15x16; larger differences are suspicious for these mosaics
  if (Math.abs(cols - rows) > 2 && !opts.gridOverride) confidence *= 0.75;
  if (!inRange) confidence = 0;
  if (!refined) confidence *= 0.85;
  debug.peak = +peak.toFixed(2);
  debug.contrast = +contrast.toFixed(2);
  debug.periodAgree = +periodAgree.toFixed(2);
  debug.ring = +bestR.toFixed(2);
  const grid: Grid = { cols, rows, corners: finalQuad, warpedSize: width, warpedWidth: width, warpedHeight: height, pitch, confidence };
  return { ok: confidence >= 0.35, grid, warped: wf.img, toSrc: wf.toSrc, matRgb, debug };
}

function failed(src: RgbaImage, quad: Quad, matRgb: [number, number, number] | null, debug: Record<string, string>): GridFitResult {
  const size = 512;
  const wf = warpQuad(src, quad, size, size, 0);
  return {
    ok: false,
    grid: { cols: 0, rows: 0, corners: quad, warpedSize: size, pitch: 0, confidence: 0 },
    warped: wf.img,
    toSrc: wf.toSrc,
    matRgb,
    debug,
  };
}

/** Build a Grid directly from known corners and size (no fitting) — for overrides/tests. */
export function gridFromCorners(src: RgbaImage, corners: Quad, size: GridSize, pitch = 48): { grid: Grid; warped: RgbaImage } {
  const W = size.cols * pitch, H = size.rows * pitch;
  const wf = warpQuad(src, corners, W, H, 0);
  return { grid: { ...size, corners, warpedSize: W, warpedWidth: W, warpedHeight: H, pitch, confidence: 1 }, warped: wf.img };
}
