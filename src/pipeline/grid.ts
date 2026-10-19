// Stud-grid fitting on a rectified image. Works for different framings (portrait shots with
// a wide mat, landscape close-ups where the mosaic is cut by the photo edge) without tuning.
//
// 1. Warp the detected quad with a wide margin (pixels outside the photo are flagged).
// 2. Per axis, find the stud pitch from a harmonic Fourier score of gradient projections over
//    the quad interior; pick the lattice phase whose cell centres sit on stud outlines.
// 3. Count cells from image evidence, not from the quad: each lattice strip's gradient signal
//    folded onto the other axis' stud lattice is strong for studs and flat for mat/lines.
//    The run of stud strips around the centre is the grid; edge cells only partly visible
//    under the frame are trimmed (frame-edge position + stud evidence at the cell centre).
//    This runs twice: the second time re-rectified on the first run.
// 4. Refine per half-band (residual keystone), warp at a fixed pitch.
// 5. Confidence from framing-independent checks (lattice residual, square studs, grid vs quad
//    aspect, edge contrast, open-ended runs); low confidence -> grid_failed, never a guess.
import { applyH, clamp01, luma, median, warpQuad } from './image';
import type { Grid, GridSize, Point, Quad, RgbaImage } from './types';

export const MIN_STUDS = 8;
export const MAX_STUDS = 48;
/** Below this grid confidence the scan reports grid_failed instead of guessing. */
export const GRID_MIN_CONFIDENCE = 0.55;
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

/** Period, peak strength and fundamental phase for one axis. */
export interface AxisPeriod {
  T: number;
  peak: number;
  /** position of the fundamental's maximum, mod T (either cell boundary or cell centre). */
  phase: number;
}

export function axisPeriod(p: Float64Array, inner: [number, number], nFixed?: number, nExpect?: number): AxisPeriod {
  const [a, b] = inner;
  const span = b - a;
  let tMin = span / (MAX_STUDS + 2), tMax = span / (MIN_STUDS - 1.5);
  if (nFixed) { tMin = (span / nFixed) * 0.85; tMax = (span / nFixed) * 1.15; }
  else if (nExpect) { tMin = (span / nExpect) * 0.92; tMax = (span / nExpect) * 1.08; }
  // analyse a slightly shrunken inner range (outer row may be partly under the mat)
  const ia = Math.round(a + span * 0.04), ib = Math.round(b - span * 0.04);
  const { T, peak } = findPeriod(p, ia, ib, tMin, tMax);
  return { T, peak, phase: findPhase(p, ia, ib, T) };
}

/**
 * Stud evidence of every lattice strip along one axis. For x strips [o+kT, o+(k+1)T) (or the
 * sub-range `sub` of each strip) the signal s(y) = sum over the strip of |dI/dy| (rows inside
 * `range`) is folded onto the stud lattice of the other axis (phase oOther, period ~Tother).
 * Studs give a strong folded profile (std/mean high); mat, a frame bevel line or glare edges
 * fold to a flat profile. Also returns, per strip, the fraction of it (along the axis, on the
 * centre line) inside the photo.
 */
export function stripPeriodicity(
  g: Float32Array, valid: Uint8Array, w: number, h: number, axis: 'x' | 'y',
  origin: number, T: number, oOther: number, Tother: number, range: [number, number],
  sub: [number, number] = [0.1, 0.9],
): { k0: number; per: number[]; inPhoto: number[] } {
  const len = axis === 'x' ? w : h, along = axis === 'x' ? h : w;
  const k0 = Math.ceil((0 - origin) / T);
  const per: number[] = [], inPhoto: number[] = [];
  const NB = 8;
  const va = Math.max(1, Math.round(range[0])), vb = Math.min(along - 1, Math.round(range[1]));
  const cl = Math.min(along - 1, Math.max(0, Math.round((range[0] + range[1]) / 2)));
  const step = axis === 'x' ? w : 1;
  const sig = new Float64Array(along), cnt = new Float64Array(along);
  // a keystoned/loose quad makes the stud period drift a little along the strip: fold at the
  // best of a few periods around Tother (folding itself is phase-free)
  const periods = [0.96, 0.97, 0.98, 0.99, 1, 1.01, 1.02, 1.03, 1.04].map((f) => f * Tother);
  for (let k = k0; origin + (k + 1) * T <= len; k++) {
    const a = Math.round(origin + (k + sub[0]) * T);
    const b = Math.max(Math.round(origin + (k + sub[1]) * T), a + 2);
    sig.fill(0);
    cnt.fill(0);
    let nv = 0, nt = 0;
    for (let u = a; u < b; u++) {
      nt++;
      if (valid[axis === 'x' ? cl * w + u : u * w + cl]) nv++;
      for (let v = va; v < vb; v++) {
        const i = axis === 'x' ? v * w + u : u * w + v;
        if (!valid[i]) continue;
        sig[v] += Math.abs(g[i + step] - g[i - step]);
        cnt[v]++;
      }
    }
    let best = 0;
    for (const P of periods) {
      const sum = new Float64Array(NB), c = new Float64Array(NB);
      for (let v = va; v < vb; v++) {
        if (!cnt[v]) continue;
        let f = ((v + 0.5 - oOther) / P) % 1;
        if (f < 0) f += 1;
        const j = Math.min(NB - 1, Math.floor(f * NB));
        sum[j] += sig[v];
        c[j] += cnt[v];
      }
      let m = 0, nb = 0;
      const prof: number[] = [];
      for (let j = 0; j < NB; j++) if (c[j] > 0) { prof.push(sum[j] / c[j]); m += sum[j] / c[j]; nb++; }
      m /= Math.max(1, nb);
      if (nb < NB / 2 || m <= 0.5) continue;
      let sd = 0;
      for (const v of prof) sd += (v - m) ** 2;
      best = Math.max(best, Math.sqrt(sd / prof.length) / m);
    }
    per.push(best);
    inPhoto.push(nt ? nv / nt : 0);
  }
  return { k0, per, inPhoto };
}

export interface RunResult {
  start: number;
  n: number;
  /** Inside reference level (median evidence of strips inside the quad window). */
  level: number;
  /** Per side: (edge cell - outside strip) / (level - out); NaN when the photo edge stopped the run. */
  edgeContrast: [number, number];
  /** Per side: the run was stopped by the photo border. */
  truncated: [boolean, boolean];
  /** Per side: the run reached the end of the sampled area (extent unknown). */
  openEnded: [boolean, boolean];
  /** Strip index range of the (untrimmed) run, the evidence array and its "not studs" level. */
  idx: [number, number];
  per: number[];
  out: number;
}

/**
 * Contiguous run of stud strips around the quad centre: stud evidence above
 * outside + 30% of (inside - outside), and the stud centre inside the photo. Independent of
 * where the quad edges are, so a cropped, loose or shifted quad still gives the right count.
 */
export function periodicRun(
  st: { k0: number; per: number[]; inPhoto: number[] },
  origin: number, T: number, inner: [number, number], nFixed?: number,
): RunResult | null {
  const { k0, per, inPhoto } = st;
  const n = per.length;
  if (n < 3) return null;
  const [a, b] = inner;
  const ins: number[] = [], all: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = origin + (k0 + i + 0.5) * T;
    if (c > a + T && c < b - T) ins.push(per[i]);
    if (inPhoto[i] >= 0.5) all.push(per[i]);
  }
  if (ins.length < 2) return null;
  const level = median(ins);
  // "not studs" level: low percentile of every in-photo strip (the wide margin is mostly mat,
  // even when the quad is cropped), capped so a mosaic filling the photo still works
  all.sort((x, y) => x - y);
  const out = Math.min(all[Math.floor(all.length * 0.15)] ?? 0, level * 0.1);
  const thr = out + 0.3 * (level - out);
  const strong = (i: number) => i >= 0 && i < n && per[i] >= thr && inPhoto[i] >= 0.5;
  // a single weak row (e.g. low-contrast black plates) is bridged when the next two are studs
  // again and it is clearly above the mat level
  const weakOk = (i: number) => i >= 0 && i < n && inPhoto[i] >= 0.5 && per[i] >= Math.max(2 * out, 0.18 * level);
  const ok = (i: number, dir: number) => strong(i) || (weakOk(i) && strong(i + dir) && strong(i + 2 * dir));
  const ci = Math.min(n - 1, Math.max(0, Math.floor(((a + b) / 2 - origin) / T) - k0));
  let lo = ci, hi = ci;
  if (nFixed) {
    let best = -Infinity;
    for (let s0 = Math.max(0, ci - nFixed + 1); s0 <= ci && s0 + nFixed <= n; s0++) {
      let sum = 0;
      for (let i = s0; i < s0 + nFixed; i++) sum += Math.min(per[i], level) * (inPhoto[i] >= 0.5 ? 1 : 0);
      if (sum > best) { best = sum; lo = s0; hi = s0 + nFixed - 1; }
    }
    if (best === -Infinity) return null;
  } else {
    if (!strong(ci)) return null;
    while (ok(lo - 1, -1)) lo--;
    while (ok(hi + 1, 1)) hi++;
  }
  const side = (inner: number, outer: number): [number, boolean, boolean] => {
    if (outer < 0 || outer >= n) return [NaN, false, true]; // ran out of sampled area
    if (inPhoto[outer] < 0.5) return [NaN, true, false]; // photo edge
    return [(per[inner] - per[outer]) / (level - out || 1e-9), false, false];
  };
  const [cLo, tLo, eLo] = side(lo, lo - 1), [cHi, tHi, eHi] = side(hi, hi + 1);
  return {
    start: origin + (k0 + lo) * T, n: hi - lo + 1, level,
    edgeContrast: [cLo, cHi], truncated: [tLo, tHi], openEnded: [eLo, eHi],
    idx: [lo, hi], per, out,
  };
}

/**
 * Lattice residual on the rectified image: in each half-band the measured stud period must
 * equal the pitch. Returns the worst accumulated drift over the grid, in pitches
 * (|T_band/pitch - 1| * n). A wrong cell count shows up as a drift of ~1 pitch or more.
 */
export function latticeResidual(img: RgbaImage, cols: number, rows: number): number {
  const g = luma(img);
  const { width: w, height: h } = img;
  const p = w / cols;
  let worst = 0;
  for (const [y0, y1] of [[0, h / 2], [h / 2, h]]) {
    const pr = gradientProfiles(g, w, h, { x0: 0, x1: w, y0: Math.round(y0), y1: Math.round(y1) });
    const { T } = findPeriod(pr.px, Math.round(p * 0.3), Math.round(w - p * 0.3), p * 0.8, p * 1.25);
    worst = Math.max(worst, Math.abs(T / p - 1) * cols);
  }
  for (const [x0, x1] of [[0, w / 2], [w / 2, w]]) {
    const pr = gradientProfiles(g, w, h, { x0: Math.round(x0), x1: Math.round(x1), y0: 0, y1: h });
    const { T } = findPeriod(pr.py, Math.round(p * 0.3), Math.round(h - p * 0.3), p * 0.8, p * 1.25);
    worst = Math.max(worst, Math.abs(T / p - 1) * rows);
  }
  return worst;
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
  // Coarse pass: lattice period/phase from the quad interior, cell run from stud evidence,
  // optional visibility trim. Returns the grid boundary quad in source px.
  const coarse = (quad: Quad, doTrim: boolean, tagP: string, expect?: { cols: number; rows: number }) => {
    // ---- pass 1: coarse, aspect follows the quad, with a wide margin so the true mosaic is
    // inside even when the quad is cropped, loose or shifted by a couple of studs.
    const qw = (dist(quad[0], quad[1]) + dist(quad[3], quad[2])) / 2;
    const qh = (dist(quad[0], quad[3]) + dist(quad[1], quad[2])) / 2;
    const sc = 800 / Math.max(qw, qh, 1);
    const innerW = Math.round(qw * sc), innerH = Math.round(qh * sc);
    const m1 = 200;
    const W1 = innerW + 2 * m1, H1 = innerH + 2 * m1;
    const w1 = warpQuad(src, quad, W1, H1, m1, 1);
    const g1 = luma(w1.img);
    const valid = new Uint8Array(W1 * H1);
    for (let i = 0; i < valid.length; i++) valid[i] = w1.img.data[i * 4 + 3] > 0 ? 1 : 0;
    const pr1 = gradientProfiles(g1, W1, H1, { x0: m1, x1: W1 - m1, y0: m1, y1: H1 - m1 });
    const innerX: [number, number] = [m1, W1 - m1];
    const innerY: [number, number] = [m1, H1 - m1];
    const ax = axisPeriod(pr1.px, innerX, opts.gridOverride?.cols, expect?.cols);
    const ay = axisPeriod(pr1.py, innerY, opts.gridOverride?.rows, expect?.rows);
    const matRgb = sampleMat(w1.img, innerX, innerY, Math.round(m1 * 0.6));
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
    const sx = stripPeriodicity(g1, valid, W1, H1, 'x', ox, ax.T, oy, ay.T, innerY);
    const sy = stripPeriodicity(g1, valid, W1, H1, 'y', oy, ay.T, ox, ax.T, innerX);
    const fx = periodicRun(sx, ox, ax.T, innerX, opts.gridOverride?.cols);
    const fy = periodicRun(sy, oy, ay.T, innerY, opts.gridOverride?.rows);
    if (!fx || !fy) return { fail: 'no periodic stud structure', matRgb } as const;
    debug[tagP + 'p1'] = `x:${fx.n}@${ax.T.toFixed(1)} pk${ax.peak.toFixed(1)} y:${fy.n}@${ay.T.toFixed(1)} pk${ay.peak.toFixed(1)}`;
    debug[tagP + 'edges'] = [...fx.edgeContrast, ...fy.edgeContrast].map((v) => (Number.isNaN(v) ? 'cut' : v.toFixed(2))).join(' ');

    // Visible grid = studs whose centre is visible. Two cues per edge cell: (a) where the
    // frame's inner edge is (strongest straight edge near the border) -> visible fraction;
    // (b) stud evidence of the cell's central sub-strip relative to interior cells. Drop the
    // cell if it is clearly less than half visible and its centre doesn't look like a full
    // stud, or if both cues are borderline low.
    let ambiguousEdges = 0;
    if (doTrim && !opts.gridOverride) {
      const trim = (f: RunResult, p: Float64Array, T: number, tag: string, ce: { k0: number; per: number[] }, o: number) => {
        const lo = frameEdge(p, f.start - 0.4 * T, f.start + 0.9 * T);
        const hiB = f.start + f.n * T;
        const hi = frameEdge(p, hiB - 0.9 * T, hiB + 0.4 * T);
        const vLo = (f.start + T - lo) / T, vHi = (hi - (hiB - T)) / T;
        const i0 = Math.round((f.start - o) / T) - ce.k0;
        const ref = median(ce.per.slice(i0 + 2, i0 + f.n - 2)) || 1e-9;
        const cLo = (ce.per[i0] ?? 0) / ref, cHi = (ce.per[i0 + f.n - 1] ?? 0) / ref;
        const drop = (v: number, c: number) => (v < EDGE_VISIBLE && c < 1) || (v < 0.6 && c < 0.62);
        const ambiguous = (v: number, c: number) => v > 0.4 && v < 0.65 && c > 0.45 && c < 1.1;
        debug[tagP + 'vis' + tag] = `${vLo.toFixed(2)}/${cLo.toFixed(2)} ${vHi.toFixed(2)}/${cHi.toFixed(2)}`;
        ambiguousEdges += +ambiguous(vLo, cLo) + +ambiguous(vHi, cHi);
        const dLo = drop(vLo, cLo) && f.n > MIN_STUDS, dHi = drop(vHi, cHi) && f.n > MIN_STUDS;
        if (dLo) { f.start += T; f.n -= 1; }
        if (dHi) f.n -= 1;
        // edge evidence after trimming: first kept cell vs the first strip outside the run
        const [r0, r1] = f.idx, den = f.level - f.out || 1e-9;
        if (!Number.isNaN(f.edgeContrast[0])) f.edgeContrast[0] = (f.per[r0 + (dLo ? 1 : 0)] - f.per[r0 - 1]) / den;
        if (!Number.isNaN(f.edgeContrast[1])) f.edgeContrast[1] = (f.per[r1 - (dHi ? 1 : 0)] - f.per[r1 + 1]) / den;
      };
      const cx = stripPeriodicity(g1, valid, W1, H1, 'x', ox, ax.T, oy, ay.T, innerY, [0.38, 0.62]);
      const cy = stripPeriodicity(g1, valid, W1, H1, 'y', oy, ay.T, ox, ax.T, innerX, [0.38, 0.62]);
      trim(fx, pr1.px, ax.T, 'X', cx, ox);
      trim(fy, pr1.py, ay.T, 'Y', cy, oy);
    }
    const cols = fx.n, rows = fy.n;
    if (cols < 2 || rows < 2) return { fail: 'grid too small', matRgb } as const;
    const quad1: Quad = [
      applyH(w1.toSrc, [fx.start, fy.start]),
      applyH(w1.toSrc, [fx.start + cols * ax.T, fy.start]),
      applyH(w1.toSrc, [fx.start + cols * ax.T, fy.start + rows * ay.T]),
      applyH(w1.toSrc, [fx.start, fy.start + rows * ay.T]),
    ];
    return { fx, fy, ax, ay, qw, qh, matRgb, bestR, ambiguousEdges, cols, rows, quad1 } as const;
  };
  // Two iterations: the first finds the full stud run from a possibly poor quad; the second
  // re-rectifies on that run so edge visibility is judged in a well-rectified view.
  const r1 = coarse(quad, false, 'a_');
  if ('fail' in r1) return failed(src, quad, r1.matRgb, { reason: String(r1.fail) });
  // the first pass's run defines the second warp's window, so its cell count fixes the period
  const r2 = coarse(r1.quad1, true, '', { cols: r1.cols, rows: r1.rows });
  if ('fail' in r2) return failed(src, quad, r1.matRgb, { reason: String(r2.fail) });
  const { fx, fy, bestR, ambiguousEdges, cols, rows, quad1 } = r2;
  const { qw, qh } = r1;
  const matRgb = r1.matRgb ?? r2.matRgb;

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

  // ---- confidence (each check is framing-independent; any clear failure -> grid_failed)
  // periodicity strength from the wide-range search of the first iteration
  const peak = Math.min(r1.ax.peak, r1.ay.peak);
  // (1) lattice residual: the rectified studs must repeat at exactly the pitch everywhere
  const drift = latticeResidual(wf.img, cols, rows);
  // (2) square studs: source-space pitch along x and y must agree (wrong count on one axis
  //     stretches the cells); tolerance covers keystone/tilt.
  const pxs = (dist(finalQuad[0], finalQuad[1]) + dist(finalQuad[3], finalQuad[2])) / 2 / cols;
  const pys = (dist(finalQuad[0], finalQuad[3]) + dist(finalQuad[1], finalQuad[2])) / 2 / rows;
  const cellAspect = Math.abs(Math.log(pxs / pys));
  // (3) grid aspect vs the detected window's aspect (scale-invariant, so a loose quad is OK)
  const quadAspect = Math.abs(Math.log((cols / rows) / (qw / qh)));
  // (4) edge evidence: each non-cut side must step clearly from studs to not-studs
  const contrasts = [...fx.edgeContrast, ...fy.edgeContrast].filter((v) => !Number.isNaN(v));
  const minEdge = contrasts.length ? Math.min(...contrasts) : 0;
  const cutSides = [...fx.truncated, ...fy.truncated].filter(Boolean).length;
  const openSides = [...fx.openEnded, ...fy.openEnded].filter(Boolean).length;
  const inRange = cols >= MIN_STUDS && cols <= MAX_STUDS && rows >= MIN_STUDS && rows <= MAX_STUDS;
  const reasons: string[] = [];
  let confidence = 0.5 + 0.5 * clamp01((peak - 2) / 2.5);
  const penal = (cond: boolean, f: number, why: string) => { if (cond) { confidence *= f; reasons.push(why); } };
  penal(drift > 0.35, drift > 0.7 ? 0.2 : 0.6, `lattice drift ${drift.toFixed(2)} pitch`);
  penal(cellAspect > 0.06, cellAspect > 0.12 ? 0.2 : 0.6, `cell aspect ${cellAspect.toFixed(2)}`);
  penal(quadAspect > 0.12 && !opts.gridOverride, quadAspect > 0.2 ? 0.35 : 0.75, `grid/quad aspect ${quadAspect.toFixed(2)}`);
  penal(openSides > 0, 0.4, `${openSides} sides run past the sampled area`);
  penal(minEdge < 0.3, minEdge < 0.15 ? 0.5 : 0.85, `weak edge ${minEdge.toFixed(2)}`);
  penal(cutSides >= 2, 0.85, `${cutSides} sides cut by photo`);
  penal(!refined, 0.85, 'keystone refinement rejected');
  penal(ambiguousEdges > 0, Math.pow(0.92, ambiguousEdges), `${ambiguousEdges} half-visible edge rows/cols`);
  if (!inRange) { confidence = 0; reasons.push('size out of range'); }
  debug.peak = +peak.toFixed(2);
  debug.drift = +drift.toFixed(2);
  debug.cellAspect = +cellAspect.toFixed(3);
  debug.quadAspect = +quadAspect.toFixed(3);
  debug.minEdge = +minEdge.toFixed(2);
  debug.ring = +bestR.toFixed(2);
  if (reasons.length) debug.reasons = reasons.join('; ');
  const grid: Grid = { cols, rows, corners: finalQuad, warpedSize: width, warpedWidth: width, warpedHeight: height, pitch, confidence };
  return { ok: confidence >= GRID_MIN_CONFIDENCE, grid, warped: wf.img, toSrc: wf.toSrc, matRgb, debug };
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
