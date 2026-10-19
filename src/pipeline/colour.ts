// Per-cell colour: robust median sampling, white balance from the mat, Lab, clustering,
// nearest palette colour by CIEDE2000.
import { PALETTE } from '../lib/palette';
import type { Grid, RgbaImage, StudCell } from './types';

export type Lab = [number, number, number];

const srgbToLin = (c: number): number => {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
};
const linToSrgb = (v: number): number => {
  const c = v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(Math.max(0, v), 1 / 2.4) - 0.055;
  return Math.max(0, Math.min(255, c * 255));
};

export function rgbToLab(r: number, g: number, b: number): Lab {
  const R = srgbToLin(r), G = srgbToLin(g), B = srgbToLin(b);
  const X = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.95047;
  const Y = 0.2126729 * R + 0.7151522 * G + 0.072175 * B;
  const Z = (0.0193339 * R + 0.119192 * G + 0.9503041 * B) / 1.08883;
  const f = (t: number) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116);
  const fx = f(X), fy = f(Y), fz = f(Z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

export function deltaE76(a: Lab, b: Lab): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

/** CIEDE2000 colour difference. */
export function deltaE2000(l1: Lab, l2: Lab): number {
  const [L1, a1, b1] = l1, [L2, a2, b2] = l2;
  const rad = Math.PI / 180;
  const C1 = Math.hypot(a1, b1), C2 = Math.hypot(a2, b2);
  const Cm = (C1 + C2) / 2;
  const G = 0.5 * (1 - Math.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)));
  const a1p = a1 * (1 + G), a2p = a2 * (1 + G);
  const C1p = Math.hypot(a1p, b1), C2p = Math.hypot(a2p, b2);
  const h = (b: number, a: number) => { if (b === 0 && a === 0) return 0; const t = Math.atan2(b, a) / rad; return t < 0 ? t + 360 : t; };
  const h1p = h(b1, a1p), h2p = h(b2, a2p);
  const dLp = L2 - L1, dCp = C2p - C1p;
  let dhp = 0;
  if (C1p * C2p !== 0) {
    dhp = h2p - h1p;
    if (dhp > 180) dhp -= 360; else if (dhp < -180) dhp += 360;
  }
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin((dhp * rad) / 2);
  const Lpm = (L1 + L2) / 2, Cpm = (C1p + C2p) / 2;
  let hpm = h1p + h2p;
  if (C1p * C2p !== 0) {
    if (Math.abs(h1p - h2p) > 180) hpm = h1p + h2p < 360 ? (h1p + h2p + 360) / 2 : (h1p + h2p - 360) / 2;
    else hpm = (h1p + h2p) / 2;
  }
  const T = 1 - 0.17 * Math.cos((hpm - 30) * rad) + 0.24 * Math.cos(2 * hpm * rad) + 0.32 * Math.cos((3 * hpm + 6) * rad) - 0.2 * Math.cos((4 * hpm - 63) * rad);
  const dTheta = 30 * Math.exp(-(((hpm - 275) / 25) ** 2));
  const Rc = 2 * Math.sqrt(Cpm ** 7 / (Cpm ** 7 + 25 ** 7));
  const Sl = 1 + (0.015 * (Lpm - 50) ** 2) / Math.sqrt(20 + (Lpm - 50) ** 2);
  const Sc = 1 + 0.045 * Cpm, Sh = 1 + 0.015 * Cpm * T;
  const Rt = -Math.sin(2 * dTheta * rad) * Rc;
  return Math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh));
}

export const PALETTE_LAB: { id: string; lab: Lab }[] = PALETTE.map((p) => ({ id: p.id, lab: rgbToLab(...p.rgb) }));

/**
 * Measured appearance of plates photographed through picture-frame glass, after the mat is
 * normalised to MAT_TARGET (glass haze lifts dark colours and desaturates). Median Lab over
 * the golden photos whose grid matched the labels (01,03,05,06,07). Used as extra reference
 * points next to the palette's nominal colours. Re-derive when new labelled photos arrive.
 */
export const GLASS_LAB: { id: string; lab: Lab }[] = [
  { id: 'black', lab: [25.3, -0.6, -9.1] },
  { id: 'blue', lab: [47.8, 13.2, -55.2] },
  { id: 'green', lab: [48.8, -38.7, 13.9] },
  { id: 'medium_azure', lab: [73.8, -18.7, -23.5] },
  { id: 'red', lab: [47.0, 49.2, 14.0] },
  { id: 'reddish_brown', lab: [33.9, 14.1, 3.0] },
  { id: 'tan', lab: [84.4, -1.4, 22.5] },
  { id: 'white', lab: [96.3, -1.6, 1.8] },
  { id: 'yellow', lab: [85.1, -3.8, 67.6] },
];

const REFS = [...PALETTE_LAB, ...GLASS_LAB];

export function nearestPalette(lab: Lab): { id: string; dE: number; second: number } {
  const best = new Map<string, number>();
  for (const p of REFS) {
    const d = deltaE2000(lab, p.lab);
    if (d < (best.get(p.id) ?? Infinity)) best.set(p.id, d);
  }
  let id = '', bd = Infinity, sd = Infinity;
  for (const [k, d] of best) {
    if (d < bd) { sd = bd; bd = d; id = k; } else if (d < sd) sd = d;
  }
  return { id, dE: bd, second: sd };
}

/** ΔE2000 from `lab` to the closest reference of palette colour `id`. */
export function distanceTo(lab: Lab, id: string): number {
  let d = Infinity;
  for (const p of REFS) if (p.id === id) d = Math.min(d, deltaE2000(lab, p.lab));
  return d;
}

/** The white mat is mapped to this sRGB level (palette white is 244). */
export const MAT_TARGET = 244;

/**
 * Per-channel gains (linear light) mapping the mat to neutral MAT_TARGET: corrects white
 * balance and exposure in one go (the mat is the only guaranteed white reference).
 */
export function whiteBalanceGains(mat: [number, number, number] | null): [number, number, number] {
  if (!mat) return [1, 1, 1];
  const target = srgbToLin(MAT_TARGET);
  return mat.map((v) => {
    const l = srgbToLin(v);
    return l > 1e-4 ? Math.max(0.7, Math.min(1.6, target / l)) : 1;
  }) as [number, number, number];
}

export function applyGains(rgb: [number, number, number], g: [number, number, number]): [number, number, number] {
  return [linToSrgb(srgbToLin(rgb[0]) * g[0]), linToSrgb(srgbToLin(rgb[1]) * g[1]), linToSrgb(srgbToLin(rgb[2]) * g[2])];
}

function histMedian(h: Uint32Array, n: number): number {
  let acc = 0;
  const half = n / 2;
  for (let i = 0; i < 256; i++) { acc += h[i]; if (acc >= half) return i; }
  return 255;
}

/**
 * Median colour of a cell's plate surface: stud top (r < 0.2 pitch) and the cell corners
 * outside the stud outline (r > 0.4 pitch, but at least 0.08 pitch from the cell edge).
 * Glare pixels are excluded unless nothing else is left.
 */
export function cellMedian(
  img: RgbaImage, x0: number, y0: number, cw: number, ch: number, glare?: Uint8Array,
): { rgb: [number, number, number]; glare: number } {
  const hr = new Uint32Array(256), hg = new Uint32Array(256), hb = new Uint32Array(256);
  const ar = new Uint32Array(256), ag = new Uint32Array(256), ab = new Uint32Array(256);
  let n = 0, na = 0, gl = 0, tot = 0;
  const step = Math.max(1, Math.round(Math.min(cw, ch) / 24));
  const cx = x0 + cw / 2, cy = y0 + ch / 2;
  const W = img.width, d = img.data;
  const ya = Math.max(0, Math.floor(y0)), yb = Math.min(img.height, Math.ceil(y0 + ch));
  const xa = Math.max(0, Math.floor(x0)), xb = Math.min(W, Math.ceil(x0 + cw));
  for (let y = ya; y < yb; y += step) {
    for (let x = xa; x < xb; x += step) {
      const i = y * W + x;
      tot++;
      const isG = glare ? glare[i] > 0 : false;
      if (isG) gl++;
      const dx = (x + 0.5 - cx) / cw, dy = (y + 0.5 - cy) / ch;
      const r = Math.hypot(dx, dy);
      const edge = Math.max(Math.abs(dx), Math.abs(dy));
      const ok = r < 0.2 || (r > 0.4 && edge < 0.42);
      if (!ok || edge > 0.45) continue;
      const k = i * 4;
      ar[d[k]]++; ag[d[k + 1]]++; ab[d[k + 2]]++; na++;
      if (isG) continue;
      hr[d[k]]++; hg[d[k + 1]]++; hb[d[k + 2]]++; n++;
    }
  }
  const useAll = n < Math.max(8, na * 0.15);
  const [R, G, B, N] = useAll ? [ar, ag, ab, na] : [hr, hg, hb, n];
  return { rgb: [histMedian(R, N), histMedian(G, N), histMedian(B, N)], glare: tot ? gl / tot : 0 };
}

export interface ColourOptions {
  matRgb?: [number, number, number] | null;
  /** ΔE76 radius for in-image colour clustering. 0 disables clustering. */
  clusterRadius?: number;
}

/** Compute StudCell[] (row-major) for a warped image and grid. */
export function classifyCells(
  warped: RgbaImage, grid: Pick<Grid, 'cols' | 'rows' | 'warpedSize'>, glare?: Uint8Array, opts: ColourOptions = {},
): StudCell[] {
  const cw = warped.width / grid.cols, ch = warped.height / grid.rows;
  const gains = whiteBalanceGains(opts.matRgb ?? null);
  const cells: StudCell[] = [];
  for (let row = 0; row < grid.rows; row++) {
    for (let col = 0; col < grid.cols; col++) {
      const m = cellMedian(warped, col * cw, row * ch, cw, ch, glare);
      const rgb = applyGains(m.rgb, gains);
      const lab = rgbToLab(rgb[0], rgb[1], rgb[2]);
      cells.push({
        col, row,
        rgb: { r: Math.round(m.rgb[0]), g: Math.round(m.rgb[1]), b: Math.round(m.rgb[2]) },
        lab: [lab[0], lab[1], lab[2]],
        colorId: '',
        glare: m.glare,
      });
    }
  }
  assignPalette(cells, opts.clusterRadius ?? 6);
  return cells;
}

/**
 * Cluster cell colours in Lab (leader clustering + 2 refinement passes) and map every
 * cluster centroid to its nearest palette colour, so photo noise doesn't flip single
 * cells between neighbouring palette entries.
 */
export function assignPalette(cells: StudCell[], radius: number): void {
  if (radius <= 0) {
    for (const c of cells) c.colorId = nearestPalette(c.lab).id;
    return;
  }
  let cents: Lab[] = [];
  const assign = new Int32Array(cells.length);
  for (let it = 0; it < 3; it++) {
    if (it === 0) {
      cells.forEach((c, i) => {
        let bi = -1, bd = radius;
        cents.forEach((m, j) => { const d = deltaE76(c.lab, m); if (d < bd) { bd = d; bi = j; } });
        if (bi < 0) { cents.push([...c.lab] as Lab); bi = cents.length - 1; }
        assign[i] = bi;
      });
    } else {
      cells.forEach((c, i) => {
        let bi = 0, bd = Infinity;
        cents.forEach((m, j) => { const d = deltaE76(c.lab, m); if (d < bd) { bd = d; bi = j; } });
        // a cell far from every centroid keeps its own identity
        assign[i] = bi;
      });
    }
    const sums = cents.map(() => [0, 0, 0, 0]);
    cells.forEach((c, i) => { const s = sums[assign[i]]; s[0] += c.lab[0]; s[1] += c.lab[1]; s[2] += c.lab[2]; s[3]++; });
    cents = sums.map((s, j) => (s[3] ? [s[0] / s[3], s[1] / s[3], s[2] / s[3]] : cents[j]) as Lab);
  }
  const clusterColour = cents.map((m) => nearestPalette(m).id);
  cells.forEach((c, i) => {
    const own = nearestPalette(c.lab);
    const cl = clusterColour[assign[i]];
    // keep the cluster's colour unless the cell itself is clearly closer to another one
    const dCl = distanceTo(c.lab, cl);
    c.colorId = own.id !== cl && dCl - own.dE > 6 ? own.id : cl;
  });
}
