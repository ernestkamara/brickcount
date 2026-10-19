// Pure-TS image helpers (no OpenCV) so they run in workers, Node eval and unit tests.
import type { Point, Quad, RgbaImage } from './types';

export function makeImage(width: number, height: number): RgbaImage {
  return { width, height, data: new Uint8ClampedArray(width * height * 4) };
}

/** Area-average downscale so the longer side is <= maxSide. Returns the image and the scale (small = src * scale). */
export function downscale(img: RgbaImage, maxSide: number): { img: RgbaImage; scale: number } {
  const long = Math.max(img.width, img.height);
  if (long <= maxSide) return { img, scale: 1 };
  const scale = maxSide / long;
  const w = Math.max(1, Math.round(img.width * scale));
  const h = Math.max(1, Math.round(img.height * scale));
  const out = makeImage(w, h);
  const sx = img.width / w;
  const sy = img.height / h;
  const src = img.data;
  const dst = out.data;
  // Sample a fixed small grid inside each destination pixel (fast approximate box filter).
  const n = Math.min(2, Math.max(1, Math.floor(Math.min(sx, sy))));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0, a = 0;
      for (let j = 0; j < n; j++) {
        const yy = Math.min(img.height - 1, Math.floor(y * sy + ((j + 0.5) * sy) / n));
        for (let i = 0; i < n; i++) {
          const xx = Math.min(img.width - 1, Math.floor(x * sx + ((i + 0.5) * sx) / n));
          const k = (yy * img.width + xx) * 4;
          r += src[k]; g += src[k + 1]; b += src[k + 2]; a += src[k + 3];
        }
      }
      const nn = n * n;
      const o = (y * w + x) * 4;
      dst[o] = r / nn; dst[o + 1] = g / nn; dst[o + 2] = b / nn; dst[o + 3] = a / nn;
    }
  }
  return { img: out, scale };
}

/** Copy of the rectangle [x0,x0+w) x [y0,y0+h) (clamped to the image). */
export function crop(img: RgbaImage, x0: number, y0: number, w: number, h: number): RgbaImage {
  x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0));
  w = Math.min(img.width - x0, Math.ceil(w)); h = Math.min(img.height - y0, Math.ceil(h));
  const out = makeImage(w, h);
  for (let y = 0; y < h; y++) {
    const s = ((y0 + y) * img.width + x0) * 4;
    out.data.set(img.data.subarray(s, s + w * 4), y * w * 4);
  }
  return out;
}

/** Exact 2x2 box downscale (half resolution). Coordinates map as p_half = (p + 0.5) / 2 - 0.5. */
export function halve(img: RgbaImage): RgbaImage {
  const w = img.width >> 1, h = img.height >> 1;
  const out = makeImage(w, h);
  const s = img.data, d = out.data, sw4 = img.width * 4;
  for (let y = 0; y < h; y++) {
    let k = 2 * y * sw4;
    let o = y * w * 4;
    for (let x = 0; x < w; x++, k += 8, o += 4) {
      const k2 = k + sw4;
      d[o] = (s[k] + s[k + 4] + s[k2] + s[k2 + 4] + 2) >> 2;
      d[o + 1] = (s[k + 1] + s[k + 5] + s[k2 + 1] + s[k2 + 5] + 2) >> 2;
      d[o + 2] = (s[k + 2] + s[k + 6] + s[k2 + 2] + s[k2 + 6] + 2) >> 2;
      d[o + 3] = 255;
    }
  }
  return out;
}

/** 3x3 homography (row-major, h[8] = 1) mapping src points to dst points. */
export function homography(src: Point[], dst: Point[]): number[] {
  const A: number[][] = [];
  const bv: number[] = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    bv.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    bv.push(v);
  }
  const h = solve(A, bv);
  return [...h, 1];
}

function solve(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c] || 1e-12;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / d;
      if (f === 0) continue;
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, i) => row[n] / (row[i] || 1e-12));
}

export function applyH(h: number[], p: Point): Point {
  const [x, y] = p;
  const w = h[6] * x + h[7] * y + h[8];
  return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w];
}

/**
 * Warp `quad` (TL,TR,BR,BL in src px) to a rectangle of outW x outH whose inner region
 * [margin, outW-margin] x [margin, outH-margin] corresponds to the quad. Bilinear sampling;
 * when the source is much larger than the destination a 2x2 supersample reduces aliasing.
 * Returns the warped image and the homography warped -> src.
 */
export function warpQuad(
  src: RgbaImage,
  quad: Quad,
  outW: number,
  outH: number,
  margin = 0,
  maxSupersample = 2,
): { img: RgbaImage; toSrc: number[] } {
  const dstPts: Point[] = [
    [margin, margin],
    [outW - margin, margin],
    [outW - margin, outH - margin],
    [margin, outH - margin],
  ];
  const toSrc = homography(dstPts, quad);
  const out = makeImage(outW, outH);
  const sw = src.width, sh = src.height, sd = src.data, od = out.data;
  // Approximate source/dest scale to decide on supersampling.
  const sideSrc = Math.hypot(quad[1][0] - quad[0][0], quad[1][1] - quad[0][1]);
  const ss = maxSupersample > 1 && sideSrc / Math.max(1, outW - 2 * margin) > 1.6 ? 2 : 1;
  const offs = ss === 1 ? [0.5] : [0.25, 0.75];
  const [h0, h1, h2, h3, h4, h5, h6, h7, h8] = toSrc;
  if (ss === 1) {
    const sw4 = sw * 4;
    for (let y = 0; y < outH; y++) {
      const yy = y + 0.5;
      for (let x = 0; x < outW; x++) {
        const xx = x + 0.5;
        const w = h6 * xx + h7 * yy + h8;
        let fx = (h0 * xx + h1 * yy + h2) / w - 0.5;
        let fy = (h3 * xx + h4 * yy + h5) / w - 0.5;
        // alpha 0 marks samples outside the photo (clamped colour), so callers can ignore them
        const inside = fx >= -1 && fy >= -1 && fx <= sw && fy <= sh;
        if (fx < 0) fx = 0; else if (fx > sw - 1.001) fx = sw - 1.001;
        if (fy < 0) fy = 0; else if (fy > sh - 1.001) fy = sh - 1.001;
        const ix = fx | 0, iy = fy | 0;
        const ax = fx - ix, ay = fy - iy;
        const k = (iy * sw + ix) * 4;
        const k2 = k + sw4;
        const w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay;
        const o = (y * outW + x) * 4;
        od[o] = sd[k] * w00 + sd[k + 4] * w10 + sd[k2] * w01 + sd[k2 + 4] * w11;
        od[o + 1] = sd[k + 1] * w00 + sd[k + 5] * w10 + sd[k2 + 1] * w01 + sd[k2 + 5] * w11;
        od[o + 2] = sd[k + 2] * w00 + sd[k + 6] * w10 + sd[k2 + 2] * w01 + sd[k2 + 6] * w11;
        od[o + 3] = inside ? 255 : 0;
      }
    }
    return { img: out, toSrc };
  }
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      let r = 0, g = 0, b = 0;
      for (const oy of offs) {
        const yy = y + oy;
        for (const ox of offs) {
          const xx = x + ox;
          const w = h6 * xx + h7 * yy + h8;
          let fx = (h0 * xx + h1 * yy + h2) / w - 0.5;
          let fy = (h3 * xx + h4 * yy + h5) / w - 0.5;
          if (fx < 0) fx = 0; else if (fx > sw - 1.001) fx = sw - 1.001;
          if (fy < 0) fy = 0; else if (fy > sh - 1.001) fy = sh - 1.001;
          const ix = fx | 0, iy = fy | 0;
          const ax = fx - ix, ay = fy - iy;
          const k = (iy * sw + ix) * 4;
          const k2 = k + sw * 4;
          const w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay;
          r += sd[k] * w00 + sd[k + 4] * w10 + sd[k2] * w01 + sd[k2 + 4] * w11;
          g += sd[k + 1] * w00 + sd[k + 5] * w10 + sd[k2 + 1] * w01 + sd[k2 + 5] * w11;
          b += sd[k + 2] * w00 + sd[k + 6] * w10 + sd[k2 + 2] * w01 + sd[k2 + 6] * w11;
        }
      }
      const n = offs.length * offs.length;
      const o = (y * outW + x) * 4;
      od[o] = r / n; od[o + 1] = g / n; od[o + 2] = b / n; od[o + 3] = 255;
    }
  }
  return { img: out, toSrc };
}

/** Rec.601 luma as Float32Array. */
export function luma(img: RgbaImage): Float32Array {
  const n = img.width * img.height;
  const out = new Float32Array(n);
  const d = img.data;
  for (let i = 0, k = 0; i < n; i++, k += 4) out[i] = 0.299 * d[k] + 0.587 * d[k + 1] + 0.114 * d[k + 2];
  return out;
}

/** Bilinear sample of one channel set at (x, y) in pixel-centre coords; writes rgb into out at o. */
export function sampleRgb(img: RgbaImage, x: number, y: number, out: Float32Array | number[], o: number): void {
  const w = img.width, h = img.height, d = img.data;
  let fx = x - 0.5, fy = y - 0.5;
  if (fx < 0) fx = 0; else if (fx > w - 1.001) fx = w - 1.001;
  if (fy < 0) fy = 0; else if (fy > h - 1.001) fy = h - 1.001;
  const ix = fx | 0, iy = fy | 0;
  const ax = fx - ix, ay = fy - iy;
  const k = (iy * w + ix) * 4;
  const k2 = k + w * 4;
  for (let c = 0; c < 3; c++) {
    out[o + c] =
      d[k + c] * (1 - ax) * (1 - ay) + d[k + 4 + c] * ax * (1 - ay) + d[k2 + c] * (1 - ax) * ay + d[k2 + 4 + c] * ax * ay;
  }
}

/** Order 4 arbitrary corner points as TL, TR, BR, BL. */
export function orderCorners(pts: Point[]): Quad {
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  const sorted = [...pts].sort(
    (a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx),
  );
  // atan2 order (y down): starts near -pi (left), goes TL(-3pi/4), TR(-pi/4), BR(pi/4), BL(3pi/4).
  let tl = 0;
  for (let i = 1; i < 4; i++) if (sorted[i][0] + sorted[i][1] < sorted[tl][0] + sorted[tl][1]) tl = i;
  const q = [0, 1, 2, 3].map((i) => sorted[(tl + i) % 4]) as Quad;
  return q;
}

export function median(arr: ArrayLike<number>): number {
  if (arr.length === 0) return 0;
  const a = Array.from(arr).sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Otsu threshold for an array of values; returns threshold and separability (between/total variance). */
export function otsu(values: number[], bins = 64): { thr: number; sep: number } {
  if (values.length < 2) return { thr: values[0] ?? 0, sep: 0 };
  let lo = Infinity, hi = -Infinity;
  for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (hi - lo < 1e-9) return { thr: lo, sep: 0 };
  const hist = new Float64Array(bins);
  for (const v of values) hist[Math.min(bins - 1, Math.floor(((v - lo) / (hi - lo)) * bins))]++;
  const total = values.length;
  let sumAll = 0;
  for (let i = 0; i < bins; i++) sumAll += i * hist[i];
  const meanAll = sumAll / total;
  let varAll = 0;
  for (let i = 0; i < bins; i++) varAll += hist[i] * (i - meanAll) ** 2;
  varAll /= total;
  let wB = 0, sumB = 0, best = -1, bestI = 0;
  for (let i = 0; i < bins - 1; i++) {
    wB += hist[i];
    sumB += i * hist[i];
    const wF = total - wB;
    if (wB === 0 || wF === 0) continue;
    const mB = sumB / wB;
    const mF = (sumAll - sumB) / wF;
    const between = (wB / total) * (wF / total) * (mB - mF) ** 2;
    if (between > best) { best = between; bestI = i; }
  }
  return { thr: lo + ((bestI + 1) / bins) * (hi - lo), sep: varAll > 0 ? best / varAll : 0 };
}

/** Separable Gaussian blur of the RGB channels (alpha kept). */
export function gaussianBlur(img: RgbaImage, sigma: number): RgbaImage {
  if (sigma <= 0.05) return img;
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k: number[] = [];
  let sum = 0;
  for (let i = -r; i <= r; i++) { const v = Math.exp(-(i * i) / (2 * sigma * sigma)); k.push(v); sum += v; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const { width: w, height: h, data } = img;
  const tmp = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let a = 0, b = 0, c = 0;
    for (let i = -r; i <= r; i++) {
      const xx = Math.min(w - 1, Math.max(0, x + i));
      const o = (y * w + xx) * 4, kv = k[i + r];
      a += data[o] * kv; b += data[o + 1] * kv; c += data[o + 2] * kv;
    }
    const t = (y * w + x) * 3;
    tmp[t] = a; tmp[t + 1] = b; tmp[t + 2] = c;
  }
  const out = makeImage(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let a = 0, b = 0, c = 0;
    for (let i = -r; i <= r; i++) {
      const yy = Math.min(h - 1, Math.max(0, y + i));
      const t = (yy * w + x) * 3, kv = k[i + r];
      a += tmp[t] * kv; b += tmp[t + 1] * kv; c += tmp[t + 2] * kv;
    }
    const o = (y * w + x) * 4;
    out.data[o] = a; out.data[o + 1] = b; out.data[o + 2] = c; out.data[o + 3] = 255;
  }
  return out;
}
