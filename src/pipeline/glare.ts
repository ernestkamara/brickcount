// Glare (specular highlight through the glass) mask on the rectified image.
// Glare = bright, unsaturated pixels that are clearly brighter than their neighbourhood
// (a top-hat-like test via a large box mean, so white plates aren't flagged), plus fully
// clipped pixels. Cleaned with a 3x3 opening and a small dilation. Pure TS.
import type { RgbaImage } from './types';

export interface GlareOptions {
  /** Box radius of the local background, px. Default ~1.5 pitch. */
  radius: number;
}

function boxMean(v: Float32Array, w: number, h: number, r: number): Float32Array {
  const ii = new Float64Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y++) {
    let row = 0;
    for (let x = 0; x < w; x++) {
      row += v[y * w + x];
      ii[(y + 1) * (w + 1) + x + 1] = ii[y * (w + 1) + x + 1] + row;
    }
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
      const s = ii[y1 * (w + 1) + x1] - ii[y0 * (w + 1) + x1] - ii[y1 * (w + 1) + x0] + ii[y0 * (w + 1) + x0];
      out[y * w + x] = s / ((y1 - y0) * (x1 - x0));
    }
  }
  return out;
}

function morph(src: Uint8Array, w: number, h: number, dilate: boolean): Uint8Array<ArrayBuffer> {
  // separable 3x3
  const tmp = new Uint8Array(w * h), out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const a = src[i], l = x > 0 ? src[i - 1] : a, r = x < w - 1 ? src[i + 1] : a;
      tmp[i] = dilate ? (a | l | r) : (a & l & r);
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const a = tmp[i], u = y > 0 ? tmp[i - w] : a, d = y < h - 1 ? tmp[i + w] : a;
      out[i] = dilate ? (a | u | d) : (a & u & d);
    }
  }
  return out;
}

/** Returns a 0/255 mask the size of `img`. */
export function glareMask(img: RgbaImage, opts: Partial<GlareOptions> = {}): Uint8Array {
  const { width: w, height: h, data } = img;
  const n = w * h;
  const V = new Float32Array(n);
  const S = new Float32Array(n);
  for (let i = 0, k = 0; i < n; i++, k += 4) {
    const mx = Math.max(data[k], data[k + 1], data[k + 2]);
    const mn = Math.min(data[k], data[k + 1], data[k + 2]);
    V[i] = mx;
    S[i] = mx > 0 ? (mx - mn) / mx : 0;
  }
  const r = Math.max(4, Math.round(opts.radius ?? Math.min(w, h) / 12));
  const bg = boxMean(V, w, h, r);
  let m: Uint8Array<ArrayBuffer> = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const v = V[i], s = S[i];
    if ((v >= 200 && s < 0.2 && v - bg[i] > 30) || (v >= 250 && s < 0.08)) m[i] = 1;
  }
  m = morph(morph(m, w, h, false), w, h, true); // open
  m = morph(m, w, h, true); // dilate 1px
  for (let i = 0; i < n; i++) m[i] = m[i] ? 255 : 0;
  return m;
}

/** Fraction of glare pixels inside the axis-aligned box. */
export function glareFraction(mask: Uint8Array, w: number, x0: number, y0: number, x1: number, y1: number): number {
  let c = 0, t = 0;
  const h = mask.length / w;
  for (let y = Math.max(0, Math.floor(y0)); y < Math.min(h, Math.ceil(y1)); y++) {
    for (let x = Math.max(0, Math.floor(x0)); x < Math.min(w, Math.ceil(x1)); x++) {
      t++;
      if (mask[y * w + x]) c++;
    }
  }
  return t ? c / t : 0;
}
