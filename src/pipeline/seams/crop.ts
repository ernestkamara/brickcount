// Gap-crop extractor — the exact spec shared with ml/dataset.py (see CROP_W/CROP_H in ./index):
//
//   Pair (col,row,'h'): centre c = ((col+1)*Pw, (row+0.5)*Ph), ex = (1,0),  ey = (0,1)
//   Pair (col,row,'v'): centre c = ((col+0.5)*Pw, (row+1)*Ph), ex = (0,1),  ey = (-1,0)
//     (vertical pairs rotated 90°: the upper stud lands on the left half of the crop)
//   Crop pixel (i,j), i in 0..31, j in 0..15 samples the warped image at
//     c + ((i + 0.5 - 16) * ex * Pa + (j + 0.5 - 8) * ey * Pb) / 16
//   where Pa / Pb are the pitches along ex / ey (equal for square cells), bilinear,
//   pixel (k,l) centred at (k+0.5, l+0.5), clamp-to-edge. Output float32 CHW RGB / 255.
import type { StudPair } from '../types';
import { CROP_H, CROP_W } from './index';

export interface WarpedLike {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

/** Write one crop (3*CROP_H*CROP_W floats, CHW) into `out` at `offset`. */
export function extractGapCrop(
  img: WarpedLike,
  cols: number,
  rows: number,
  pair: StudPair,
  out: Float32Array,
  offset = 0,
): void {
  const Pw = img.width / cols, Ph = img.height / rows;
  let cx: number, cy: number, exx: number, exy: number, eyx: number, eyy: number, Pa: number, Pb: number;
  if (pair.dir === 'h') {
    cx = (pair.col + 1) * Pw; cy = (pair.row + 0.5) * Ph;
    exx = 1; exy = 0; eyx = 0; eyy = 1; Pa = Pw; Pb = Ph;
  } else {
    cx = (pair.col + 0.5) * Pw; cy = (pair.row + 1) * Ph;
    exx = 0; exy = 1; eyx = -1; eyy = 0; Pa = Ph; Pb = Pw;
  }
  const W = img.width, H = img.height, d = img.data;
  const plane = CROP_W * CROP_H;
  for (let j = 0; j < CROP_H; j++) {
    const b = ((j + 0.5 - CROP_H / 2) * Pb) / 16;
    for (let i = 0; i < CROP_W; i++) {
      const a = ((i + 0.5 - CROP_W / 2) * Pa) / 16;
      const X = cx + a * exx + b * eyx;
      const Y = cy + a * exy + b * eyy;
      const fx = X - 0.5, fy = Y - 0.5;
      const x0 = Math.floor(fx), y0 = Math.floor(fy);
      const ax = fx - x0, ay = fy - y0;
      const x0c = Math.min(W - 1, Math.max(0, x0)), x1c = Math.min(W - 1, Math.max(0, x0 + 1));
      const y0c = Math.min(H - 1, Math.max(0, y0)), y1c = Math.min(H - 1, Math.max(0, y0 + 1));
      const k00 = (y0c * W + x0c) * 4, k10 = (y0c * W + x1c) * 4, k01 = (y1c * W + x0c) * 4, k11 = (y1c * W + x1c) * 4;
      const o = offset + j * CROP_W + i;
      for (let c = 0; c < 3; c++) {
        const v = (d[k00 + c] * (1 - ax) + d[k10 + c] * ax) * (1 - ay) + (d[k01 + c] * (1 - ax) + d[k11 + c] * ax) * ay;
        out[o + c * plane] = v / 255;
      }
    }
  }
}

/** Batch of crops as one float32 [N,3,CROP_H,CROP_W] buffer. */
export function extractGapCrops(img: WarpedLike, cols: number, rows: number, pairs: StudPair[]): Float32Array {
  const per = 3 * CROP_H * CROP_W;
  const out = new Float32Array(pairs.length * per);
  pairs.forEach((p, n) => extractGapCrop(img, cols, rows, p, out, n * per));
  return out;
}
