import type { Point, Quad } from '../pipeline/types';

/** Row-major 3x3 homography. */
export type Mat3 = [number, number, number, number, number, number, number, number, number];

/**
 * Homography mapping the unit square (0,0),(1,0),(1,1),(0,1) onto quad TL,TR,BR,BL
 * (Heckbert's closed form).
 */
export function squareToQuad(q: Quad): Mat3 {
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = q;
  const dx1 = x1 - x2, dx2 = x3 - x2, dy1 = y1 - y2, dy2 = y3 - y2;
  const sx = x0 - x1 + x2 - x3, sy = y0 - y1 + y2 - y3;
  if (Math.abs(sx) < 1e-9 && Math.abs(sy) < 1e-9) {
    return [x1 - x0, x3 - x0, x0, y1 - y0, y3 - y0, y0, 0, 0, 1];
  }
  const den = dx1 * dy2 - dx2 * dy1;
  const g = (sx * dy2 - dx2 * sy) / den;
  const h = (dx1 * sy - sx * dy1) / den;
  return [x1 - x0 + g * x1, x3 - x0 + h * x3, x0, y1 - y0 + g * y1, y3 - y0 + h * y3, y0, g, h, 1];
}

export function apply(m: Mat3, u: number, v: number): Point {
  const w = m[6] * u + m[7] * v + m[8];
  return [(m[0] * u + m[1] * v + m[2]) / w, (m[3] * u + m[4] * v + m[5]) / w];
}

export function invert(m: Mat3): Mat3 {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    A / det, -(b * i - c * h) / det, (b * f - c * e) / det,
    B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
    C / det, -(a * h - b * g) / det, (a * e - b * d) / det,
  ];
}

/** Maps stud-grid coordinates (col,row units, 0..cols × 0..rows) to image pixels. */
export function gridMapper(corners: Quad, cols: number, rows: number) {
  const m = squareToQuad(corners);
  return (x: number, y: number): Point => apply(m, x / cols, y / rows);
}

export function polygonPoints(pts: Point[]): string {
  return pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
}

export function defaultCorners(w: number, h: number): Quad {
  const s = Math.min(w, h) * 0.8;
  const x = (w - s) / 2, y = (h - s) / 2;
  return [[x, y], [x + s, y], [x + s, y + s], [x, y + s]];
}
