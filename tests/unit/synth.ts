// Tiny pure-TS renderer for synthetic mosaic photos used by unit tests.
import { PALETTE_BY_ID } from '../../src/lib/palette';
import type { PieceRect, RgbaImage } from '../../src/pipeline/types';

export interface SynthOptions {
  cols: number;
  rows: number;
  pitch: number;
  pieces: PieceRect[];
  /** Mat border around the mosaic, px. */
  margin?: number;
  /** Rotation of the mosaic about the image centre, degrees. */
  rotateDeg?: number;
  seamWidth?: number;
  noise?: number;
  seed?: number;
}

function rng(seed: number) {
  let s = seed >>> 0 || 1;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}

/** Render pieces on a white mat. Returns the image and the mosaic corners (TL,TR,BR,BL). */
export function renderMosaic(o: SynthOptions): { img: RgbaImage; corners: [number, number][] } {
  const margin = o.margin ?? Math.round(o.pitch * 2);
  const W = o.cols * o.pitch + 2 * margin, H = o.rows * o.pitch + 2 * margin;
  const data = new Uint8ClampedArray(W * H * 4);
  const owner = new Int32Array(o.cols * o.rows).fill(-1);
  o.pieces.forEach((p, i) => {
    for (let y = p.y; y < p.y + p.h; y++) for (let x = p.x; x < p.x + p.w; x++) owner[y * o.cols + x] = i;
  });
  const th = ((o.rotateDeg ?? 0) * Math.PI) / 180, c = Math.cos(th), s = Math.sin(th);
  const cx = W / 2, cy = H / 2;
  const P = o.pitch, sw = o.seamWidth ?? Math.max(1, P / 30);
  const rnd = rng(o.seed ?? 1);
  const noise = o.noise ?? 4;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      // inverse-rotate into mosaic coords
      const dx = x + 0.5 - cx, dy = y + 0.5 - cy;
      const u = c * dx + s * dy + (o.cols * P) / 2, v = -s * dx + c * dy + (o.rows * P) / 2;
      let r = 236, g = 236, b = 238;
      const col = Math.floor(u / P), row = Math.floor(v / P);
      if (u >= 0 && v >= 0 && col < o.cols && row < o.rows) {
        const id = owner[row * o.cols + col];
        const rgb = id >= 0 ? PALETTE_BY_ID[o.pieces[id].color].rgb : [80, 80, 80];
        [r, g, b] = rgb;
        let f = 1;
        const fu = u - col * P, fv = v - row * P;
        // seams on boundaries between different pieces
        const nb = (cc: number, rr: number) => (cc < 0 || rr < 0 || cc >= o.cols || rr >= o.rows ? -2 : owner[rr * o.cols + cc]);
        if ((fu < sw && nb(col - 1, row) !== id) || (P - fu < sw && nb(col + 1, row) !== id) ||
            (fv < sw && nb(col, row - 1) !== id) || (P - fv < sw && nb(col, row + 1) !== id)) f = 0.45;
        // stud: shaded outline + slightly brighter top
        const rr = Math.hypot(fu - P / 2, fv - P / 2) / P;
        if (rr < 0.3 && rr > 0.26) {
          const lit = (P / 2 - fu + (P / 2 - fv)) / (P * 0.6); // top-left lit
          f *= 1 + 0.35 * lit;
        } else if (rr <= 0.26) f *= 1.06;
        r *= f; g *= f; b *= f;
      }
      const n = (rnd() - 0.5) * 2 * noise;
      const k = (y * W + x) * 4;
      data[k] = r + n; data[k + 1] = g + n; data[k + 2] = b + n; data[k + 3] = 255;
    }
  }
  const corner = (u: number, v: number): [number, number] => {
    const du = u - (o.cols * P) / 2, dv = v - (o.rows * P) / 2;
    return [c * du - s * dv + cx, s * du + c * dv + cy];
  };
  return {
    img: { width: W, height: H, data },
    corners: [corner(0, 0), corner(o.cols * P, 0), corner(o.cols * P, o.rows * P), corner(0, o.rows * P)],
  };
}

/** A deterministic layout of valid plates tiling cols x rows, alternating colours. */
export function demoLayout(cols: number, rows: number): PieceRect[] {
  const colours = ['red', 'medium_azure', 'yellow', 'blue', 'white', 'black', 'green', 'tan'];
  const out: PieceRect[] = [];
  const used = new Uint8Array(cols * rows);
  const shapes: [number, number][] = [[4, 1], [2, 2], [3, 1], [2, 1], [1, 1], [1, 2], [6, 1], [2, 3]];
  let k = 0;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      if (used[y * cols + x]) continue;
      for (let t = 0; t < shapes.length; t++) {
        const [w, h] = shapes[(k + t) % shapes.length];
        let ok = x + w <= cols && y + h <= rows;
        for (let j = y; ok && j < y + h; j++) for (let i = x; ok && i < x + w; i++) if (used[j * cols + i]) ok = false;
        if (!ok) continue;
        for (let j = y; j < y + h; j++) for (let i = x; i < x + w; i++) used[j * cols + i] = 1;
        // same colour as the previous piece every third time, so seams between same colours exist
        out.push({ x, y, w, h, color: colours[(k % 3 === 2 ? k - 1 : k) % colours.length] });
        k++;
        break;
      }
    }
  }
  return out;
}
