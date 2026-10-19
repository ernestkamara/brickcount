// Mock WorkerApi for UI development and e2e tests. Runs on the main thread and
// produces fake but plausible results: a square grid fitted inside the image,
// cell colours sampled from the photo and snapped to the palette, cells greedily
// tiled into real plate sizes, seams derived from the tiling.
import type { GridSize, PairDecision, Piece, Quad, ScanOptions, ScanResult, StudCell } from '../pipeline/types';
import type { WorkerApi } from './api';
import { LEGO_PLATE_SIZES, PALETTE } from '../lib/palette';
import { computeBreakdown, sizeLabel } from '../lib/breakdown';

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Deterministic hash → [0,1). */
function rand(a: number, b: number, c = 0): number {
  let h = (a * 374761393 + b * 668265263 + c * 2147483647) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function nearestColor(r: number, g: number, b: number): string {
  let best = PALETTE[0].id;
  let bd = Infinity;
  for (const c of PALETTE) {
    const d = (c.rgb[0] - r) ** 2 + (c.rgb[1] - g) ** 2 + (c.rgb[2] - b) ** 2;
    if (d < bd) { bd = d; best = c.id; }
  }
  return best;
}

function lerp(a: [number, number], b: [number, number], t: number): [number, number] {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
}

/** Bilinear point inside quad (good enough for a mock). */
function quadPoint(q: Quad, u: number, v: number): [number, number] {
  return lerp(lerp(q[0], q[1], u), lerp(q[3], q[2], u), v);
}

function sampleCells(image: ImageBitmap, corners: Quad, size: GridSize): StudCell[] {
  let data: Uint8ClampedArray | null = null;
  const { width: W, height: H } = image;
  try {
    const c = new OffscreenCanvas(W, H);
    const ctx = c.getContext('2d')!;
    ctx.drawImage(image, 0, 0);
    data = ctx.getImageData(0, 0, W, H).data;
  } catch {
    data = null;
  }
  const cells: StudCell[] = [];
  for (let row = 0; row < size.rows; row++) {
    for (let col = 0; col < size.cols; col++) {
      let rgb = { r: 128, g: 128, b: 128 };
      if (data) {
        // sample a point between the stud centre and the cell corner (avoids stud rim shading)
        const [x, y] = quadPoint(corners, (col + 0.2) / size.cols, (row + 0.2) / size.rows);
        const xi = Math.min(W - 1, Math.max(0, Math.round(x)));
        const yi = Math.min(H - 1, Math.max(0, Math.round(y)));
        const o = (yi * W + xi) * 4;
        rgb = { r: data[o], g: data[o + 1], b: data[o + 2] };
      }
      cells.push({ col, row, rgb, lab: [50, 0, 0], colorId: nearestColor(rgb.r, rgb.g, rgb.b), glare: 0 });
    }
  }
  return cells;
}

/** Greedy tiling of same-colour regions into plate sizes → seam decisions. */
function tile(cells: StudCell[], size: GridSize): PairDecision[] {
  const { cols, rows } = size;
  const owner = new Int32Array(cols * rows).fill(-1);
  const color = (c: number, r: number) => cells[r * cols + c].colorId;
  const shapes = LEGO_PLATE_SIZES.flatMap(([a, b]) => (a === b ? [[a, b]] : [[a, b], [b, a]]));
  let id = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (owner[r * cols + c] >= 0) continue;
      const fits = shapes.filter(([w, h]) => {
        if (c + w > cols || r + h > rows || w * h > 8) return false;
        for (let y = r; y < r + h; y++) for (let x = c; x < c + w; x++) {
          if (owner[y * cols + x] >= 0 || color(x, y) !== color(c, r)) return false;
        }
        return true;
      });
      const [w, h] = fits[Math.floor(rand(c, r, 7) * fits.length)] ?? [1, 1];
      for (let y = r; y < r + h; y++) for (let x = c; x < c + w; x++) owner[y * cols + x] = id;
      id++;
    }
  }
  const pairs: PairDecision[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const conf = (dir: number) => {
        const v = rand(c, r, dir);
        return v < 0.015 ? 0.35 + v * 10 : 0.75 + v * 0.25;
      };
      if (c + 1 < cols) pairs.push({ col: c, row: r, dir: 'h', seam: owner[r * cols + c] !== owner[r * cols + c + 1], confidence: conf(1) });
      if (r + 1 < rows) pairs.push({ col: c, row: r, dir: 'v', seam: owner[r * cols + c] !== owner[(r + 1) * cols + c], confidence: conf(2) });
    }
  }
  return pairs;
}

/** Union-find over non-seam pairs; non-rectangular components are split into row runs. */
function group(cells: StudCell[], grid: GridSize, pairs: PairDecision[]) {
  const { cols, rows } = grid;
  const n = cols * rows;
  const parent = Int32Array.from({ length: n }, (_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  const idx = (c: number, r: number) => r * cols + c;
  for (const p of pairs) {
    if (p.seam) continue;
    const a = idx(p.col, p.row);
    const b = p.dir === 'h' ? idx(p.col + 1, p.row) : idx(p.col, p.row + 1);
    parent[find(a)] = find(b);
  }
  const comps = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const r = find(i);
    const list = comps.get(r);
    if (list) list.push(i); else comps.set(r, [i]);
  }
  const rects: { x: number; y: number; w: number; h: number; split?: boolean }[] = [];
  for (const members of comps.values()) {
    const xs = members.map((i) => i % cols), ys = members.map((i) => Math.floor(i / cols));
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const w = x1 - x0 + 1, h = y1 - y0 + 1;
    const isPlate = LEGO_PLATE_SIZES.some(([a, b]) => (a === w && b === h) || (a === h && b === w));
    if (w * h === members.length && isPlate) { rects.push({ x: x0, y: y0, w, h }); continue; }
    const set = new Set(members);
    for (let y = y0; y <= y1; y++) {
      let x = x0;
      while (x <= x1) {
        if (!set.has(idx(x, y))) { x++; continue; }
        let e = x;
        while (e + 1 <= x1 && set.has(idx(e + 1, y)) && e + 1 - x < 8) e++;
        rects.push({ x, y, w: e - x + 1, h: 1, split: true });
        x = e + 1;
      }
    }
  }
  rects.sort((a, b) => a.y - b.y || a.x - b.x);
  const conf = new Map<string, number>(pairs.map((p) => [`${p.col},${p.row},${p.dir}`, p.confidence]));
  const pieces: Piece[] = rects.map((r, i) => {
    let c = 1;
    for (let y = r.y; y < r.y + r.h; y++) for (let x = r.x; x < r.x + r.w; x++) {
      c = Math.min(c, conf.get(`${x},${y},h`) ?? 1, conf.get(`${x},${y},v`) ?? 1);
      if (x === r.x && x > 0) c = Math.min(c, conf.get(`${x - 1},${y},h`) ?? 1);
      if (y === r.y && y > 0) c = Math.min(c, conf.get(`${x},${y - 1},v`) ?? 1);
    }
    return { id: i + 1, x: r.x, y: r.y, w: r.w, h: r.h, color: cells[idx(r.x, r.y)].colorId, size: sizeLabel(r.w, r.h), confidence: c, ...(r.split ? { split: true } : {}) };
  });
  return {
    pieces,
    pieceCount: pieces.length,
    studCount: n,
    lowConfidenceCount: pieces.filter((p) => p.confidence < 0.6).length,
  };
}

export function createMockWorker(): WorkerApi {
  return {
    async init() {
      await sleep(150);
    },
    async scan(image: ImageBitmap, opts: ScanOptions, onProgress?: (stage: string, frac: number) => void): Promise<ScanResult> {
      const t0 = performance.now();
      const { width: W, height: H } = image;
      const stages = ['decode', 'grid', 'rectify', 'colors', 'seams', 'pieces'];
      const step = async (i: number) => { onProgress?.(stages[i], i / stages.length); await sleep(60); };
      await step(0);
      await step(1);
      const aspect = W / H;
      if (!opts.corners && (aspect > 2.2 || aspect < 1 / 2.2)) {
        image.close();
        return { status: 'grid_failed', message: 'No square grid found (mock: extreme aspect ratio)', pieceCount: 0, studCount: 0, lowConfidenceCount: 0, timingsMs: { total: performance.now() - t0 }, classifier: opts.classifier };
      }
      const side = Math.min(W, H) * 0.94;
      const x0 = (W - side) / 2, y0 = (H - side) / 2;
      const corners: Quad = opts.corners ?? [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]];
      const size: GridSize = opts.gridOverride ?? { cols: 16, rows: 16 };
      await step(2);
      const cells = sampleCells(image, corners, size);
      image.close();
      await step(3);
      const pairs = tile(cells, size);
      await step(4);
      const g = group(cells, size, pairs);
      await step(5);
      onProgress?.('done', 1);
      const warpedSize = size.cols * 32;
      return {
        status: 'ok',
        grid: { ...size, corners, warpedSize, pitch: warpedSize / size.cols, confidence: 0.93 },
        cells,
        pairs,
        ...g,
        breakdown: computeBreakdown(g.pieces),
        timingsMs: { total: performance.now() - t0 },
        classifier: opts.classifier,
      };
    },
    async regroup(cells, grid, pairs) {
      await sleep(20);
      return group(cells, grid, pairs);
    },
  };
}
