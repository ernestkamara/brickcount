// Turn pair decisions into pieces: union studs across non-seam pairs, then make every
// component a real LEGO plate rectangle, splitting impossible shapes (L-shapes, 1x5, 5x5…).
import { LEGO_PLATE_SIZES } from '../lib/palette';
import type { GridSize, PairDecision, Piece, PieceRect, StudCell } from './types';
import { UnionFind } from './unionFind';

const VALID = new Set(LEGO_PLATE_SIZES.flatMap(([a, b]) => [`${a}x${b}`, `${b}x${a}`]));

export function isValidPlate(w: number, h: number): boolean {
  return VALID.has(`${w}x${h}`);
}

export function sizeLabel(w: number, h: number): string {
  return `${Math.min(w, h)}x${Math.max(w, h)}`;
}

/** Candidate (w,h) shapes, largest area first. */
const SHAPES: [number, number][] = [...VALID]
  .map((s) => s.split('x').map(Number) as [number, number])
  .sort((a, b) => b[0] * b[1] - a[0] * a[1] || b[0] - a[0]);

export interface Rect { x: number; y: number; w: number; h: number }

/** Cost/confidence lookup for internal (non-seam) pairs, keyed "c,r,dir". */
export type PairConf = Map<string, number>;
const key = (c: number, r: number, d: 'h' | 'v') => `${c},${r},${d}`;

/**
 * Partition a set of grid cells (one connected component) into valid plate rectangles,
 * minimising the summed confidence of the "same piece" decisions that must be cut, plus a
 * small per-piece penalty (prefer fewer, larger pieces). Exact branch & bound for small
 * components, greedy beyond a node budget.
 */
export function splitComponent(cells: [number, number][], conf: PairConf, nodeBudget = 4000): Rect[] {
  if (cells.length === 0) return [];
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of cells) { minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); }
  const W = maxX - minX + 1, H = maxY - minY + 1;
  const inComp = new Uint8Array(W * H);
  for (const [x, y] of cells) inComp[(y - minY) * W + (x - minX)] = 1;
  // already a valid rectangle?
  if (cells.length === W * H && isValidPlate(W, H)) return [{ x: minX, y: minY, w: W, h: H }];

  const PIECE_PENALTY = 0.35;
  const owner = new Int32Array(W * H).fill(-1);
  const order: number[] = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (inComp[y * W + x]) order.push(y * W + x);
  const cutCost = (ax: number, ay: number, dir: 'h' | 'v') => conf.get(key(ax + minX, ay + minY, dir)) ?? 0.5;

  let best: Rect[] | null = null;
  let bestCost = Infinity;
  let nodes = 0;
  const cur: Rect[] = [];

  const fits = (x: number, y: number, w: number, h: number) => {
    if (x + w > W || y + h > H) return false;
    for (let j = y; j < y + h; j++) for (let i = x; i < x + w; i++) {
      const k = j * W + i;
      if (!inComp[k] || owner[k] >= 0) return false;
    }
    return true;
  };
  // cost of cuts between rect and already-placed neighbours (each cut counted once)
  const placeCost = (x: number, y: number, w: number, h: number) => {
    let c = 0;
    for (let j = y; j < y + h; j++) {
      if (x > 0 && inComp[j * W + x - 1] && owner[j * W + x - 1] >= 0) c += cutCost(x - 1, j, 'h');
      if (x + w < W && inComp[j * W + x + w] && owner[j * W + x + w] >= 0) c += cutCost(x + w - 1, j, 'h');
    }
    for (let i = x; i < x + w; i++) {
      if (y > 0 && inComp[(y - 1) * W + i] && owner[(y - 1) * W + i] >= 0) c += cutCost(i, y - 1, 'v');
      if (y + h < H && inComp[(y + h) * W + i] && owner[(y + h) * W + i] >= 0) c += cutCost(i, y + h - 1, 'v');
    }
    return c;
  };
  const setOwner = (x: number, y: number, w: number, h: number, v: number) => {
    for (let j = y; j < y + h; j++) for (let i = x; i < x + w; i++) owner[j * W + i] = v;
  };

  const greedy = nodeBudget <= 0 || cells.length > 64;
  const dfs = (idx: number, cost: number): void => {
    if (cost >= bestCost) return;
    while (idx < order.length && owner[order[idx]] >= 0) idx++;
    if (idx >= order.length) {
      bestCost = cost;
      best = cur.map((r) => ({ ...r }));
      return;
    }
    if (++nodes > nodeBudget && best) return;
    const k = order[idx];
    const x = k % W, y = (k / W) | 0;
    for (const [w, h] of SHAPES) {
      if (!fits(x, y, w, h)) continue;
      const c = placeCost(x, y, w, h) + PIECE_PENALTY;
      setOwner(x, y, w, h, cur.length);
      cur.push({ x, y, w, h });
      dfs(idx + 1, cost + c);
      cur.pop();
      setOwner(x, y, w, h, -1);
      if (greedy || (nodes > nodeBudget && best)) break;
    }
  };
  dfs(0, 0);
  const res: Rect[] = best ?? [];
  return res.map((r) => ({ x: r.x + minX, y: r.y + minY, w: r.w, h: r.h }));
}

export interface GroupResult {
  pieces: Piece[];
  pieceCount: number;
  studCount: number;
  lowConfidenceCount: number;
}

export const LOW_CONFIDENCE = 0.6;

/** Union studs across non-seam pairs and validate/split into Piece[]. */
export function groupPieces(cells: StudCell[], grid: GridSize, pairs: PairDecision[]): GroupResult {
  const { cols, rows } = grid;
  const n = cols * rows;
  const uf = new UnionFind(n);
  const conf: PairConf = new Map();
  const decision = new Map<string, PairDecision>();
  for (const p of pairs) {
    decision.set(key(p.col, p.row, p.dir), p);
    if (!p.seam) {
      const b = p.dir === 'h' ? p.row * cols + p.col + 1 : (p.row + 1) * cols + p.col;
      uf.union(p.row * cols + p.col, b);
      conf.set(key(p.col, p.row, p.dir), p.manual ? 10 : p.confidence);
    }
  }
  const colourAt = (c: number, r: number) => cells[r * cols + c]?.colorId ?? 'unknown';
  const pairConf = (c: number, r: number, d: 'h' | 'v') => decision.get(key(c, r, d))?.confidence ?? 1;

  const pieces: Piece[] = [];
  const groups = uf.groups().sort((a, b) => a[0] - b[0]);
  for (const g of groups) {
    const coords = g.map((i) => [i % cols, (i / cols) | 0] as [number, number]);
    const rects = splitComponent(coords, conf);
    const split = rects.length > 1;
    for (const r of rects) {
      // colour: majority over cells
      const counts = new Map<string, number>();
      for (let y = r.y; y < r.y + r.h; y++) for (let x = r.x; x < r.x + r.w; x++) {
        const c = colourAt(x, y);
        counts.set(c, (counts.get(c) ?? 0) + 1);
      }
      const color = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
      // confidence: min over internal pairs and boundary pairs
      let cMin = 1;
      for (let y = r.y; y < r.y + r.h; y++) for (let x = r.x; x < r.x + r.w; x++) {
        if (x + 1 < cols) cMin = Math.min(cMin, pairConf(x, y, 'h'));
        if (y + 1 < rows) cMin = Math.min(cMin, pairConf(x, y, 'v'));
        if (x === r.x && x > 0) cMin = Math.min(cMin, pairConf(x - 1, y, 'h'));
        if (y === r.y && y > 0) cMin = Math.min(cMin, pairConf(x, y - 1, 'v'));
      }
      if (split) cMin = Math.min(cMin, 0.5);
      if (counts.size > 1) cMin = Math.min(cMin, 0.4);
      const piece: Piece = { id: 0, x: r.x, y: r.y, w: r.w, h: r.h, color, size: sizeLabel(r.w, r.h), confidence: cMin };
      if (split) piece.split = true;
      pieces.push(piece);
    }
  }
  pieces.sort((a, b) => a.y - b.y || a.x - b.x);
  pieces.forEach((p, i) => (p.id = i));
  return {
    pieces,
    pieceCount: pieces.length,
    studCount: pieces.reduce((s, p) => s + p.w * p.h, 0),
    lowConfidenceCount: pieces.filter((p) => p.confidence < LOW_CONFIDENCE).length,
  };
}

export function breakdown(pieces: PieceRect[]): { byColor: Record<string, number>; bySize: Record<string, number>; byColorSize: Record<string, number> } {
  const byColor: Record<string, number> = {}, bySize: Record<string, number> = {}, byColorSize: Record<string, number> = {};
  for (const p of pieces) {
    const s = sizeLabel(p.w, p.h);
    byColor[p.color] = (byColor[p.color] ?? 0) + 1;
    bySize[s] = (bySize[s] ?? 0) + 1;
    byColorSize[`${p.color}|${s}`] = (byColorSize[`${p.color}|${s}`] ?? 0) + 1;
  }
  return { byColor, bySize, byColorSize };
}
