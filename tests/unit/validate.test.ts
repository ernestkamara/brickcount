import { describe, expect, it } from 'vitest';
import { UnionFind } from '../../src/pipeline/unionFind';
import { groupPieces, isValidPlate, sizeLabel, splitComponent, type PairConf } from '../../src/pipeline/validate';
import type { PairDecision, StudCell } from '../../src/pipeline/types';

const cellsOf = (cols: number, rows: number, color = 'red'): StudCell[] =>
  Array.from({ length: cols * rows }, (_, i) => ({
    col: i % cols, row: Math.floor(i / cols), rgb: { r: 0, g: 0, b: 0 }, lab: [0, 0, 0], colorId: color, glare: 0,
  }));

/** Pair decisions from a label-like map of piece ids (same id = no seam). */
function pairsFromIds(ids: number[][], conf = 0.9): PairDecision[] {
  const rows = ids.length, cols = ids[0].length;
  const out: PairDecision[] = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    if (c + 1 < cols) out.push({ col: c, row: r, dir: 'h', seam: ids[r][c] !== ids[r][c + 1], confidence: conf });
    if (r + 1 < rows) out.push({ col: c, row: r, dir: 'v', seam: ids[r][c] !== ids[r + 1][c], confidence: conf });
  }
  return out;
}

describe('UnionFind', () => {
  it('unions and groups', () => {
    const uf = new UnionFind(6);
    expect(uf.union(0, 1)).toBe(true);
    expect(uf.union(1, 2)).toBe(true);
    expect(uf.union(0, 2)).toBe(false);
    uf.union(4, 5);
    expect(uf.find(2)).toBe(uf.find(0));
    expect(uf.find(3)).not.toBe(uf.find(0));
    expect(uf.groups().map((g) => g.length).sort()).toEqual([1, 2, 3]);
  });
});

describe('plate sizes', () => {
  it('knows real plates in both orientations', () => {
    expect(isValidPlate(4, 2)).toBe(true);
    expect(isValidPlate(2, 4)).toBe(true);
    expect(isValidPlate(1, 5)).toBe(false);
    expect(isValidPlate(3, 3)).toBe(true);
    expect(isValidPlate(5, 5)).toBe(false);
    expect(sizeLabel(4, 1)).toBe('1x4');
  });
});

describe('groupPieces / validation', () => {
  it('keeps a 2x4 as one piece', () => {
    const ids = [[0, 0, 0, 0], [0, 0, 0, 0]];
    const g = groupPieces(cellsOf(4, 2), { cols: 4, rows: 2 }, pairsFromIds(ids));
    expect(g.pieceCount).toBe(1);
    expect(g.pieces[0]).toMatchObject({ x: 0, y: 0, w: 4, h: 2, size: '2x4' });
    expect(g.pieces[0].split).toBeUndefined();
  });

  it('splits a 1x5 into two valid plates', () => {
    const ids = [[0, 0, 0, 0, 0]];
    const g = groupPieces(cellsOf(5, 1), { cols: 5, rows: 1 }, pairsFromIds(ids));
    expect(g.pieceCount).toBe(2);
    expect(g.pieces.every((p) => p.split && isValidPlate(p.w, p.h))).toBe(true);
    expect(g.studCount).toBe(5);
    expect(g.pieces.every((p) => p.confidence <= 0.5)).toBe(true);
  });

  it('splits a 1x5 at the least confident internal pair', () => {
    const pairs = pairsFromIds([[0, 0, 0, 0, 0]]);
    pairs.find((p) => p.col === 0)!.confidence = 0.2; // weak "same piece" between 0 and 1
    const g = groupPieces(cellsOf(5, 1), { cols: 5, rows: 1 }, pairs);
    expect(g.pieces.map((p) => p.size).sort()).toEqual(['1x1', '1x4']);
    expect(g.pieces.find((p) => p.size === '1x1')!.x).toBe(0);
  });

  it('splits an L-shape into two rectangles', () => {
    // X X X
    // X . .
    const ids = [[0, 0, 0], [0, 1, 2]];
    const g = groupPieces(cellsOf(3, 2), { cols: 3, rows: 2 }, pairsFromIds(ids));
    const L = g.pieces.filter((p) => p.split);
    expect(L.length).toBe(2);
    expect(L.reduce((s, p) => s + p.w * p.h, 0)).toBe(4);
    expect(g.pieceCount).toBe(4);
  });

  it('splits a 5x5 block into valid plates covering every stud', () => {
    const ids = Array.from({ length: 5 }, () => [0, 0, 0, 0, 0]);
    const g = groupPieces(cellsOf(5, 5), { cols: 5, rows: 5 }, pairsFromIds(ids));
    expect(g.studCount).toBe(25);
    expect(g.pieces.every((p) => isValidPlate(p.w, p.h))).toBe(true);
    expect(g.pieceCount).toBeLessThanOrEqual(4);
  });

  it('respects seams between separate pieces', () => {
    const ids = [[0, 0, 1, 1], [2, 2, 2, 2]];
    const g = groupPieces(cellsOf(4, 2), { cols: 4, rows: 2 }, pairsFromIds(ids));
    expect(g.pieces.map((p) => p.size).sort()).toEqual(['1x2', '1x2', '1x4']);
    expect(g.lowConfidenceCount).toBe(0);
  });

  it('splitComponent handles a large component greedily', () => {
    const cells: [number, number][] = [];
    for (let y = 0; y < 10; y++) for (let x = 0; x < 10; x++) cells.push([x, y]);
    const rects = splitComponent(cells, new Map() as PairConf);
    expect(rects.reduce((s, r) => s + r.w * r.h, 0)).toBe(100);
    expect(rects.every((r) => isValidPlate(r.w, r.h))).toBe(true);
  });
});
