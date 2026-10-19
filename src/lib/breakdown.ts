import type { Breakdown, Piece } from '../pipeline/types';

/** Breakdown recomputed from pieces (regroup returns pieces only). */
export function computeBreakdown(pieces: readonly Piece[]): Breakdown {
  const byColor: Record<string, number> = {};
  const bySize: Record<string, number> = {};
  const byColorSize: Record<string, number> = {};
  for (const p of pieces) {
    byColor[p.color] = (byColor[p.color] ?? 0) + 1;
    bySize[p.size] = (bySize[p.size] ?? 0) + 1;
    const k = `${p.color}|${p.size}`;
    byColorSize[k] = (byColorSize[k] ?? 0) + 1;
  }
  return { byColor, bySize, byColorSize };
}

export function mergeCounts(into: Record<string, number>, from: Record<string, number>): Record<string, number> {
  for (const [k, v] of Object.entries(from)) into[k] = (into[k] ?? 0) + v;
  return into;
}

/** Size label short side first, e.g. 4x1 → "1x4". */
export function sizeLabel(w: number, h: number): string {
  return `${Math.min(w, h)}x${Math.max(w, h)}`;
}

/** Sort size labels by area then short side ("1x1", "1x2", "2x2", ...). */
export function compareSize(a: string, b: string): number {
  const [a1, a2] = a.split('x').map(Number);
  const [b1, b2] = b.split('x').map(Number);
  return a1 * a2 - b1 * b2 || a1 - b1 || a.localeCompare(b);
}
