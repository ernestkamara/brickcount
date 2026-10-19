import { describe, expect, it } from 'vitest';
import { classifyCells, nearestPalette, rgbToLab } from '../../src/pipeline/colour';
import { glareMask } from '../../src/pipeline/glare';
import { gridFromCorners } from '../../src/pipeline/grid';
import { extractGapCrop } from '../../src/pipeline/seams/crop';
import { allPairs, HeuristicSeamClassifier } from '../../src/pipeline/seams/heuristic';
import { CnnSeamClassifier, type OrtLike, type OrtSessionLike } from '../../src/pipeline/seams/cnn';
import { CROP_H, CROP_W, type SeamContext } from '../../src/pipeline/seams/index';
import { PALETTE } from '../../src/lib/palette';
import type { PieceRect, Quad } from '../../src/pipeline/types';
import { groupPieces } from '../../src/pipeline/validate';
import { renderMosaic } from './synth';

function context(pieces: PieceRect[], cols: number, rows: number, pitch = 56): SeamContext {
  const { img, corners } = renderMosaic({ cols, rows, pitch, pieces, seed: 7 });
  const { grid, warped } = gridFromCorners(img, corners as Quad, { cols, rows }, pitch);
  const glare = glareMask(warped, { radius: pitch });
  const cells = classifyCells(warped, grid, glare);
  return { warped, grid, cells, glareMask: glare };
}

describe('colour', () => {
  it('maps palette colours to themselves', () => {
    for (const p of PALETTE) expect(nearestPalette(rgbToLab(...p.rgb)).id).toBe(p.id);
  });
});

describe('heuristic seams', () => {
  it('finds the seam between two same-colour 1x4 plates and nothing inside them', async () => {
    const pieces: PieceRect[] = [
      { x: 0, y: 0, w: 4, h: 1, color: 'red' },
      { x: 4, y: 0, w: 4, h: 1, color: 'red' },
      { x: 0, y: 1, w: 4, h: 1, color: 'blue' },
      { x: 4, y: 1, w: 4, h: 1, color: 'blue' },
    ];
    const ctx = context(pieces, 8, 2);
    expect(ctx.cells.filter((c) => c.row === 0).every((c) => c.colorId === 'red')).toBe(true);
    const pairs = await new HeuristicSeamClassifier().classify(ctx, allPairs(8, 2));
    const h = pairs.filter((p) => p.dir === 'h');
    for (const p of h) expect(p.seam).toBe(p.col === 3);
    expect(pairs.filter((p) => p.dir === 'v').every((p) => p.seam)).toBe(true);
    const g = groupPieces(ctx.cells, ctx.grid, pairs);
    expect(g.pieces.map((p) => `${p.x},${p.y},${p.size},${p.color}`)).toEqual(['0,0,1x4,red', '4,0,1x4,red', '0,1,1x4,blue', '4,1,1x4,blue']);
  });

  it('separates a 2x2 from neighbouring same-colour 1x1s', async () => {
    const pieces: PieceRect[] = [
      { x: 0, y: 0, w: 2, h: 2, color: 'yellow' },
      { x: 2, y: 0, w: 1, h: 1, color: 'yellow' },
      { x: 2, y: 1, w: 1, h: 1, color: 'yellow' },
      { x: 0, y: 2, w: 3, h: 1, color: 'yellow' },
    ];
    const ctx = context(pieces, 3, 3);
    const pairs = await new HeuristicSeamClassifier().classify(ctx, allPairs(3, 3));
    const g = groupPieces(ctx.cells, ctx.grid, pairs);
    expect(g.pieces.map((p) => p.size).sort()).toEqual(['1x1', '1x1', '1x3', '2x2']);
  });
});

describe('gap crops', () => {
  it('follows the shared spec: gap in the middle column, vertical pairs rotated (upper stud left)', () => {
    // 2x2 grid, pitch 16: left column dark, right column bright; top row red-ish, bottom green-ish
    const P = 16, W = 32, H = 32;
    const data = new Uint8ClampedArray(W * H * 4);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const k = (y * W + x) * 4;
      data[k] = y < P ? 200 : 20; data[k + 1] = y < P ? 20 : 200; data[k + 2] = x < P ? 0 : 255; data[k + 3] = 255;
    }
    const img = { width: W, height: H, data };
    const out = new Float32Array(3 * CROP_H * CROP_W);
    const plane = CROP_H * CROP_W;
    extractGapCrop(img, 2, 2, { col: 0, row: 0, dir: 'h' }, out);
    expect(out[2 * plane + 4 * CROP_W + 5]).toBeCloseTo(0, 2); // left half = left column (blue 0)
    expect(out[2 * plane + 4 * CROP_W + 26]).toBeCloseTo(1, 2); // right half = right column
    extractGapCrop(img, 2, 2, { col: 0, row: 0, dir: 'v' }, out);
    expect(out[8 * CROP_W + 5]).toBeCloseTo(200 / 255, 2); // left of crop = upper stud (red)
    expect(out[1 * plane + 8 * CROP_W + 26]).toBeCloseTo(200 / 255, 2); // right = lower stud (green)
  });
});

describe('CNN classifier', () => {
  const ctx = context([{ x: 0, y: 0, w: 2, h: 1, color: 'red' }, { x: 0, y: 1, w: 2, h: 1, color: 'red' }], 2, 2, 40);
  const ort: OrtLike = {
    Tensor: class { constructor(public type: string, public data: Float32Array, public dims: number[]) {} } as unknown as OrtLike['Tensor'],
    InferenceSession: { create: async () => { throw new Error('no model'); } },
  };

  it('batches crops through the injected session and thresholds sigmoid(logit)', async () => {
    let seen: number[] = [];
    const session: OrtSessionLike = {
      async run(feeds) {
        const t = feeds.input as { dims: number[] };
        seen = t.dims;
        return { logit: { data: Float32Array.from({ length: t.dims[0] }, (_, i) => (i % 2 ? 3 : -3)) } };
      },
    };
    const cnn = new CnnSeamClassifier({ session, ort });
    const pairs = allPairs(2, 2);
    const out = await cnn.classify(ctx, pairs);
    expect(seen).toEqual([pairs.length, 3, CROP_H, CROP_W]);
    expect(out.map((p) => p.seam)).toEqual(pairs.map((_, i) => i % 2 === 1));
    expect(cnn.fallbackReason).toBeNull();
  });

  it('falls back to the heuristic when the model cannot load', async () => {
    const cnn = new CnnSeamClassifier({ ort });
    const out = await cnn.classify(ctx, allPairs(2, 2));
    expect(out.length).toBe(4);
    expect(cnn.fallbackReason).toMatch(/no model/);
  });
});
