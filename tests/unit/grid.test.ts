import { describe, expect, it } from 'vitest';
import { fitGrid } from '../../src/pipeline/grid';
import { orderCorners } from '../../src/pipeline/image';
import type { Quad } from '../../src/pipeline/types';
import { demoLayout, renderMosaic } from './synth';

describe('orderCorners', () => {
  it('orders TL,TR,BR,BL', () => {
    expect(orderCorners([[10, 10], [0, 10], [10, 0], [0, 0]])).toEqual([[0, 0], [10, 0], [10, 10], [0, 10]]);
  });
});

describe('fitGrid on a synthetic stud grid', () => {
  for (const [cols, rows, rot] of [[16, 16, 0], [12, 12, 3], [15, 16, -2]] as const) {
    it(`finds ${cols}x${rows} (rotated ${rot}°) from a loose quad`, () => {
      const { img, corners } = renderMosaic({ cols, rows, pitch: 48, pieces: demoLayout(cols, rows), rotateDeg: rot, seed: cols });
      // perturb the quad by a few px, as a detector would
      const quad = corners.map(([x, y], i) => [x + (i % 2 ? 4 : -3), y + (i > 1 ? 3 : -4)]) as Quad;
      const fit = fitGrid(img, quad);
      expect(fit.ok).toBe(true);
      expect(fit.grid.cols).toBe(cols);
      expect(fit.grid.rows).toBe(rows);
      expect(fit.grid.confidence).toBeGreaterThan(0.5);
      // fitted corners within ~15% pitch of the truth
      fit.grid.corners.forEach(([x, y], i) => {
        expect(Math.hypot(x - corners[i][0], y - corners[i][1])).toBeLessThan(48 * 0.15);
      });
      expect(fit.warped.width).toBe(cols * fit.grid.pitch);
      expect(fit.warped.height).toBe(rows * fit.grid.pitch);
    });
  }

  it('counts from stud evidence, not the quad: a 15% shrunken quad still gives 16x16', () => {
    const { img, corners } = renderMosaic({ cols: 16, rows: 16, pitch: 40, pieces: demoLayout(16, 16), seed: 5 });
    const cx = img.width / 2, cy = img.height / 2;
    const quad = corners.map(([x, y]) => [cx + (x - cx) * 0.85, cy + (y - cy) * 0.85]) as Quad;
    const fit = fitGrid(img, quad);
    expect(fit.ok).toBe(true);
    expect([fit.grid.cols, fit.grid.rows]).toEqual([16, 16]);
  });

  it('close-up framing: mosaic cut by the photo edge keeps only rows whose studs are in the photo', () => {
    const P = 40;
    const { img, corners } = renderMosaic({ cols: 16, rows: 16, pitch: P, pieces: demoLayout(16, 16), margin: 80, seed: 9 });
    // crop the photo through the middle of row 0 and row 15 (more than half of each hidden)
    const top = Math.round(corners[0][1] + P * 0.7), bottom = Math.round(corners[3][1] - P * 0.7);
    const h = bottom - top;
    const data = img.data.slice(top * img.width * 4, bottom * img.width * 4);
    const cropped = { width: img.width, height: h, data };
    const quad: Quad = [[corners[0][0], 0], [corners[1][0], 0], [corners[2][0], h - 1], [corners[3][0], h - 1]];
    const fit = fitGrid(cropped, quad);
    expect(fit.grid.cols).toBe(16);
    expect(fit.grid.rows).toBe(14);
  });

  it('honours gridOverride', () => {
    const { img, corners } = renderMosaic({ cols: 10, rows: 10, pitch: 40, pieces: demoLayout(10, 10) });
    const fit = fitGrid(img, corners as Quad, { gridOverride: { cols: 10, rows: 10 } });
    expect(fit.grid.cols).toBe(10);
    expect(fit.grid.rows).toBe(10);
  });

  it('reports low confidence on a blank image', () => {
    const { img } = renderMosaic({ cols: 16, rows: 16, pitch: 30, pieces: [], noise: 2 });
    // pieces: [] -> all cells grey "unknown" without seams; studs remain -> make truly blank instead
    img.data.fill(200);
    const fit = fitGrid(img, [[60, 60], [420, 60], [420, 420], [60, 420]]);
    expect(fit.ok).toBe(false);
  });
});
