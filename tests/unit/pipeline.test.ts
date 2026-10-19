import { beforeAll, describe, expect, it } from 'vitest';
import type { CV } from '../../src/pipeline/cv';
import { loadOpenCvNode as loadOpenCv } from './cvNode';
import { detectQuad } from '../../src/pipeline/preprocess';
import { regroup, runPipeline } from '../../src/pipeline/run';
import { demoLayout, renderMosaic } from './synth';
import type { Quad } from '../../src/pipeline/types';

let cv: CV;
beforeAll(async () => {
  cv = await loadOpenCv();
});

describe('runPipeline (OpenCV.js) on a synthetic photo', () => {
  const pieces = demoLayout(16, 16);
  const { img, corners } = renderMosaic({ cols: 16, rows: 16, pitch: 40, pieces, rotateDeg: 2, margin: 140, seed: 3 });

  it('detects the mosaic quad', () => {
    const q = detectQuad(cv, img);
    q.quad.forEach(([x, y], i) => expect(Math.hypot(x - corners[i][0], y - corners[i][1])).toBeLessThan(12));
  });

  it('counts pieces end to end and regroups after a manual edit', async () => {
    const r = await runPipeline(cv, img, { classifier: 'heuristic', countBaseplate: true });
    expect(r.status).toBe('ok');
    expect(r.grid).toMatchObject({ cols: 16, rows: 16 });
    expect(r.studCount).toBe(256);
    const truth = new Set(pieces.map((p) => `${p.x},${p.y},${p.w},${p.h}`));
    const matched = r.pieces!.filter((p) => truth.has(`${p.x},${p.y},${p.w},${p.h}`)).length;
    expect(matched / pieces.length).toBeGreaterThan(0.9);
    expect(Object.values(r.breakdown!.bySize).reduce((a, b) => a + b, 0)).toBe(r.pieceCount);
    expect(r.timingsMs.total).toBeGreaterThan(0);

    // manual edit: force a seam inside the first multi-stud piece
    const p0 = r.pieces!.find((p) => p.w > 1)!;
    const pairs = r.pairs!.map((p) => (p.col === p0.x && p.row === p0.y && p.dir === 'h' ? { ...p, seam: true, manual: true } : p));
    const g = regroup(r.cells!, r.grid!, pairs);
    expect(g.pieceCount).toBe(r.pieceCount + 1);
  });

  it('re-scan with user corners slightly outside the mosaic succeeds', async () => {
    const cx = img.width / 2, cy = img.height / 2;
    const loose = corners.map(([x, y]) => [cx + (x - cx) * 1.05, cy + (y - cy) * 1.05]) as Quad;
    const r = await runPipeline(cv, img, { classifier: 'heuristic', countBaseplate: true, corners: loose });
    expect(r.status).toBe('ok');
    expect(r.grid).toMatchObject({ cols: 16, rows: 16 });
  });

  it('re-scan with user corners at the photo corners of a tightly cropped photo succeeds', async () => {
    const pad = 8;
    const x0 = Math.round(Math.min(...corners.map((c) => c[0]))) - pad, y0 = Math.round(Math.min(...corners.map((c) => c[1]))) - pad;
    const x1 = Math.round(Math.max(...corners.map((c) => c[0]))) + pad, y1 = Math.round(Math.max(...corners.map((c) => c[1]))) + pad;
    const w = x1 - x0, h = y1 - y0;
    const data = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) data.set(img.data.subarray(((y0 + y) * img.width + x0) * 4, ((y0 + y) * img.width + x0 + w) * 4), y * w * 4);
    const r = await runPipeline(cv, { width: w, height: h, data }, {
      classifier: 'heuristic', countBaseplate: true, corners: [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
    });
    expect(r.status).toBe('ok');
    expect(r.grid).toMatchObject({ cols: 16, rows: 16 });
  });

  it('returns grid_failed for a blank photo', async () => {
    const blank = { width: 600, height: 800, data: new Uint8ClampedArray(600 * 800 * 4).fill(230) };
    const r = await runPipeline(cv, blank, { classifier: 'heuristic', countBaseplate: true });
    expect(r.status).toBe('grid_failed');
  });
});
