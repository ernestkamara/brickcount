/// <reference lib="webworker" />
// Pipeline worker: loads OpenCV.js once, converts the transferred ImageBitmap to RGBA and
// runs the pipeline. Exposed with Comlink as WorkerApi (src/worker/api.ts).
import * as Comlink from 'comlink';
// Static default import on purpose: the bundler wraps a dynamic import() of this CJS module
// in an object inheriting Promise.prototype, which breaks promise adoption in the browser.
import opencvModule from '@techstark/opencv-js';
import type { CV } from '../pipeline/cv';
import { resolveCv } from '../pipeline/cv';
import { regroup as regroupPieces, runPipeline } from '../pipeline/run';
import type { GridSize, PairDecision, RgbaImage, ScanOptions, StudCell } from '../pipeline/types';
import type { WorkerApi } from './api';

/** Largest image side handed to the pipeline (memory guard; the UI already caps at 4096). */
const MAX_INPUT_SIDE = 6000;

let cvPromise: Promise<CV> | null = null;

function getCv(): Promise<CV> {
  cvPromise ??= resolveCv(opencvModule).catch((e) => {
    cvPromise = null;
    throw e;
  });
  return cvPromise;
}

function bitmapToRgba(image: ImageBitmap): RgbaImage {
  const scale = Math.min(1, MAX_INPUT_SIDE / Math.max(image.width, image.height));
  const width = Math.max(1, Math.round(image.width * scale));
  const height = Math.max(1, Math.round(image.height * scale));
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('OffscreenCanvas 2D context unavailable');
  ctx.drawImage(image, 0, 0, width, height);
  const { data } = ctx.getImageData(0, 0, width, height);
  return { width, height, data };
}

const api: WorkerApi = {
  async init() {
    await getCv();
  },

  async scan(image: ImageBitmap, opts: ScanOptions, onProgress?: (stage: string, frac: number) => void) {
    const report = (stage: string, frac: number) => {
      try {
        void onProgress?.(stage, frac);
      } catch {
        /* progress is best-effort */
      }
    };
    report('load', 0);
    const cv = await getCv();
    report('decode', 0.02);
    const t = performance.now();
    const img = bitmapToRgba(image);
    image.close?.();
    const decodeMs = Math.round(performance.now() - t);
    const res = await runPipeline(cv, img, opts, { onProgress: report });
    res.timingsMs = { decode: decodeMs, ...res.timingsMs };
    return res;
  },

  async regroup(cells: StudCell[], grid: GridSize, pairs: PairDecision[]) {
    return regroupPieces(cells, grid, pairs);
  },
};

Comlink.expose(api);
