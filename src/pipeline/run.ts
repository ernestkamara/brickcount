import type { RgbaImage, ScanOptions, ScanResult } from './types';

/** Full pipeline entry. Pure w.r.t. inputs; caller supplies an initialised cv (OpenCV.js). */
export async function runPipeline(_cv: unknown, _img: RgbaImage, opts: ScanOptions): Promise<ScanResult> {
  return { status: 'error', message: 'not implemented', pieceCount: 0, studCount: 0, lowConfidenceCount: 0, timingsMs: {}, classifier: opts.classifier };
}
