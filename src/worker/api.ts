import type { ScanOptions, ScanResult, PairDecision, Piece, StudCell, GridSize } from '../pipeline/types';

/** Comlink API exposed by src/worker/pipeline.worker.ts. UI depends only on this. */
export interface WorkerApi {
  init(): Promise<void>;
  /** Image is decoded on the main thread to an ImageBitmap and transferred. */
  scan(image: ImageBitmap, opts: ScanOptions, onProgress?: (stage: string, frac: number) => void): Promise<ScanResult>;
  /** Re-derive pieces after manual seam edits (no image work). */
  regroup(cells: StudCell[], grid: GridSize, pairs: PairDecision[]): Promise<{ pieces: Piece[]; pieceCount: number; studCount: number; lowConfidenceCount: number }>;
}
