import type { PairDecision, StudCell, StudPair, Grid } from '../types';

/** Everything a seam classifier may look at for one image. */
export interface SeamContext {
  /** Rectified square RGBA image, side = grid.warpedSize. */
  warped: { width: number; height: number; data: Uint8ClampedArray };
  grid: Grid;
  /** cells[row * cols + col] */
  cells: StudCell[];
  /** Glare mask over the warped image, 0/255, same size. */
  glareMask: Uint8Array;
}

export interface SeamClassifier {
  readonly name: 'heuristic' | 'cnn';
  /** Decide seam vs same-piece for every pair. Must return one decision per input pair, same order. */
  classify(ctx: SeamContext, pairs: StudPair[]): Promise<PairDecision[]>;
}

/**
 * Crop spec shared with ml/: a 32x16 (w x h) RGB crop centred on the gap between
 * two studs, oriented so the gap is a vertical line in the middle (vertical pairs
 * are rotated 90°). Crop spans one pitch either side horizontally and half a pitch
 * vertically, resampled bilinearly. Normalisation: float32 CHW, value/255.
 */
export const CROP_W = 32;
export const CROP_H = 16;
