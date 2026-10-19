// Shared contracts between pipeline, worker, UI, eval and ML. Treat as frozen:
// change only with a matching update to every consumer.

export type Point = [x: number, y: number];

/** Four corners of the mosaic in source-image pixels: TL, TR, BR, BL. */
export type Quad = [Point, Point, Point, Point];

export interface GridSize {
  cols: number;
  rows: number;
}

export interface Grid extends GridSize {
  /** Corners in the (downscaled) source image. */
  corners: Quad;
  /** Side length of the rectified square image, px. */
  warpedSize: number;
  /** Stud pitch in rectified px (warpedSize / cols). */
  pitch: number;
  /** 0..1 confidence that the grid fit is right. */
  confidence: number;
}

export interface Rgb {
  r: number;
  g: number;
  b: number;
}

export interface StudCell {
  col: number;
  row: number;
  /** Median colour in the cell, sRGB 0..255. */
  rgb: Rgb;
  /** CIE Lab of the median colour. */
  lab: [L: number, a: number, b: number];
  /** Palette colour id, see src/lib/palette.ts. */
  colorId: string;
  /** Fraction of the cell covered by glare mask, 0..1. */
  glare: number;
}

export type PairDirection = 'h' | 'v';

/** Neighbour pair: (col,row) and its right ('h') or lower ('v') neighbour. */
export interface StudPair {
  col: number;
  row: number;
  dir: PairDirection;
}

export interface PairDecision extends StudPair {
  /** true = a seam separates the two studs (different pieces). */
  seam: boolean;
  /** 0..1 confidence in the decision. */
  confidence: number;
  /** Set when the user overrode the decision in the review editor. */
  manual?: boolean;
}

/** Rectangle on the stud grid, inclusive-exclusive: [x, x+w) × [y, y+h). */
export interface PieceRect {
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
}

export interface Piece extends PieceRect {
  id: number;
  /** Size label, short side first, e.g. "1x4", "2x2". */
  size: string;
  /** Min confidence over its internal pairs and boundary, 0..1. */
  confidence: number;
  /** True if the validator had to split an impossible shape to get this piece. */
  split?: boolean;
}

export interface Breakdown {
  byColor: Record<string, number>;
  bySize: Record<string, number>;
  /** Key "color|size" for the combined parts list. */
  byColorSize: Record<string, number>;
}

export type ScanStatus = 'ok' | 'grid_failed' | 'error';

export interface ScanResult {
  status: ScanStatus;
  message?: string;
  grid?: Grid;
  cells?: StudCell[];
  pairs?: PairDecision[];
  pieces?: Piece[];
  pieceCount: number;
  studCount: number;
  breakdown?: Breakdown;
  lowConfidenceCount: number;
  timingsMs: Record<string, number>;
  classifier: SeamClassifierName;
}

export type SeamClassifierName = 'heuristic' | 'cnn';

export interface ScanOptions {
  /** Force grid size instead of auto-detect. */
  gridOverride?: GridSize;
  /** Use these corners instead of auto quad detection (from the corner editor). */
  corners?: Quad;
  classifier: SeamClassifierName;
  /** If false, studs matching the dominant border/background colour are still counted as pieces. Default true. */
  countBaseplate: boolean;
  /** Max side of the downscaled working image. Default 1600. */
  maxSide?: number;
}

/** Plain image buffer usable in browser workers and Node. RGBA, row-major. */
export interface RgbaImage {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

/** Ground-truth label file, fixtures/labels/<name>.json. */
export interface FixtureLabel {
  image: string;
  grid: GridSize;
  /** Optional; omitted when labelled from a view without pixel coords. */
  corners?: Quad;
  pieces: PieceRect[];
  notes?: string;
}
