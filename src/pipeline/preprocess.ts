// Mosaic quad detection.
//
// Photos are large (e.g. 3213x5712 portrait) with a dark frame, a wide white mat and a
// comparatively small mosaic in the middle. We detect the quad on a small downscaled copy
// (ScanOptions.maxSide is used ONLY as the detection size) and return corners in the
// coordinates of the full-resolution input; later stages warp straight from full-res so
// that thin seams survive.
import type { CV } from './cv';
import { downscale, orderCorners } from './image';
import type { Point, Quad, RgbaImage } from './types';

export const DEFAULT_DETECT_SIDE = 1024;

export interface QuadResult {
  /** TL,TR,BR,BL in full-res source pixels. */
  quad: Quad;
  /** 'override' | 'contour' | 'minAreaRect' | 'fallback' */
  method: string;
  /** 0..1 rough confidence of the detection. */
  confidence: number;
}

/**
 * Find the mosaic: the large textured / saturated / dark region inside the white mat that
 * doesn't touch the image border. Returns a quad in full-res coords.
 */
export function detectQuad(cv: CV, img: RgbaImage, maxSide = DEFAULT_DETECT_SIDE): QuadResult {
  const { img: small, scale } = downscale(img, maxSide);
  const W = small.width, H = small.height;
  const mats: { delete(): void }[] = [];
  const track = <T extends { delete(): void }>(m: T): T => { mats.push(m); return m; };
  try {
    const rgba = track(cv.matFromImageData({ data: small.data, width: W, height: H }));
    const rgb = track(new cv.Mat());
    cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
    const hsv = track(new cv.Mat());
    cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
    const gray = track(new cv.Mat());
    cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY);

    // Texture cue: local mean of gradient magnitude (studs/seams) — rescues white pieces.
    const gx = track(new cv.Mat());
    const gy = track(new cv.Mat());
    cv.Sobel(gray, gx, cv.CV_16S, 1, 0, 3);
    cv.Sobel(gray, gy, cv.CV_16S, 0, 1, 3);
    const ax = track(new cv.Mat());
    const ay = track(new cv.Mat());
    cv.convertScaleAbs(gx, ax);
    cv.convertScaleAbs(gy, ay);
    const mag = track(new cv.Mat());
    cv.addWeighted(ax, 0.5, ay, 0.5, 0, mag);
    const k = Math.max(5, Math.round(Math.min(W, H) / 60)) | 1;
    const tex = track(new cv.Mat());
    cv.blur(mag, tex, new cv.Size(k, k));

    const hd = hsv.data as Uint8Array;
    const td = tex.data as Uint8Array;
    // Adaptive texture threshold: mat is smooth; use a high percentile of the whole image's
    // texture as scale reference.
    const hist = new Uint32Array(256);
    for (let i = 0; i < td.length; i++) hist[td[i]]++;
    const pct = (p: number) => {
      let acc = 0;
      for (let i = 0; i < 256; i++) { acc += hist[i]; if (acc >= p * td.length) return i; }
      return 255;
    };
    const texThr = Math.max(10, pct(0.5) * 2.2, pct(0.97) * 0.3);

    const mask = track(new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(0)));
    const md = mask.data as Uint8Array;
    for (let i = 0, j = 0; i < W * H; i++, j += 3) {
      const s = hd[j + 1], v = hd[j + 2];
      if (s > 45 || v < 140 || td[i] > texThr) md[i] = 255;
    }
    // Clean up: close gaps (white pieces, glare), open specks.
    const kc = Math.max(3, Math.round(Math.min(W, H) / 70)) | 1;
    const kernelC = track(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kc, kc)));
    cv.morphologyEx(mask, mask, cv.MORPH_CLOSE, kernelC);
    const ko = Math.max(3, Math.round(Math.min(W, H) / 120)) | 1;
    const kernelO = track(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(ko, ko)));
    cv.morphologyEx(mask, mask, cv.MORPH_OPEN, kernelO);

    const contours = track(new cv.MatVector());
    const hierarchy = track(new cv.Mat());
    cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    let best = -1, bestScore = 0;
    const border = 2;
    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i);
      const r = cv.boundingRect(c);
      const area = cv.contourArea(c);
      c.delete();
      // Close-up framings crop the mosaic at the photo edge, so border contact is allowed,
      // but background/frame regions touch 3+ sides (or two adjacent ones) and are rejected.
      const tL = r.x <= border, tT = r.y <= border, tR = r.x + r.width >= W - border, tB = r.y + r.height >= H - border;
      const nTouch = +tL + +tT + +tR + +tB;
      // A tightly cropped photo is (almost) all mosaic: allow 3-4 sides when the region fills
      // most of the frame (frame bars and backgrounds never do).
      const fillsFrame = area > W * H * 0.55 && r.width > W * 0.8 && r.height > H * 0.8;
      if (!fillsFrame && (nTouch >= 3 || (tL && tT) || (tT && tR) || (tR && tB) || (tB && tL))) continue;
      const aspect = Math.min(r.width, r.height) / Math.max(r.width, r.height);
      if (aspect < 0.4) continue;
      const fill = area / (r.width * r.height);
      if (fill < 0.5) continue;
      const cx = r.x + r.width / 2, cy = r.y + r.height / 2;
      const dc = Math.hypot((cx - W / 2) / W, (cy - H / 2) / H);
      const score = area * (1 - Math.min(0.8, dc)) * fill * Math.pow(0.85, nTouch);
      if (score > bestScore) { bestScore = score; best = i; }
    }
    if (best < 0 || bestScore < W * H * 0.01) {
      return { quad: fallbackQuad(img), method: 'fallback', confidence: 0.1 };
    }
    const cnt = contours.get(best);
    const hull = track(new cv.Mat());
    cv.convexHull(cnt, hull, false, true);
    cnt.delete();
    const peri = cv.arcLength(hull, true);
    let pts: Point[] | null = null;
    for (const f of [0.01, 0.02, 0.03, 0.045, 0.06, 0.08]) {
      const approx = new cv.Mat();
      cv.approxPolyDP(hull, approx, f * peri, true);
      if (approx.rows === 4) {
        const d = approx.data32S as Int32Array;
        pts = [[d[0], d[1]], [d[2], d[3]], [d[4], d[5]], [d[6], d[7]]];
        approx.delete();
        break;
      }
      approx.delete();
    }
    let method = 'contour';
    // A photographed square stays near-rectangular (small keystone); a polygon with a corner
    // far from 90° means the hull lost a corner (e.g. white border pieces) -> use minAreaRect.
    if (pts && maxCornerDeviation(orderCorners(pts)) > 10) pts = null;
    if (!pts) {
      const rr = cv.minAreaRect(hull);
      const verts = cv.RotatedRect.points(rr) as { x: number; y: number }[];
      pts = verts.map((p) => [p.x, p.y] as Point);
      method = 'minAreaRect';
    }
    const inv = 1 / scale;
    const quad = orderCorners(pts.map(([x, y]) => [(x + 0.5) * inv - 0.5, (y + 0.5) * inv - 0.5] as Point));
    const hullArea = cv.contourArea(hull);
    const qa = quadArea(pts);
    const conf = method === 'contour' ? Math.max(0.3, 1 - Math.abs(1 - hullArea / Math.max(1, qa)) * 4) : 0.4;
    return { quad, method, confidence: conf };
  } finally {
    for (const m of mats) m.delete();
  }
}

/** Largest |angle - 90°| over the corners of an ordered quad, degrees. */
export function maxCornerDeviation(q: Point[]): number {
  let worst = 0;
  for (let i = 0; i < 4; i++) {
    const p = q[i], a = q[(i + 3) % 4], b = q[(i + 1) % 4];
    const v1 = [a[0] - p[0], a[1] - p[1]], v2 = [b[0] - p[0], b[1] - p[1]];
    const cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (Math.hypot(v1[0], v1[1]) * Math.hypot(v2[0], v2[1]) || 1);
    worst = Math.max(worst, Math.abs((Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI - 90));
  }
  return worst;
}

export function quadArea(p: Point[]): number {
  let a = 0;
  for (let i = 0; i < p.length; i++) {
    const [x1, y1] = p[i];
    const [x2, y2] = p[(i + 1) % p.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
}

/** Centred square covering ~45% of the shorter side (typical framed-photo layout). */
export function fallbackQuad(img: RgbaImage): Quad {
  const s = Math.min(img.width, img.height) * 0.45;
  const cx = img.width / 2, cy = img.height / 2;
  return [
    [cx - s, cy - s],
    [cx + s, cy - s],
    [cx + s, cy + s],
    [cx - s, cy + s],
  ];
}
