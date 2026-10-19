"""Seam-crop dataset generator for BrickCount.

Renders random rectified mosaics (ml/render_synthetic.py), applies photo/glass degradations at a
"full-resolution" pitch, resamples to a "warped" pitch (like the pipeline's rectification, ~40-90 px
per stud), then cuts 32x16 crops with EXACTLY the crop spec shared with src/pipeline/seams:

  Pair (col,row,'h'): boundary centre c = ((col+1)*P, (row+0.5)*P), ex = (1,0), ey = (0,1)
  Pair (col,row,'v'): boundary centre c = ((col+0.5)*P, (row+1)*P), ex = (0,1), ey = (-1,0)
      (vertical pairs rotated 90 deg: the upper stud lands on the left half of the crop)
  Crop pixel (i,j), i in 0..31, j in 0..15, samples the warped image at
      c + ((i + 0.5 - 16) * ex + (j + 0.5 - 8) * ey) * (P / 16)
  i.e. a 2P x P window centred on the gap, bilinear, where image pixel (k,l) has its centre at
  (k + 0.5, l + 0.5). Values RGB / 255, CHW.  Label 1 = seam (different pieces).

Training jitter on top of the spec: pitch scale +-10%, centre offset +-15% pitch (both axes),
rotation +-4 deg; optional prefilter blur.

  python ml/dataset.py --mosaics 700 --per 300 --out ml/data/seams.npz
"""
from __future__ import annotations

import argparse
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

import render_synthetic as RS

CROP_W, CROP_H = 32, 16


def pair_frame(col, row, d, P):
    """Centre and crop axes (ex: across the gap, ey: along the gap) in warped px."""
    if d == 0:  # 'h'
        return np.array([(col + 1) * P, (row + 0.5) * P]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    return np.array([(col + 0.5) * P, (row + 1) * P]), np.array([0.0, 1.0]), np.array([-1.0, 0.0])


def crop_coords(c, ex, ey, step):
    i = (np.arange(CROP_W) + 0.5 - CROP_W / 2) * step
    j = (np.arange(CROP_H) + 0.5 - CROP_H / 2) * step
    X = c[0] + i[None, :] * ex[0] + j[:, None] * ey[0]
    Y = c[1] + i[None, :] * ex[1] + j[:, None] * ey[1]
    return X, Y


def bilinear(img, X, Y):
    """Sample img (H,W,C) at continuous coords (pixel centres at k+0.5), clamp-to-edge."""
    H, W = img.shape[:2]
    fx = X - 0.5
    fy = Y - 0.5
    x0 = np.floor(fx).astype(np.int64); y0 = np.floor(fy).astype(np.int64)
    ax = (fx - x0)[..., None]; ay = (fy - y0)[..., None]
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x0 + 1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y0 + 1, 0, H - 1)
    a = img[y0c, x0c]; b = img[y0c, x1c]; c = img[y1c, x0c]; d = img[y1c, x1c]
    return (a * (1 - ax) + b * ax) * (1 - ay) + (c * (1 - ax) + d * ax) * ay


def extract_crop(img, col, row, d, P, rng=None, jitter=True):
    c, ex, ey = pair_frame(col, row, d, P)
    step = P / 16.0
    if jitter and rng is not None:
        step *= rng.uniform(0.9, 1.1)
        c = c + ex * rng.uniform(-0.15, 0.15) * P + ey * rng.uniform(-0.15, 0.15) * P
        th = math.radians(rng.uniform(-4, 4))
        cs, sn = math.cos(th), math.sin(th)
        rot = np.array([[cs, -sn], [sn, cs]])
        ex, ey = rot @ ex, rot @ ey
    X, Y = crop_coords(c, ex, ey, step)
    return bilinear(img, X, Y)


def one_mosaic(args):
    seed, per = args
    rng = np.random.default_rng(seed)
    if rng.random() < 0.7:
        cols = rows = 16
    else:
        cols, rows = int(rng.integers(8, 21)), int(rng.integers(8, 21))
    owner, pieces = RS.random_layout(rng, cols, rows)
    Pw = rng.uniform(40, 90)                                   # warped pitch (pipeline ~64)
    Pr = int(np.clip(Pw * rng.uniform(1.0, 2.0), 40, 130))     # "photo" pitch
    img = RS.render_mosaic(owner, pieces, Pr, rng)
    img = RS.photo_effects(img, rng, scale=Pr)
    img = RS.jpeg(img.astype(np.uint8), rng).astype(np.float32)
    Wt, Ht = int(round(Pw * cols)), int(round(Pw * rows))
    interp = cv2.INTER_LINEAR if rng.random() < 0.6 else cv2.INTER_AREA
    warped = cv2.resize(img, (Wt, Ht), interpolation=interp)
    P = Wt / cols  # actual pitch after rounding
    if rng.random() < 0.3:  # some pipelines prefilter before point sampling
        warped = cv2.GaussianBlur(warped, (0, 0), 0.5 * P / 16)
    if rng.random() < 0.3:  # extra glare after warp
        warped = RS.add_glare(warped, rng, n=int(rng.integers(1, 3)), scale=P)

    ci = np.array([p["ci"] for p in pieces])
    pairs = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                pairs.append((c, r, 0, owner[r, c] != owner[r, c + 1], ci[owner[r, c]] == ci[owner[r, c + 1]]))
            if r + 1 < rows:
                pairs.append((c, r, 1, owner[r, c] != owner[r + 1, c], ci[owner[r, c]] == ci[owner[r + 1, c]]))
    pairs = np.array(pairs, dtype=np.int64)
    groups = [np.nonzero(~pairs[:, 4].astype(bool))[0],                              # diff colour (seam)
              np.nonzero(pairs[:, 4].astype(bool) & pairs[:, 3].astype(bool))[0],    # same colour seam
              np.nonzero(pairs[:, 4].astype(bool) & ~pairs[:, 3].astype(bool))[0]]   # same colour, same piece
    want = [int(per * 0.15), int(per * 0.425), per - int(per * 0.15) - int(per * 0.425)]
    sel = []
    for g, w in zip(groups, want):
        if len(g):
            sel.append(rng.choice(g, size=min(w, len(g)), replace=False))
    sel = np.concatenate(sel)
    X = np.empty((len(sel), CROP_H, CROP_W, 3), np.uint8)
    for k, idx in enumerate(sel):
        c, r, d = pairs[idx, :3]
        X[k] = np.clip(np.round(extract_crop(warped, c, r, d, P, rng)), 0, 255).astype(np.uint8)
    y = pairs[sel, 3].astype(np.uint8)
    same = pairs[sel, 4].astype(np.uint8)
    return X, y, same, np.full(len(sel), seed, np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mosaics", type=int, default=700)
    ap.add_argument("--per", type=int, default=300)
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "seams.npz"))
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    jobs = [(a.seed + i, a.per) for i in range(a.mosaics)]
    Xs, ys, ss, ms = [], [], [], []
    with Pool(a.workers) as pool:
        for i, (X, y, s, m) in enumerate(pool.imap_unordered(one_mosaic, jobs, chunksize=2)):
            Xs.append(X); ys.append(y); ss.append(s); ms.append(m)
            if i % 50 == 0:
                print(f"{i}/{a.mosaics} mosaics, {time.time() - t:.0f}s", flush=True)
    X = np.concatenate(Xs); y = np.concatenate(ys); s = np.concatenate(ss); m = np.concatenate(ms)
    np.savez(a.out, X=X, y=y, same=s, mosaic=m)
    print(f"saved {len(y)} crops to {a.out}: seam={y.mean():.3f} same_colour={s.mean():.3f} "
          f"same&seam={(y[s == 1]).mean():.3f} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
