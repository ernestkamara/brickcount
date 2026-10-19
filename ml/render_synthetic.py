"""Synthetic LEGO mosaic generator + photo renderer for BrickCount.

Library part (imported by dataset.py):
  random_layout(rng, cols, rows)          -> (owner[r,c] piece index, pieces list)
  render_mosaic(owner, pieces, P, rng)    -> rectified float32 RGB image (rows*P x cols*P), 0..255
  photo_effects(img, rng, ...)            -> lighting / glare / haze / blur / noise (float32 in, float32 out)

CLI part: renders full framed "phone photos" with labels in FixtureLabel format.

  python ml/render_synthetic.py --n 30 --out fixtures/synthetic            # 16x16 (some 32x32)
  python ml/render_synthetic.py --n 5 --grid 32 --out /tmp/synth32

Coordinate convention (shared with crop spec): pixel (i, j) covers [i, i+1) x [j, j+1), its
centre is at (i + 0.5, j + 0.5). Label corners use the same convention.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ML_DIR = Path(__file__).resolve().parent
_PAL = json.loads((ML_DIR / "palette.json").read_text())
PALETTE_IDS: list[str] = [p["id"] for p in _PAL["palette"]]
PALETTE_RGB = np.array([p["rgb"] for p in _PAL["palette"]], dtype=np.float32)
PID = {k: i for i, k in enumerate(PALETTE_IDS)}
SIZES: list[tuple[int, int]] = [tuple(s) for s in _PAL["sizes"]]
# both orientations
ORIENTED = sorted({(a, b) for a, b in SIZES} | {(b, a) for a, b in SIZES})

BG_WEIGHTS = {
    "medium_azure": 3.0, "black": 1.6, "blue": 1.4, "dark_azure": 1.0, "white": 1.0,
    "light_bluish_gray": 1.0, "tan": 0.7, "light_yellow": 0.5, "green": 0.5, "dark_bluish_gray": 0.5,
    "yellow": 0.3, "bright_pink": 0.3, "lime": 0.2, "light_nougat": 0.3, "red": 0.3, "orange": 0.2,
    "dark_red": 0.2, "reddish_brown": 0.2, "dark_brown": 0.15,
}
OUTLINE_COLOURS = ["black", "black", "black", "dark_brown", "reddish_brown", "dark_bluish_gray", "blue"]


# --------------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------------

def _pick(rng, weights: dict[str, float], exclude=()) -> int:
    keys = [k for k in weights if k not in exclude]
    w = np.array([weights[k] for k in keys], dtype=np.float64)
    return PID[keys[rng.choice(len(keys), p=w / w.sum())]]


def _rand_colour(rng, exclude=()) -> int:
    choices = [i for i in range(len(PALETTE_IDS)) if i not in exclude]
    return int(rng.choice(choices))


def _sprite_mask(rng, h, w):
    """Random left-right symmetric pixel-art blob."""
    half = (w + 1) // 2
    m = rng.random((h, half)) < rng.uniform(0.45, 0.7)
    for _ in range(rng.integers(0, 3)):  # majority smoothing
        p = np.pad(m.astype(np.int32), 1)
        s = sum(p[1 + dy:1 + dy + h, 1 + dx:1 + dx + half] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        m = s >= 5
    full = np.concatenate([m, m[:, ::-1][:, (w % 2):]], axis=1)[:, :w]
    return full


def random_colour_map(rng, cols, rows) -> np.ndarray:
    cmap = np.full((rows, cols), _pick(rng, BG_WEIGHTS), dtype=np.int32)
    bg = int(cmap[0, 0])
    yy, xx = np.mgrid[0:rows, 0:cols]
    # optional second background region (ground / band / sun)
    r = rng.random()
    if r < 0.25:
        ground = _rand_colour(rng, exclude=(bg,))
        level = rng.integers(rows * 2 // 3, rows)
        wave = level + np.round(rng.uniform(0, 1.5) * np.sin(xx / rng.uniform(2, 6) + rng.uniform(0, 6))).astype(int)
        cmap[yy >= wave] = ground
    elif r < 0.4:
        c = _rand_colour(rng, exclude=(bg,))
        cy, cx, rad = rng.uniform(0, rows), rng.uniform(0, cols), rng.uniform(2, max(3, cols / 4))
        cmap[(yy - cy) ** 2 + (xx - cx) ** 2 < rad ** 2] = c
    elif r < 0.5:
        c = _rand_colour(rng, exclude=(bg,))
        a, b = sorted(rng.integers(0, rows, 2))
        cmap[a:b + 1, :] = c
    # pixel-art figures
    n_fig = rng.integers(1, 4 if cols <= 16 else 7)
    for _ in range(n_fig):
        kind = rng.random()
        h = int(rng.integers(3, max(4, int(rows * 0.8))))
        w = int(rng.integers(3, max(4, int(cols * 0.8))))
        y0 = int(rng.integers(-1, max(0, rows - h) + 1))
        x0 = int(rng.integers(-1, max(0, cols - w) + 1))
        if kind < 0.6:
            m = _sprite_mask(rng, h, w)
        elif kind < 0.8:
            yy2, xx2 = np.mgrid[0:h, 0:w]
            m = ((yy2 - (h - 1) / 2) / (h / 2)) ** 2 + ((xx2 - (w - 1) / 2) / (w / 2)) ** 2 <= 1.0
        else:
            m = np.ones((h, w), bool)
        fill = _rand_colour(rng, exclude=(bg,))
        canvas = np.zeros((rows, cols), bool)
        ys, xs = np.nonzero(m)
        ys, xs = ys + y0, xs + x0
        ok = (ys >= 0) & (ys < rows) & (xs >= 0) & (xs < cols)
        canvas[ys[ok], xs[ok]] = True
        cmap[canvas] = fill
        if rng.random() < 0.6:  # outline
            p = np.pad(canvas, 1)
            inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
            edge = canvas & ~inner
            cmap[edge] = _pick(rng, {k: 1.0 for k in OUTLINE_COLOURS}, exclude=(PALETTE_IDS[fill],))
        for _ in range(rng.integers(0, 4)):  # detail pixels / short strokes
            if not canvas.any():
                break
            ys, xs = np.nonzero(canvas)
            k = rng.integers(len(ys))
            c = _rand_colour(rng)
            ln = rng.integers(1, 4)
            if rng.random() < 0.5:
                cmap[ys[k], xs[k]:xs[k] + ln] = c
            else:
                cmap[ys[k]:ys[k] + ln, xs[k]] = c
    return cmap


def _size_weights(rng):
    style = rng.choice(["long", "mixed", "big", "small"], p=[0.4, 0.3, 0.15, 0.15])
    w = []
    for a, b in ORIENTED:
        area = a * b
        lo, hi = min(a, b), max(a, b)
        if style == "long":
            v = {1: 1.0, 2: 1.2}.get(lo, 0.3) * (2.5 if hi in (4, 6) else 1.0) * (0.15 if area == 1 else 1)
        elif style == "mixed":
            v = 1.0
        elif style == "big":
            v = area ** 0.8
        else:
            v = 1.0 / area
        w.append(v)
    w = np.array(w, dtype=np.float64)
    return w / w.sum()


def tile_colour_map(cmap: np.ndarray, rng):
    """Greedily tile every same-colour region with legal plates. Returns owner grid + pieces."""
    k = int(rng.integers(0, 4))
    # random scan direction: transform, tile row-major, transform back
    tf = [lambda a: a, lambda a: a[:, ::-1], lambda a: a[::-1, :], lambda a: a.T][k]
    cm = np.ascontiguousarray(tf(cmap))
    rows, cols = cm.shape
    owner = -np.ones((rows, cols), np.int32)
    weights = _size_weights(rng)
    rects = []
    for r in range(rows):
        for c in range(cols):
            if owner[r, c] >= 0:
                continue
            col = cm[r, c]
            order = rng.choice(len(ORIENTED), size=len(ORIENTED), replace=False, p=weights)
            placed = False
            for oi in order:
                h, w = ORIENTED[oi]
                if r + h > rows or c + w > cols:
                    continue
                if (owner[r:r + h, c:c + w] >= 0).any() or (cm[r:r + h, c:c + w] != col).any():
                    continue
                owner[r:r + h, c:c + w] = len(rects)
                rects.append((c, r, w, h, int(col)))
                placed = True
                break
            if not placed:  # 1x1 always fits
                owner[r, c] = len(rects)
                rects.append((c, r, 1, 1, int(col)))
    # inverse transform of owner; rebuild rects from owner
    inv = [lambda a: a, lambda a: a[:, ::-1], lambda a: a[::-1, :], lambda a: a.T][k]
    owner = np.ascontiguousarray(inv(owner))
    pieces = []
    for i, (_, _, _, _, col) in enumerate(rects):
        ys, xs = np.nonzero(owner == i)
        pieces.append({"x": int(xs.min()), "y": int(ys.min()), "w": int(xs.max() - xs.min() + 1),
                       "h": int(ys.max() - ys.min() + 1), "color": PALETTE_IDS[col], "ci": col})
    return owner, pieces


def random_layout(rng, cols=16, rows=16):
    cmap = random_colour_map(rng, cols, rows)
    return tile_colour_map(cmap, rng)


# --------------------------------------------------------------------------------------------
# Rectified mosaic rendering
# --------------------------------------------------------------------------------------------

_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def _text_mask(size: int) -> np.ndarray:
    fs = max(8, size)
    if fs not in _FONT_CACHE:
        _FONT_CACHE[fs] = ImageFont.load_default(size=fs)
    im = Image.new("L", (fs * 4, fs * 2))
    ImageDraw.Draw(im).text((fs // 4, fs // 4), "LEGO", font=_FONT_CACHE[fs], fill=255)
    bb = im.getbbox()
    return np.asarray(im.crop(bb), dtype=np.float32) / 255.0


def _stud_templates(P: int, rng, light: np.ndarray, n_var: int = 4):
    """Per-cell multiplicative (M) and additive (A) maps for a stud, n_var text orientations."""
    c = (np.arange(P, dtype=np.float32) + 0.5) - P / 2.0
    v, u = np.meshgrid(c, c, indexing="ij")  # v = y offset, u = x offset
    R = P * rng.uniform(0.28, 0.31)
    rr = np.sqrt(u ** 2 + v ** 2) + 1e-6
    aa = max(0.7, P * 0.012)
    inside = np.clip((R - rr) / aa + 0.5, 0, 1)

    # cast shadow (away from light)
    off = -light * P * rng.uniform(0.04, 0.10)
    rs = np.sqrt((u - off[0]) ** 2 + (v - off[1]) ** 2)
    soft = P * rng.uniform(0.03, 0.08)
    shadow = np.clip((R + soft - rs) / (2 * soft), 0, 1) * (1 - inside)
    M = 1.0 - rng.uniform(0.15, 0.45) * shadow

    # rim shading on the stud top edge
    ring_w = P * rng.uniform(0.04, 0.08)
    ring = np.clip(1 - np.abs(rr - (R - ring_w / 2)) / (ring_w / 2), 0, 1) * inside
    ndot = (u * light[0] + v * light[1]) / rr
    M = M * (1 + rng.uniform(0.08, 0.25) * ndot * ring)
    # dark outline at stud edge
    outline = np.clip(1 - np.abs(rr - R) / max(0.8, P * 0.015), 0, 1)
    M = M * (1 - rng.uniform(0.1, 0.45) * outline)
    # stud top slightly different brightness
    M = M * (1 + rng.uniform(-0.04, 0.06) * inside)
    A = rng.uniform(0.05, 0.35) * 255 * np.clip(ndot, 0, 1) ** 4 * ring

    # embossed LEGO text
    tw = int(R * rng.uniform(1.2, 1.5))
    tm = _text_mask(max(8, int(tw / 2.6)))
    th = max(2, int(tm.shape[0] * tw / tm.shape[1]))
    tm = cv2.resize(tm, (tw, th), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.7:  # italic-ish shear
        sh = 0.25 * th
        Ms = np.float32([[1, -0.25, sh], [0, 1, 0]])
        tm = cv2.warpAffine(tm, Ms, (tw + int(sh) + 1, th))
    strength = rng.uniform(0.05, 0.25)
    Ms_, As_ = [], []
    for k in range(n_var):
        t = np.rot90(tm, k)
        cell = np.zeros((P, P), np.float32)
        y0, x0 = (P - t.shape[0]) // 2, (P - t.shape[1]) // 2
        if y0 >= 0 and x0 >= 0:
            cell[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
        cell = cv2.GaussianBlur(cell, (0, 0), max(0.3, P * 0.006))
        dx, dy = light * max(0.7, P * 0.012)
        sh_ = cv2.warpAffine(cell, np.float32([[1, 0, dx], [0, 1, dy]]), (P, P))
        emb = (sh_ - cell) * inside  # edge facing light bright, other dark
        Ms_.append(M * (1 + strength * emb) * (1 - strength * 0.5 * cell * inside))
        As_.append(A)
    return np.stack(Ms_).astype(np.float32), np.stack(As_).astype(np.float32)


def render_mosaic(owner: np.ndarray, pieces: list[dict], P: int, rng, light=None) -> np.ndarray:
    """Rectified top-down mosaic, float32 HxWx3 0..255 (before camera effects)."""
    rows, cols = owner.shape
    H, W = rows * P, cols * P
    if light is None:
        ang = rng.uniform(math.radians(-170), math.radians(-10)) if rng.random() < 0.8 else rng.uniform(-math.pi, math.pi)
        light = np.array([math.cos(ang), math.sin(ang)], np.float32)  # direction TOWARDS the light
    n = len(pieces)
    X0 = np.array([p["x"] for p in pieces], np.float32) * P
    Y0 = np.array([p["y"] for p in pieces], np.float32) * P
    X1 = X0 + np.array([p["w"] for p in pieces], np.float32) * P
    Y1 = Y0 + np.array([p["h"] for p in pieces], np.float32) * P
    col = PALETTE_RGB[[p["ci"] for p in pieces]].copy()
    # colour variation (per image tint + per piece)
    col *= rng.uniform(0.94, 1.06, (1, 3)).astype(np.float32)
    bright_sd = rng.uniform(0.0, 0.035)
    pb = (1 + rng.normal(0, bright_sd, (n, 1))).astype(np.float32)
    col = col * pb

    # seam params: per image + per piece/side
    line_hw = P * rng.uniform(0.006, 0.022)                  # half core width (≈1..4px at P=90)
    dark = rng.uniform(0.15, 0.85) if rng.random() < 0.85 else rng.uniform(0.05, 0.2)
    gap_max = P * rng.uniform(0.0, 0.05)
    bev_w = P * rng.uniform(0.01, 0.05)
    bev_s = rng.uniform(0.0, 0.25)
    hw_side = (line_hw * rng.uniform(0.6, 1.4, (n, 4))).astype(np.float32)
    gap_side = (gap_max * rng.random((n, 4)) ** 2).astype(np.float32)
    dark_side = np.clip(dark * rng.uniform(0.6, 1.3, (n, 4)), 0, 0.95).astype(np.float32)

    ys = np.arange(H, dtype=np.float32) + 0.5
    xs = np.arange(W, dtype=np.float32) + 0.5
    cell_r = np.minimum((ys // P).astype(np.int32), rows - 1)
    cell_c = np.minimum((xs // P).astype(np.int32), cols - 1)
    pid = owner[cell_r[:, None], cell_c[None, :]]
    d = np.stack([xs[None, :] - X0[pid], X1[pid] - xs[None, :], ys[:, None] - Y0[pid], Y1[pid] - ys[:, None]])  # L R T B
    normals = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], np.float32)
    seam = np.zeros((H, W), np.float32)
    bevel = np.zeros((H, W), np.float32)
    for s in range(4):
        hw = hw_side[pid, s]
        gp = gap_side[pid, s]
        ds = d[s]
        core = np.clip(hw - ds + 0.5, 0, 1)
        gapz = np.clip(hw + gp - ds + 0.5, 0, 1)
        seam = np.maximum(seam, dark_side[pid, s] * np.maximum(core, 0.45 * gapz))
        bz = np.clip(1 - (ds - hw - gp) / bev_w, 0, 1) * (ds > hw)
        bevel += bev_s * float(normals[s] @ light) * bz
    img = col[pid] * (1 + bevel)[..., None] * (1 - seam)[..., None]

    # studs
    Ms, As = _stud_templates(P, rng, light)
    var = rng.integers(0, len(Ms), (rows, cols))
    Mt = Ms[var].transpose(0, 2, 1, 3).reshape(H, W)
    At = As[var].transpose(0, 2, 1, 3).reshape(H, W)
    img = img * Mt[..., None] + At[..., None]
    # surface noise / texture
    img += rng.normal(0, rng.uniform(0.5, 2.5), (H, W, 1)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.float32)


# --------------------------------------------------------------------------------------------
# Camera / glass effects
# --------------------------------------------------------------------------------------------

def add_glare(img, rng, n=None, scale=1.0, region=None):
    """Bright specular ovals + faint scratches. region=(x0,y0,x1,y1) bias for ellipse centres."""
    H, W = img.shape[:2]
    if n is None:
        n = int(rng.integers(0, 4))
    if region is None:
        region = (0, 0, W, H)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for _ in range(n):
        cx = rng.uniform(region[0], region[2]); cy = rng.uniform(region[1], region[3])
        a = rng.uniform(0.3, 2.0) * scale
        b = a * rng.uniform(0.15, 0.6)
        th = rng.uniform(-0.4, 0.4)
        c, s = math.cos(th), math.sin(th)
        x0 = int(max(0, cx - a - 4)); x1 = int(min(W, cx + a + 4)); y0 = int(max(0, cy - a - 4)); y1 = int(min(H, cy + a + 4))
        if x1 <= x0 or y1 <= y0:
            continue
        dx = xx[y0:y1, x0:x1] - cx; dy = yy[y0:y1, x0:x1] - cy
        r = np.sqrt(((dx * c + dy * s) / a) ** 2 + ((-dx * s + dy * c) / b) ** 2)
        hard = rng.uniform(0.05, 0.6)
        alpha = np.clip((1 - r) / hard, 0, 1) * rng.uniform(0.5, 1.0)
        tint = np.array([1.0, 1.0, rng.uniform(1.0, 1.05)], np.float32) * rng.uniform(235, 255)
        sub = img[y0:y1, x0:x1]
        img[y0:y1, x0:x1] = sub + (tint - sub) * alpha[..., None]
    # broad sheen
    if rng.random() < 0.5:
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        sig = rng.uniform(0.2, 0.6) * max(H, W)
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2)) * rng.uniform(0.03, 0.15)
        img += (255 - img) * g[..., None]
    # scratches / dust
    if rng.random() < 0.6:
        lay = np.zeros((H, W), np.float32)
        for _ in range(int(rng.integers(1, 12))):
            p0 = (int(rng.uniform(0, W)), int(rng.uniform(0, H)))
            ln = rng.uniform(0.05, 0.6) * scale
            ang = rng.uniform(0, math.pi)
            p1 = (int(p0[0] + ln * math.cos(ang)), int(p0[1] + ln * math.sin(ang)))
            cv2.line(lay, p0, p1, float(rng.uniform(0.1, 0.5)), 1, cv2.LINE_AA)
        img += (255 - img) * lay[..., None]
    return img


def photo_effects(img, rng, scale=1.0, glare=True, glare_region=None, blur_px=None):
    """Lighting gradient, white balance, haze (washed out), glare, blur, noise. float32 0..255 in/out.
    `scale` ~ size of one stud pitch in px (sets glare / blur sizes)."""
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    th = rng.uniform(0, 2 * math.pi)
    g = ((xx / W - 0.5) * math.cos(th) + (yy / H - 0.5) * math.sin(th))
    light = 1 + rng.uniform(0.0, 0.35) * g
    vig = 1 - rng.uniform(0, 0.15) * (((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) * 2)
    img = img * (light * vig)[..., None] * rng.uniform(0.85, 1.12)
    img *= rng.uniform(0.92, 1.08, 3).astype(np.float32)  # white balance
    haze = rng.uniform(0.0, 0.25)
    img = img * (1 - haze) + haze * rng.uniform(170, 240)
    if glare:
        img = add_glare(img, rng, scale=scale, region=glare_region)
    if blur_px is None:
        blur_px = rng.uniform(0, 0.025) * scale
    if blur_px > 0.2:
        img = cv2.GaussianBlur(img, (0, 0), blur_px)
    if rng.random() < 0.25:  # motion blur
        L = int(rng.integers(2, max(3, int(0.06 * scale) + 3)))
        k = np.zeros((L, L), np.float32)
        ang = rng.uniform(0, math.pi)
        cv2.line(k, (L // 2 - int(L / 2 * math.cos(ang)), L // 2 - int(L / 2 * math.sin(ang))),
                 (L // 2 + int(L / 2 * math.cos(ang)), L // 2 + int(L / 2 * math.sin(ang))), 1.0, 1)
        k /= max(k.sum(), 1e-6)
        img = cv2.filter2D(img, -1, k)
    img = img + rng.normal(0, rng.uniform(0.5, 4.0), img.shape).astype(np.float32)
    return np.clip(img, 0, 255)


def jpeg(img_u8: np.ndarray, rng, q=None) -> np.ndarray:
    q = int(q if q is not None else rng.integers(55, 96))
    ok, buf = cv2.imencode(".jpg", img_u8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]


# --------------------------------------------------------------------------------------------
# Full framed photo
# --------------------------------------------------------------------------------------------

def render_photo(rng, cols=16, rows=16, out_w=1600, out_h=2844):
    owner, pieces = random_layout(rng, cols, rows)
    mosaic_frac = rng.uniform(0.40, 0.75)          # mosaic width / image width
    target_pitch = out_w * mosaic_frac / cols
    P = int(min(150, max(24, round(target_pitch * rng.uniform(1.2, 1.5)))))
    mos = render_mosaic(owner, pieces, P, rng)
    S = cols * P
    Hm = rows * P

    # scene: mosaic + optional baseplate rim + mat (with opening bevel) + black frame + background
    bp = int(P * rng.uniform(0.1, 0.4)) if rng.random() < 0.3 else 0
    overlap = int(P * rng.uniform(0, 0.25)) if rng.random() < 0.5 else 0  # mat covers mosaic edge
    # real photos: the white mat runs past the left/right image edges, black frame bars top/bottom
    mat = int(S * max(rng.uniform(0.25, 0.4), 0.525 / mosaic_frac - 0.5 + rng.uniform(0.0, 0.1)))
    frame = int(S * rng.uniform(0.05, 0.10))
    beyond = int(S * 0.9)
    off = beyond + frame + mat + bp
    SW, SH = S + 2 * off, Hm + 2 * off
    scene = np.empty((SH, SW, 3), np.float32)
    scene[:] = rng.uniform(15, 60) * rng.uniform(0.9, 1.1, 3)                    # background
    fcol = rng.uniform(10, 45) * rng.uniform(0.95, 1.05, 3)
    scene[beyond:SH - beyond, beyond:SW - beyond] = fcol                           # frame
    m0 = beyond + frame
    matc = rng.uniform(230, 252) * rng.uniform(0.97, 1.03, 3)
    scene[m0:SH - m0, m0:SW - m0] = matc
    # frame inner shadow on the mat
    sh_w = int(S * rng.uniform(0.01, 0.04))
    for i in range(sh_w):
        f = 1 - 0.35 * (1 - i / sh_w) ** 2
        scene[m0 + i, m0:SW - m0] *= f
    bpcol = PALETTE_RGB[rng.choice([PID["light_bluish_gray"], PID["white"], PID["black"], PID["blue"], PID["green"]])]
    scene[off - bp:off + Hm + bp, off - bp:off + S + bp] = bpcol
    scene[off:off + Hm, off:off + S] = mos
    # mat opening bevel (thin inner frame border), covering `overlap` px of mosaic
    bev = max(2, int(S * rng.uniform(0.004, 0.012)))
    o0, o1y, o1x = off + overlap, off + Hm - overlap, off + S - overlap
    mat_mask = np.ones((SH, SW), bool)
    mat_mask[o0:o1y, o0:o1x] = False
    inner = np.zeros((SH, SW), bool)
    inner[m0:SH - m0, m0:SW - m0] = True
    mat_area = mat_mask & inner
    scene[mat_area] = matc
    # bevel band: lighter/darker per side
    bshade = rng.uniform(0.75, 1.0)
    scene[o0 - bev:o0, o0 - bev:o1x + bev] = matc * bshade          # top bevel (shadowed)
    scene[o1y:o1y + bev, o0 - bev:o1x + bev] = np.minimum(matc * 1.03, 255)
    scene[o0 - bev:o1y + bev, o0 - bev:o0] = matc * (bshade + 0.1)
    scene[o0 - bev:o1y + bev, o1x:o1x + bev] = matc * 0.97
    # mat casts soft shadow onto the mosaic
    shadow = np.zeros((SH, SW), np.float32)
    shadow[o0:o0 + max(1, int(P * 0.15)), o0:o1x] = 0.25
    shadow[o0:o1y, o0:o0 + max(1, int(P * 0.08))] = 0.15
    shadow = cv2.GaussianBlur(shadow, (0, 0), max(1, P * 0.05))
    scene *= (1 - shadow)[..., None]
    scene += rng.normal(0, 1.0, (SH, SW, 1)).astype(np.float32)

    # camera: 3D tilt (<= 25 deg), small roll, perspective projection
    tilt = math.radians(rng.uniform(0, 12) if rng.random() < 0.7 else rng.uniform(12, 25))
    phi = rng.uniform(0, 2 * math.pi)
    ax, ay = tilt * math.cos(phi), tilt * math.sin(phi)
    rz = math.radians(rng.uniform(-6, 6))
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Rx @ Ry
    D = S * rng.uniform(1.6, 3.0)
    cx, cy = SW / 2, SH / 2

    def project(pts):
        X = np.c_[pts[:, 0] - cx, pts[:, 1] - cy, np.zeros(len(pts))] @ R.T
        return np.c_[X[:, 0] / (X[:, 2] + D), X[:, 1] / (X[:, 2] + D)] * D

    # cv2 coords: pixel centre at integer -> mosaic edge at off - 0.5
    mos_c = np.array([[off - 0.5, off - 0.5], [off + S - 0.5, off - 0.5], [off + S - 0.5, off + Hm - 0.5], [off - 0.5, off + Hm - 0.5]])
    pm = project(mos_c)
    span = max(pm[:, 0].max() - pm[:, 0].min(), 1)
    sc = out_w * mosaic_frac / span
    pm_c = pm.mean(0)
    shift = np.array([out_w / 2 + rng.uniform(-0.06, 0.06) * out_w, out_h / 2 + rng.uniform(-0.06, 0.06) * out_h])
    to_out = lambda p: (p - pm_c) * sc + shift
    src4 = np.array([[0, 0], [SW - 1, 0], [SW - 1, SH - 1], [0, SH - 1]], np.float64)
    Hmat = cv2.getPerspectiveTransform(src4.astype(np.float32), to_out(project(src4)).astype(np.float32))
    photo = cv2.warpPerspective(scene, Hmat, (out_w, out_h), flags=cv2.INTER_AREA if sc < 0.8 else cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    corners = to_out(pm) + 0.5  # -> pixel-edge convention

    # glare on mat and mosaic
    qx0, qy0 = corners[:, 0].min(), corners[:, 1].min()
    qx1, qy1 = corners[:, 0].max(), corners[:, 1].max()
    pitch_out = (qx1 - qx0) / cols
    photo = add_glare(photo, rng, n=int(rng.integers(1, 4)), scale=pitch_out * 1.5,
                      region=(qx0, qy0, qx1, qy1))
    photo = add_glare(photo, rng, n=int(rng.integers(0, 3)), scale=pitch_out * 2.0,
                      region=(qx0 - 0.3 * (qx1 - qx0), qy0 - 0.5 * (qy1 - qy0), qx1 + 0.3 * (qx1 - qx0), qy1 + 0.5 * (qy1 - qy0)))
    photo = photo_effects(photo, rng, scale=pitch_out, glare=False, blur_px=rng.uniform(0, 1.6))
    out = jpeg(photo.astype(np.uint8), rng, q=int(rng.integers(70, 95)))
    label_pieces = [{k: p[k] for k in ("x", "y", "w", "h", "color")} for p in pieces]
    return out, [[round(float(x), 2), round(float(y), 2)] for x, y in corners], label_pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default=str(ML_DIR.parent / "fixtures" / "synthetic"))
    ap.add_argument("--grid", type=int, default=0, help="16 or 32; 0 = mostly 16 with ~15%% 32")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=2844)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        rng = np.random.default_rng(a.seed + i)
        g = a.grid or (32 if rng.random() < 0.15 else 16)
        img, corners, pieces = render_photo(rng, g, g, a.width, a.height)
        name = f"synth_{i:03d}"
        Image.fromarray(img).save(out / "images" / f"{name}.jpg", quality=92)
        label = {"image": f"{name}.jpg", "grid": {"cols": g, "rows": g}, "corners": corners, "pieces": pieces,
                 "notes": f"synthetic (ml/render_synthetic.py seed={a.seed + i}); corners in pixel-edge coords"}
        (out / "labels" / f"{name}.json").write_text(json.dumps(label, indent=1))
        print(name, g, len(pieces), flush=True)


if __name__ == "__main__":
    main()
