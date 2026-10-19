# BrickCount ML: seam classifier

A tiny CNN decides, for each pair of neighbouring studs, whether a **seam** separates them (two
different pieces, label 1) or they belong to the **same piece** (label 0). The browser pipeline runs it
with onnxruntime-web (`public/models/seam.onnx`, metadata in `public/models/seam.json`).

Everything is trained on synthetic data. No real photos are used for training.

## Setup

    pip install -r ml/requirements.txt      # CPU torch is fine

## Files

| file | purpose |
|---|---|
| `palette.json` | palette ids/rgb and legal plate sizes (mirrors `src/lib/palette.ts`) |
| `render_synthetic.py` | random valid plate layouts + renderer: rectified mosaic, camera/glass effects, framed phone photos with `FixtureLabel` JSON |
| `dataset.py` | renders rectified mosaics and cuts 32x16 seam crops with the shared crop spec |
| `model.py` | `SeamNet`, about 58k params; `(x-0.5)*4` normalisation is built into the graph |
| `train.py` | training with BCE-with-logits; the val split is by mosaic; prints accuracy and confusion matrices |
| `export_onnx.py` | ONNX export (opset 17), onnxruntime parity check, writes `seam.json` |

## Regenerate / train / export

    python ml/render_synthetic.py --n 30 --out fixtures/synthetic       # eval photos + labels (about 15% are 32x32)
    python ml/dataset.py --mosaics 700 --per 300                       # about 200k crops -> ml/data/seams.npz (about 5 min on 4 cores)
    python ml/train.py --epochs 12                                     # -> ml/runs/seam.pt (+ .metrics.json)
    python ml/export_onnx.py                                           # -> public/models/seam.onnx + seam.json

`ml/data/` and `ml/runs/` are git-ignored.

## Crop spec (must match `src/pipeline/seams`)

Warped (rectified) image with stud pitch `P`. Image pixel `k` has its centre at `k + 0.5`.

* Horizontal pair `(col,row,'h')`: centre `c = ((col+1)P, (row+0.5)P)`, `ex = (1,0)`, `ey = (0,1)`.
* Vertical pair `(col,row,'v')`: centre `c = ((col+0.5)P, (row+1)P)`, `ex = (0,1)`, `ey = (-1,0)`.
  The crop is rotated 90°, so the upper stud lands on the left.
* Crop pixel `(i,j)`, with `i` in 0..31 and `j` in 0..15, is sampled bilinearly at
  `c + ((i+0.5-16)·ex + (j+0.5-8)·ey) · P/16`. Coordinates are clamped at the image edge.
* The tensor is float32 NCHW RGB with values `pixel/255`. Input name `input` `[N,3,16,32]`, output `logit` `[N,1]`.
  A positive logit means seam.

The model is trained with horizontal and vertical flips, so the exact orientation of `v` pairs does not matter.

## Synthetic data, briefly

* **Layouts:** a colour map is built first: a background (often medium azure, black or blue), optional
  ground, band or sun regions, and pixel-art sprites with outlines. Each same-colour region is then tiled
  greedily with legal plates, in both orientations and with a random scan direction. Size-preference
  styles are "long" (1x4, 1x6, 2x4), mixed, big and small. This produces many same-colour neighbours.
* **Rendering:** each piece gets a slightly different brightness. Each piece side gets its own dark seam
  line (about 0.6–2% of the pitch as half-width, darkness 5–85%), an optional gap band, and a bevel
  highlight or shadow that depends on the light direction. Studs come from per-cell templates with a
  cast shadow, rim shading, a dark outline, a specular arc and embossed "LEGO" text in 4 orientations.
* **Camera and glass:** lighting gradient, vignette, white balance, haze (washed-out colours), specular
  glare ovals, scratches, sheen, gaussian or motion blur, noise and JPEG.
* **Photos:** portrait 1600x2844. The mosaic is 40–75% of the width, inside a white mat with an opening
  bevel that sometimes overlaps the outer studs. A black frame and dark background surround it.
  Perspective tilt is up to 25°, roll up to ±6°, and glare falls on both the mosaic and the mat.
* **Training crops:** the mosaic is rendered at a "photo" pitch (`P_w`×1–2, 40–130 px). Effects and JPEG
  are applied, then it is resized to a warped pitch `P_w` of 40–90 px (the pipeline uses about 64),
  using bilinear or area resampling. 30% get a pre-blur. Crops are taken with jitter: pitch ±10%,
  offset ±15%, rotation ±4°. The mix per mosaic is 15% different-colour pairs, 42.5% same-colour seams
  and 42.5% same-colour non-seams.

## Notes for the pipeline

* At pitch 64 the crop samples every 4 px with plain bilinear interpolation, so a 1–2 px seam can be
  partly skipped. The model is trained on both the raw and the pre-blurred variants. If accuracy on real
  photos is poor, try a light blur (σ ≈ 0.5·P/16) on the warped image before cropping, or a pitch near 32.
* Different-colour pairs are almost always seams. The model's value is in same-colour pairs.
