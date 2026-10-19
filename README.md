# BrickCount

Count the LEGO **pieces** (not studs) in a photo of a flat, framed LEGO mosaic. BrickCount is an installable PWA for phone and desktop. All image processing runs on the device in a Web Worker, so photos never leave it unless you turn on the optional cloud check.

## How it works

1. **Find the mosaic.** Detect the mosaic quad inside the white mat and perspective-warp it to a rectified grid (OpenCV.js).
2. **Fit the stud grid.** Estimate stud pitch and count the visible columns and rows. If the fit is doubtful, ask the user to adjust corners or retake.
3. **Read colours.** Take each stud cell's median colour in Lab space, snap it to the LEGO palette, and mask glare.
4. **Classify seams.** For every pair of neighbouring studs, decide *seam* or *same piece*. A tiny CNN (ONNX Runtime Web) does this by default, with a colour + dark-gap heuristic as a fallback.
5. **Build pieces.** Union-find studs into pieces. Split any shape that isn't a real plate size (e.g. an L-shape) and flag it as low confidence.
6. **Review.** The review overlay shows numbered pieces. Tap a seam to split or merge, and the count updates live. Amber means low confidence.

## Quick start

Requires Node 22. Python 3.13 is needed only for `ml/`.

```bash
npm install
npm run dev          # http://localhost:5173  (add ?mock=1 for a fake worker)
npm test             # unit tests (vitest)
npm run e2e          # Playwright: production build + real and mock worker
npm run eval -- --set real --classifier cnn   # accuracy on the labelled photos
npm run build        # production build in dist/
```

## Repository layout

| Path | What |
| --- | --- |
| `src/pipeline/` | Image pipeline (pure TS + OpenCV.js): preprocess, grid, colour, glare, seams, union-find, validation |
| `src/worker/` | Comlink Web Worker wrapping the pipeline |
| `src/ui/`, `src/lib/` | React UI, store, settings, export |
| `ml/` | Synthetic mosaic renderer, seam CNN training and ONNX export |
| `public/models/` | Exported `seam.onnx` + `seam.json` (crop spec) |
| `fixtures/` | Labelled photos (`images/`, `closeup/`) and synthetic eval set |
| `scripts/` | `eval.ts` (accuracy report), `label-draft.ts` (pre-fill labels) |
| `functions/` | Optional Firebase Function: vision-LLM second opinion (off by default) |

## Status

This is early and accuracy is below target. Run `npm run eval` for the current numbers. Known gaps:

- Exact count matches on only a minority of photos.
- Grid fitting does not yet generalise across framings.
- Seams hidden by glass glare are often missed.
- The fixture labels are hand-made and may contain mistakes.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to help.
