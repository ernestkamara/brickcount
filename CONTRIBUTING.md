# Contributing to BrickCount

Thanks for helping. The most valuable contributions right now are **better accuracy** and **more labelled photos**.

## Getting set up

```bash
git clone <repo-url> && cd brickcount
npm install
npm run dev            # open http://localhost:5173/?mock=1 to work on UI without the CV worker
```

The ML tools are optional (Python 3.13):

```bash
python3 -m venv ml/.venv && . ml/.venv/bin/activate
pip install -r ml/requirements.txt
```

## Before you open a pull request

Run all of these and make sure they pass:

```bash
npx tsc -b             # type-check
npm run lint           # oxlint
npm test               # unit tests
npm run e2e            # end-to-end (builds the app; uses Chromium)
```

If you touch anything in `src/pipeline/` or `public/models/`, also run the eval on every set. Put the before/after numbers in the PR description:

```bash
npm run eval -- --set real     --classifier cnn
npm run eval -- --set closeup  --classifier cnn
npm run eval -- --set synthetic --classifier cnn
```

A pipeline change should not make any set worse. Tuning that helps one set but hurts another is likely overfitting.

## Ways to help

### Add labelled photos

New photos of flat LEGO mosaics, especially in different framings, lighting and grid sizes, are the best way to improve accuracy.

1. Put the photo in `fixtures/images/` (or a new set folder `fixtures/<set>/images/`).
2. Pre-fill a label with `npx tsx scripts/label-draft.ts <photo>`.
3. Correct it by hand. Labels follow `FixtureLabel` in `src/pipeline/types.ts`:
   - `grid` counts the studs whose centres are visible.
   - `x`/`y` are 0-based from the top-left visible stud.
   - Each piece is a rectangle with a real plate size and a palette colour id from `src/lib/palette.ts`.
4. Add a `notes` field for anything you weren't sure about.

Only submit photos you took or have the right to share. Strip location metadata first, e.g. `exiftool -gps:all= photo.jpg`.

### Improve the pipeline

The contracts in `src/pipeline/types.ts`, `src/pipeline/seams/index.ts` and `src/worker/api.ts` are shared by the pipeline, worker, UI, eval and ML code. Changing one means updating every consumer in the same PR.

- When unsure, the pipeline must return `grid_failed`. Never show a confident wrong count.
- Free OpenCV `Mat`s (`.delete()`).
- Keep per-photo time within budget: the target is ≤ 1.5 s on a mid-range phone.

### Retrain the seam model

See `ml/README.md`. Keep the crop spec in `public/models/seam.json` in sync with `src/pipeline/seams/crop.ts`. Report validation accuracy and real-photo eval numbers.

## Security and secrets

- **Never commit credentials.** That covers API keys, `.env` files, Firebase service-account JSON and tokens. `.gitignore` already blocks the common ones.
- The optional cloud check reads `ANTHROPIC_API_KEY` from Firebase Secret Manager (`firebase functions:secrets:set ANTHROPIC_API_KEY`). It is never stored in the repo or shipped to the browser.
- If you find a security issue, open a private security advisory on the repository rather than a public issue.

## Code style

TypeScript strict mode. Match the style of the code around your change, and keep comments for the *why*. Use small, focused commits with descriptive messages.
