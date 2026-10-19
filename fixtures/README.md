# Fixtures
- `images/` real photos (golden set). `labels/<stem>.json` matches `FixtureLabel` in src/pipeline/types.ts.
- `synthetic/` rendered photos + labels from ml/render_synthetic.py (same label format).
- Pieces: axis-aligned stud rectangles, `x,y` = top-left stud (col,row), 0-based from the mosaic's top-left.
