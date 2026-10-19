"""Out-of-fold full-pipeline eval: each held-out photo is scored with its fold's model (ml/runs/fold{k}/seam.onnx).

  python ml/eval_folds.py                 # needs ml/finetune.py --cv first; compares with public/models/seam.onnx
  python ml/eval_folds.py --baseline ml/runs/seam-synthetic.onnx
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from finetune import FOLDS

ML = Path(__file__).resolve().parent
ROOT = ML.parent
OUT = ML / "runs" / "eval"


EVAL_ROOT = ROOT


def run_eval(set_, only, model, tag):
    out = OUT / f"{tag}_{set_}.json"
    cmd = ["npx", "tsx", "scripts/eval.ts", "--set", set_, "--classifier", "cnn", "--model", str(Path(model).resolve()), "--out", str(out)]
    if only:
        cmd += ["--only", ",".join(only)]
    subprocess.run(cmd, cwd=EVAL_ROOT, check=True, stdout=subprocess.DEVNULL)
    return json.loads(out.read_text())["photos"]


def pool(photos):
    lab = [p for p in photos if p.get("truePieces") is not None]
    m = sum(p.get("matched", 0) for p in lab); pr = sum(p["predPieces"] for p in lab); t = sum(p["truePieces"] for p in lab)
    sc = {k: sum((p.get("sameColourSeams") or {}).get(k, 0) for p in lab) for k in ("tp", "fp", "fn")}
    return dict(
        photos=len(lab),
        meanAbsPctCountError=sum(abs(p["predPieces"] - p["truePieces"]) / p["truePieces"] * 100 for p in lab) / len(lab),
        exactCountRate=sum(p["predPieces"] == p["truePieces"] for p in lab) / len(lab),
        piecePrecision=m / pr, pieceRecall=m / t,
        sameColourSeamPrecision=sc["tp"] / max(1, sc["tp"] + sc["fp"]), sameColourSeamRecall=sc["tp"] / max(1, sc["tp"] + sc["fn"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(ROOT / "public" / "models" / "seam.onnx"))
    ap.add_argument("--root", default=str(ROOT), help="repo checkout to run the pipeline from (e.g. a frozen snapshot)")
    a = ap.parse_args()
    global EVAL_ROOT
    EVAL_ROOT = Path(a.root).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for set_ in ("real", "closeup"):
        base = run_eval(set_, None, a.baseline, "baseline")
        oof = []
        for k, hold in enumerate(FOLDS):
            oof += run_eval(set_, hold, ML / "runs" / f"fold{k}" / "seam.onnx", f"fold{k}")
        res[set_] = dict(baseline=pool(base), oof_finetuned=pool(oof),
                         per_photo={p["image"]: dict(true=p.get("truePieces"), base=b["predPieces"], ft=p["predPieces"])
                                    for p in oof for b in base if b["image"] == p["image"]})
    res["both"] = dict(baseline=pool(json.loads((OUT / "baseline_real.json").read_text())["photos"] + json.loads((OUT / "baseline_closeup.json").read_text())["photos"]),
                       oof_finetuned=pool(sum((json.loads((OUT / f"fold{k}_{s}.json").read_text())["photos"] for k in range(4) for s in ("real", "closeup")), [])))
    allb = {s: json.loads((OUT / f"baseline_{s}.json").read_text())["photos"] for s in ("real", "closeup")}
    for k, hold in enumerate(FOLDS):
        imgs = {h + ".jpg" for h in hold}
        fb = [p for s in ("real", "closeup") for p in allb[s] if p["image"] in imgs]
        ff = sum((json.loads((OUT / f"fold{k}_{s}.json").read_text())["photos"] for s in ("real", "closeup")), [])
        res[f"fold{k}"] = dict(holdout=hold, baseline=pool(fb), oof_finetuned=pool(ff))
    (ML / "runs" / "cv_pipeline_results.json").write_text(json.dumps(res, indent=1))
    for k in ("fold0", "fold1", "fold2", "fold3", "real", "closeup", "both"):
        for which in ("baseline", "oof_finetuned"):
            print(f"{k:8s} {which:14s}", {m: round(v, 3) for m, v in res[k][which].items()})
    for s in ("real", "closeup"):
        print(s, res[s]["per_photo"])


if __name__ == "__main__":
    main()
