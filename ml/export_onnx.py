"""Export trained SeamNet to public/models/seam.onnx (+ seam.json) and verify with onnxruntime.

  python ml/export_onnx.py --ckpt ml/runs/seam.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from model import SeamNet

ML = Path(__file__).resolve().parent
ROOT = ML.parent


def export_and_verify(model, out, extra_x=None):
    """Export to ONNX (opset 17, dynamic batch) and check onnxruntime matches torch. Returns max |diff|."""
    model.eval()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(4, 3, 16, 32)
    torch.onnx.export(model, dummy, str(out), input_names=["input"], output_names=["logit"],
                      dynamic_axes={"input": {0: "N"}, "logit": {0: "N"}}, opset_version=17,
                      do_constant_folding=True, dynamo=False)
    m = onnx.load(str(out))
    onnx.checker.check_model(m)
    xs = [np.random.default_rng(0).random((1, 3, 16, 32), dtype=np.float32),
          np.random.default_rng(1).random((37, 3, 16, 32), dtype=np.float32)]
    if extra_x is not None:
        xs.append(np.ascontiguousarray(extra_x, dtype=np.float32))
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    maxdiff = 0.0
    with torch.no_grad():
        for x in xs:
            ref = model(torch.from_numpy(x)).numpy()
            got = sess.run(["logit"], {"input": x})[0]
            assert got.shape == (x.shape[0], 1), got.shape
            maxdiff = max(maxdiff, float(np.abs(ref - got).max()))
    print(f"{out.name}: opset {[o.version for o in m.opset_import]}, {out.stat().st_size} bytes, "
          f"onnxruntime vs torch max |diff| = {maxdiff:.2e}", "OK" if maxdiff < 1e-4 else "FAIL")
    assert maxdiff < 1e-4
    return maxdiff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ML / "runs" / "seam.pt"))
    ap.add_argument("--out", default=str(ROOT / "public" / "models" / "seam.onnx"))
    ap.add_argument("--data", default=str(ML / "data" / "seams.npz"), help="real crops used for the parity check")
    a = ap.parse_args()
    model = SeamNet()
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    extra = None
    if Path(a.data).exists():
        extra = np.load(a.data)["X"][:1000].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    maxdiff = export_and_verify(model, a.out, extra)
    out = Path(a.out)

    metrics_p = Path(a.ckpt).with_suffix(".metrics.json")
    metrics = json.loads(metrics_p.read_text()) if metrics_p.exists() else {}
    meta = {
        "input": {"name": "input", "shape": ["N", 3, 16, 32], "dtype": "float32", "layout": "NCHW RGB"},
        "output": {"name": "logit", "shape": ["N", 1], "meaning": "seam logit; sigmoid(logit) = P(seam)"},
        "crop": {"w": 32, "h": 16, "window": "2*pitch wide x 1*pitch tall centred on the gap, step = pitch/16",
                 "sample": "crop px (i,j) <- centre + ((i+0.5-16)*ex + (j+0.5-8)*ey) * pitch/16; image pixel k centre at k+0.5; bilinear, clamp",
                 "h_pair": "centre ((col+1)P, (row+0.5)P), ex=(1,0), ey=(0,1)",
                 "v_pair": "centre ((col+0.5)P, (row+1)P), ex=(0,1), ey=(-1,0) (upper stud on the left)"},
        "normalisation": "x/255",
        "threshold": 0.5,
        "val_acc": round(metrics.get("acc", float("nan")), 4),
        "same_colour_val_acc": round(metrics.get("acc_same", float("nan")), 4),
        "params": metrics.get("params"),
        "trained_on": "synthetic crops only (ml/dataset.py), warped pitch 40-90 px",
        "onnx_max_abs_diff_vs_torch": maxdiff,
    }
    (out.parent / "seam.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out.parent / "seam.json")


if __name__ == "__main__":
    main()
