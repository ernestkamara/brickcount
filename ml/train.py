"""Train SeamNet on ml/data/seams.npz (from dataset.py).

  python ml/train.py --epochs 12 --out ml/runs/seam.pt

Validation split is by mosaic (10% of mosaics held out), reported overall and on same-colour pairs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import SeamNet, param_count

ML = Path(__file__).resolve().parent


def augment(x: torch.Tensor) -> torch.Tensor:
    """x: [B,3,16,32] float 0..1. Photometric + flip augmentations that keep the crop geometry."""
    B = x.shape[0]
    flip_w = torch.rand(B) < 0.5
    x = torch.where(flip_w[:, None, None, None], x.flip(3), x)
    flip_h = torch.rand(B) < 0.5
    x = torch.where(flip_h[:, None, None, None], x.flip(2), x)
    perm = torch.rand(B) < 0.3  # channel permutation -> colour invariance
    if perm.any():
        p = torch.randperm(3)
        x[perm] = x[perm][:, p]
    gray = torch.rand(B) < 0.1
    if gray.any():
        x[gray] = x[gray].mean(1, keepdim=True).expand(-1, 3, -1, -1)
    gain = torch.empty(B, 1, 1, 1).uniform_(0.7, 1.3)
    cgain = torch.empty(B, 3, 1, 1).uniform_(0.92, 1.08)
    mean = x.mean((1, 2, 3), keepdim=True)
    contrast = torch.empty(B, 1, 1, 1).uniform_(0.6, 1.3)
    x = (x - mean) * contrast + mean
    x = x * gain * cgain
    haze = torch.empty(B, 1, 1, 1).uniform_(0, 0.2) * (torch.rand(B, 1, 1, 1) < 0.4)
    x = x * (1 - haze) + haze * 0.85
    x = x + torch.randn_like(x) * torch.empty(B, 1, 1, 1).uniform_(0, 0.03)
    return x.clamp(0, 1)


def evaluate(model, X, y, same, bs=4096):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i + bs].permute(0, 3, 1, 2).float() / 255.0
            preds.append(model(xb)[:, 0] > 0)
    p = torch.cat(preds).numpy().astype(np.uint8)
    yy = y.numpy()
    sm = same.numpy().astype(bool)
    acc = float((p == yy).mean())
    acc_same = float((p[sm] == yy[sm]).mean())
    acc_diff = float((p[~sm] == yy[~sm]).mean()) if (~sm).any() else float("nan")
    cm = [[int(((yy == a) & (p == b)).sum()) for b in (0, 1)] for a in (0, 1)]
    cm_same = [[int(((yy == a) & (p == b) & sm).sum()) for b in (0, 1)] for a in (0, 1)]
    return dict(acc=acc, acc_same=acc_same, acc_diff=acc_diff, cm=cm, cm_same=cm_same)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ML / "data" / "seams.npz"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(ML / "runs" / "seam.pt"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    d = np.load(a.data)
    X, y, same, mosaic = d["X"], d["y"], d["same"], d["mosaic"]
    ids = np.unique(mosaic)
    rng = np.random.default_rng(0)
    val_ids = rng.choice(ids, size=max(1, len(ids) // 10), replace=False)
    is_val = np.isin(mosaic, val_ids)
    Xt, yt = torch.from_numpy(X[~is_val]), torch.from_numpy(y[~is_val]).float()
    Xv, yv, sv = torch.from_numpy(X[is_val]), torch.from_numpy(y[is_val]), torch.from_numpy(same[is_val])
    print(f"train {len(yt)}  val {len(yv)} (mosaics {len(ids) - len(val_ids)}/{len(val_ids)})")

    model = SeamNet()
    print("params", param_count(model))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(yt) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    t0 = time.time()
    best = None
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(yt))
        tot, n = 0.0, 0
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            idx = perm[i:i + a.bs]
            xb = augment(Xt[idx].permute(0, 3, 1, 2).float() / 255.0)
            loss = F.binary_cross_entropy_with_logits(model(xb)[:, 0], yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx); n += len(idx)
        r = evaluate(model, Xv, yv, sv)
        print(f"ep {ep + 1:2d} loss {tot / n:.4f} val acc {r['acc']:.4f} same-colour {r['acc_same']:.4f} "
              f"diff-colour {r['acc_diff']:.4f} [{time.time() - t0:.0f}s]", flush=True)
        best = r
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.out)
    best["params"] = param_count(model)
    best["train_n"] = int(len(yt)); best["val_n"] = int(len(yv))
    Path(a.out).with_suffix(".metrics.json").write_text(json.dumps(best, indent=1))
    print("confusion (rows=true 0/1, cols=pred 0/1):", best["cm"])
    print("same-colour confusion:", best["cm_same"])
    print("saved", a.out)


if __name__ == "__main__":
    main()
