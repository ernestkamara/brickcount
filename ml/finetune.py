"""Fine-tune the synthetic seam CNN on real crops exported by scripts/export-seam-crops.ts.

  npx tsx scripts/export-seam-crops.ts                    # -> ml/data/real/
  python ml/finetune.py --cv                              # 4 mosaic-grouped folds -> ml/runs/fold{k}/seam.onnx
  python ml/finetune.py --final --export                  # all 16 photos -> public/models/seam.onnx + seam.json

Training mixes real and synthetic crops 50/50 per batch, so the model does not forget the synthetic data.
Real crops are the exact TypeScript crops plus jittered re-crops (pitch +-10%, offset +-10%, rotation +-3 deg)
cut from the same pre-blurred rectified image. Same-colour real pairs are sampled 3x more often. Pairs
touching a cell listed in ml/real_uncertain.json get loss weight 0.3. The architecture and ONNX I/O are unchanged.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import extract_crop
from model import SeamNet
from train import augment

ML = Path(__file__).resolve().parent
ROOT = ML.parent
MOSAICS = ["01_mario", "02_qblock", "03_flower", "04_mushroom", "05_yoshi", "06_star", "07_piranha", "08_goomba"]
FOLDS = [["01_mario", "05_yoshi"], ["02_qblock", "06_star"], ["03_flower", "07_piranha"], ["04_mushroom", "08_goomba"]]
UNCERTAIN_W = 0.3


def load_real(d: Path):
    unc = json.loads((ML / "real_uncertain.json").read_text())
    photos = []
    for j in sorted(d.glob("*.json")):
        if j.name == "summary.json":
            continue
        m = json.loads(j.read_text())
        base = str(j)[:-5]
        X = np.fromfile(base + ".crops.f32", np.float32).reshape(-1, 3, 16, 32)
        img = np.asarray(Image.open(base + ".warped.png").convert("RGB"), dtype=np.float32)
        pr = m["pairs"]
        ucells = {tuple(c) for c in unc.get(m["mosaic"], [])}
        a = lambda k: np.array([p[k] for p in pr])
        lc, lr, dirs = a("lcol"), a("lrow"), np.array([0 if p["dir"] == "h" else 1 for p in pr])
        lc2, lr2 = lc + (dirs == 0), lr + (dirs == 1)
        w = np.array([UNCERTAIN_W if ((x1, y1) in ucells or (x2, y2) in ucells) else 1.0
                      for x1, y1, x2, y2 in zip(lc, lr, lc2, lr2)], np.float32)
        photos.append(dict(key=f"{m['set']}/{m['stem']}", mosaic=m["mosaic"], X=X, img=img,
                           P=m["width"] / m["cols"], Ph=m["height"] / m["rows"],
                           col=a("col"), row=a("row"), dir=dirs, y=a("seam").astype(np.float32),
                           same=a("sameColour").astype(bool), glare=a("glare"), w=w))
    return photos


def build_real_train(photos, n_jit, rng):
    Xs, ys, ws, ss = [], [], [], []
    for ph in photos:
        Xs.append(ph["X"].astype(np.float16)); ys.append(ph["y"]); ws.append(ph["w"]); ss.append(ph["same"])
        for _ in range(n_jit):
            Xj = np.empty_like(ph["X"], dtype=np.float16)
            for k in range(len(ph["y"])):
                c = extract_crop(ph["img"], ph["col"][k], ph["row"][k], ph["dir"][k], ph["P"], rng, True,
                                 Ph=ph["Ph"], jit=(0.10, 0.10, 3.0))
                Xj[k] = (c / 255.0).transpose(2, 0, 1)
            Xs.append(Xj); ys.append(ph["y"]); ws.append(ph["w"]); ss.append(ph["same"])
    return (torch.from_numpy(np.concatenate(Xs)), torch.from_numpy(np.concatenate(ys)),
            torch.from_numpy(np.concatenate(ws)), np.concatenate(ss))


def pair_metrics(model, photos):
    """Exact (un-jittered) TS crops. Accuracy overall / same-colour, seam P/R on same-colour pairs."""
    model.eval()
    tot = dict(n=0, ok=0, sn=0, sok=0, tp=0, fp=0, fn=0)
    with torch.no_grad():
        for ph in photos:
            p = (model(torch.from_numpy(ph["X"]))[:, 0] > 0).numpy()
            y = ph["y"].astype(bool); s = ph["same"]
            tot["n"] += len(y); tot["ok"] += int((p == y).sum())
            tot["sn"] += int(s.sum()); tot["sok"] += int((p == y)[s].sum())
            tot["tp"] += int((p & y & s).sum()); tot["fp"] += int((p & ~y & s).sum()); tot["fn"] += int((~p & y & s).sum())
    return dict(acc=tot["ok"] / max(1, tot["n"]), acc_same=tot["sok"] / max(1, tot["sn"]),
                same_seam_precision=tot["tp"] / max(1, tot["tp"] + tot["fp"]),
                same_seam_recall=tot["tp"] / max(1, tot["tp"] + tot["fn"]), **tot)


def synth_val(model, X, y, same):
    model.eval()
    with torch.no_grad():
        p = torch.cat([model(X[i:i + 4096].permute(0, 3, 1, 2).float() / 255.0)[:, 0] > 0 for i in range(0, len(X), 4096)]).numpy()
    yy = y.numpy().astype(bool); s = same.numpy().astype(bool)
    return dict(acc=float((p == yy).mean()), acc_same=float((p == yy)[s].mean()))


def finetune(train_photos, synth, args, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xr, yr, wr, sr = build_real_train(train_photos, args.jit, rng)
    pr = np.where(sr, 3.0, 1.0); pr = torch.from_numpy(pr / pr.sum())
    Xs, ys = synth
    model = SeamNet()
    model.load_state_dict(torch.load(args.init, map_location="cpu"))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    half = args.bs // 2
    t0 = time.time()
    for step in range(args.steps):
        model.train()
        ir = torch.multinomial(pr, half, replacement=True)
        isy = torch.randint(0, len(ys), (args.bs - half,))
        xb = torch.cat([Xr[ir].float(), Xs[isy].permute(0, 3, 1, 2).float() / 255.0])
        yb = torch.cat([yr[ir], ys[isy].float()])
        wb = torch.cat([wr[ir] * args.real_w, torch.ones(args.bs - half)])
        loss = (F.binary_cross_entropy_with_logits(model(augment(xb))[:, 0], yb, reduction="none") * wb).sum() / wb.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 500 == 0:
            print(f"  step {step + 1}/{args.steps} loss {loss.item():.4f} [{time.time() - t0:.0f}s]", flush=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=str(ML / "data" / "real"))
    ap.add_argument("--synth", default=str(ML / "data" / "seams.npz"))
    ap.add_argument("--init", default=str(ML / "runs" / "seam.pt"))
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--export", action="store_true", help="with --final: write public/models/seam.onnx + seam.json")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--jit", type=int, default=8, help="jittered re-crops per real pair")
    ap.add_argument("--real_w", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from export_onnx import export_and_verify

    photos = load_real(Path(a.real))
    print(f"real photos: {len(photos)}; pairs {sum(len(p['y']) for p in photos)}, "
          f"seam {int(sum(p['y'].sum() for p in photos))}, same-colour {int(sum(p['same'].sum() for p in photos))}, "
          f"same-colour seam {int(sum((p['y'].astype(bool) & p['same']).sum() for p in photos))}, "
          f"down-weighted {int(sum((p['w'] < 1).sum() for p in photos))}")
    d = np.load(a.synth)
    mosaic = d["mosaic"]
    ids = np.unique(mosaic)
    val_ids = np.random.default_rng(0).choice(ids, size=max(1, len(ids) // 10), replace=False)  # same split as train.py
    is_val = np.isin(mosaic, val_ids)
    synth_tr = (torch.from_numpy(d["X"][~is_val]), torch.from_numpy(d["y"][~is_val]))
    synth_va = (torch.from_numpy(d["X"][is_val]), torch.from_numpy(d["y"][is_val]), torch.from_numpy(d["same"][is_val]))

    base = SeamNet(); base.load_state_dict(torch.load(a.init, map_location="cpu")); base.eval()
    results = {}
    if a.cv:
        oof_base, oof_ft = [], []
        for k, hold in enumerate(FOLDS):
            tr = [p for p in photos if p["mosaic"] not in hold]
            te = [p for p in photos if p["mosaic"] in hold]
            print(f"fold {k}: hold out {hold} ({len(te)} photos), train on {len(tr)} photos")
            m = finetune(tr, synth_tr, a, seed=k)
            rb, rf = pair_metrics(base, te), pair_metrics(m, te)
            print(f"  base: acc {rb['acc']:.4f} same {rb['acc_same']:.4f} sameSeam P/R {rb['same_seam_precision']:.3f}/{rb['same_seam_recall']:.3f}")
            print(f"  ft  : acc {rf['acc']:.4f} same {rf['acc_same']:.4f} sameSeam P/R {rf['same_seam_precision']:.3f}/{rf['same_seam_recall']:.3f}")
            out = ML / "runs" / f"fold{k}"
            out.mkdir(parents=True, exist_ok=True)
            torch.save(m.state_dict(), out / "seam.pt")
            export_and_verify(m, out / "seam.onnx", te[0]["X"][:500])
            results[f"fold{k}"] = dict(holdout=hold, base=rb, finetuned=rf, synth_val=synth_val(m, *synth_va))
            oof_base.append(rb); oof_ft.append(rf)

        def pool(rs):
            s = {k: sum(r[k] for r in rs) for k in ("n", "ok", "sn", "sok", "tp", "fp", "fn")}
            return dict(acc=s["ok"] / s["n"], acc_same=s["sok"] / s["sn"], same_seam_precision=s["tp"] / (s["tp"] + s["fp"]),
                        same_seam_recall=s["tp"] / (s["tp"] + s["fn"]), **s)
        results["oof_base"], results["oof_finetuned"] = pool(oof_base), pool(oof_ft)
        print("OOF base     :", json.dumps({k: round(v, 4) for k, v in results["oof_base"].items()}))
        print("OOF finetuned:", json.dumps({k: round(v, 4) for k, v in results["oof_finetuned"].items()}))
        (ML / "runs" / "cv_results.json").write_text(json.dumps(results, indent=1))
    if a.final:
        m = finetune(photos, synth_tr, a, seed=100)
        out = ML / "runs" / "final"
        out.mkdir(parents=True, exist_ok=True)
        torch.save(m.state_dict(), out / "seam.pt")
        sv = synth_val(m, *synth_va)
        rtrain = pair_metrics(m, photos)
        print(f"final: synthetic val acc {sv['acc']:.4f} same {sv['acc_same']:.4f}; real (train) acc {rtrain['acc']:.4f}")
        target = (ROOT / "public" / "models" / "seam.onnx") if a.export else (out / "seam.onnx")
        maxdiff = export_and_verify(m, target, photos[0]["X"][:500])
        cvp = ML / "runs" / "cv_results.json"
        cv = json.loads(cvp.read_text()) if cvp.exists() else {}
        oof = cv.get("oof_finetuned", {})
        meta = json.loads((ROOT / "public" / "models" / "seam.json").read_text())
        meta.update({
            "val_acc": round(sv["acc"], 4),
            "same_colour_val_acc": round(sv["acc_same"], 4),
            "val_note": "val_acc/same_colour_val_acc = held-out synthetic mosaics; real_oof_* = 4-fold mosaic-grouped CV on exact pipeline crops of the 16 labelled photos",
            "real_oof_pair_acc": round(oof.get("acc", float("nan")), 4),
            "real_oof_same_colour_acc": round(oof.get("acc_same", float("nan")), 4),
            "real_oof_same_colour_seam_precision": round(oof.get("same_seam_precision", float("nan")), 4),
            "real_oof_same_colour_seam_recall": round(oof.get("same_seam_recall", float("nan")), 4),
            "trained_on": ("synthetic crops (ml/dataset.py) then fine-tuned (ml/finetune.py) on "
                           f"{len(photos)} labelled real photos (fixtures real + closeup, {sum(len(p['y']) for p in photos)} pairs, "
                           f"x{a.jit + 1} with jittered re-crops) mixed 50/50 with synthetic; pipeline pre-blur 0.5 crop px"),
            "onnx_max_abs_diff_vs_torch": maxdiff,
        })
        if a.export:
            (ROOT / "public" / "models" / "seam.json").write_text(json.dumps(meta, indent=1))
            print("wrote public/models/seam.onnx + seam.json")
        else:
            (out / "seam.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
