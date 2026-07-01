"""Sharpened spatial-JEPA metric: how ACCURATELY can we predict unobserved panorama regions?

Two cleaner readouts than raw cosine (which sits on DINOv3's ~0.68 anisotropy floor):
  (1) CENTERED cosine — subtract the per-pano feature mean -> removes the floor, shows true margin.
  (2) SEG decode — train a linear seg head on TRUE latent, then decode the PREDICTED latent of masked
      cells -> "% of unobserved cells whose CLASS we predict correctly from partial observation",
      vs nearest-fill / mean-fill / true-latent (oracle), broken down by distance to observed.

Reuses the validated gate predictor (FieldUNet) + masking. Run:
  CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pano python scripts/spatial_jepa_metric.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import anyres_e2p as a2p  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
from spatial_jepa_gate import FieldUNet, random_block, nearest_fill, dist_to_observed, cos_np  # noqa: E402
from viz_merged_field import build_field, cell_ids, HF, WF, TILE, PATCH, ERP_W, ERP_H  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = "cuda"
SEED = 0
MODEL = os.environ.get("MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m")
EPOCHS = int(os.environ.get("EPOCHS", 40))
TR = int(os.environ.get("TR", 120))
VA = int(os.environ.get("VA", 30))
BINS = [(0, 2), (2, 4), (4, 7), (7, 100)]


def cache(enc, files, plan, ids):
    out = []
    for f in files:
        erp = np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H), Image.BILINEAR))
        fs, cov = build_field(enc, erp, plan, ids)
        _, lab = P.load_rgb_label(f)
        seg = P.label_to_grid(lab, HF, WF).astype(np.int64)
        out.append((fs.reshape(HF, WF, enc.dim).astype(np.float32), cov.reshape(HF, WF), seg))
    return out


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    P.configure("stanford2d3d"); P.TILE = TILE
    enc = PanoEncoder(model_id=MODEL, lora_rank=0).to(DEVICE).eval(); P.enc_patch = enc.patch
    D = enc.dim
    plan = a2p.plan_tiles("full_sphere", 65.0, 65.0, 0.25)
    ids = cell_ids(plan, TILE // PATCH)
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR]
    va = [f for f in files if "5" in area(f)][:VA]
    print(f"spatial-JEPA metric | enc={MODEL.split('/')[-1]} D={D} field={HF}×{WF} tr={len(tr)} va={len(va)} ep={EPOCHS}", flush=True)
    ctr = cache(enc, tr, plan, ids); cva = cache(enc, va, plan, ids)
    scale = float(np.std(ctr[0][0][ctr[0][1]])) + 1e-6

    # ---- seg linear head on TRUE latent (the "reader" that turns latent -> class)
    Xtr = np.concatenate([fs[cov] for fs, cov, _ in ctr]); ytr = np.concatenate([sg[cov] for _, cov, sg in ctr])
    keep = ytr != P.IGNORE; Xtr, ytr = Xtr[keep], ytr[keep]
    sel = np.random.RandomState(SEED).permutation(len(Xtr))[:300000]
    Xt = torch.from_numpy(Xtr[sel]).to(DEVICE); yt = torch.from_numpy(ytr[sel]).to(DEVICE)
    head = nn.Linear(D, P.N_CLASS).to(DEVICE); ho = torch.optim.Adam(head.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(600):
        ho.zero_grad(); F.cross_entropy(head(Xt), yt, ignore_index=P.IGNORE).backward(); ho.step()

    # ---- predictor (same as gate)
    net = FieldUNet(D).to(DEVICE); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(SEED)

    def to_in(field, m):
        x = torch.from_numpy(field).permute(2, 0, 1)[None].to(DEVICE) / scale
        mt = torch.from_numpy(m.astype(np.float32))[None, None].to(DEVICE)
        return x * (1 - mt), mt

    for _ in range(EPOCHS):
        net.train()
        for i in torch.randperm(len(ctr), generator=g).tolist():
            field, cov, _ = ctr[i]; m = random_block(g); ev = m & cov
            if ev.sum() < 5:
                continue
            xin, mt = to_in(field, m)
            pred = net(xin, mt)[0].permute(1, 2, 0) * scale
            tgt = torch.from_numpy(field).to(DEVICE); s = torch.from_numpy(ev).to(DEVICE)
            p = F.normalize(pred[s], dim=1); t = F.normalize(tgt[s], dim=1)
            (1 - (p * t).sum(1)).mean().add(0.1 * F.mse_loss(pred[s] / scale, tgt[s] / scale)).backward()
            opt.step(); opt.zero_grad()

    # ---- eval: centered cosine + seg pixel-acc, per distance bin
    gv = torch.Generator().manual_seed(123); vmasks = [random_block(gv) for _ in cva]
    cosb = {k: {b: [] for b in BINS} for k in ("mean", "near", "pred")}
    acc = {k: {b: [0, 0] for b in BINS} for k in ("true", "mean", "near", "pred")}

    @torch.no_grad()
    def decode(arr):                                   # (N,D)->class
        return head(torch.from_numpy(arr).float().to(DEVICE)).argmax(1).cpu().numpy()

    net.eval()
    with torch.no_grad():
        for (field, cov, seg), m in zip(cva, vmasks):
            ev = m & cov; valid = (~m) & cov
            if ev.sum() < 5:
                continue
            xin, mt = to_in(field, m)
            pr = (net(xin, mt)[0].permute(1, 2, 0) * scale).cpu().numpy()
            nf = nearest_fill(field, valid)
            mu = field[valid].mean(0)
            d = dist_to_observed(m); de = d[ev]; true = field[ev]; ge = seg[ev]
            lat = {"true": true, "pred": pr[ev], "near": nf[ev], "mean": np.broadcast_to(mu, true.shape).copy()}
            cc = {k: cos_np(lat[k] - mu[None], true - mu[None]) for k in ("mean", "near", "pred")}
            cls = {k: decode(lat[k]) for k in lat}
            segok = ge != P.IGNORE
            for lo, hi in BINS:
                bm = (de >= lo) & (de < hi)
                for k in ("mean", "near", "pred"):
                    if bm.any():
                        cosb[k][(lo, hi)].append(cc[k][bm])
                bs = bm & segok
                if bs.any():
                    for k in lat:
                        acc[k][(lo, hi)][0] += int((cls[k][bs] == ge[bs]).sum()); acc[k][(lo, hi)][1] += int(bs.sum())

    def cm(lst): return float(np.concatenate(lst).mean()) if lst else float("nan")
    def am(k, b): n = acc[k][b][1]; return acc[k][b][0] / n if n else float("nan")
    def overall_cos(k): return cm([x for b in BINS for x in cosb[k][b]])
    def overall_acc(k):
        c = sum(acc[k][b][0] for b in BINS); n = sum(acc[k][b][1] for b in BINS); return c / n if n else float("nan")

    hdr = " ".join(f"d∈[{lo},{hi})".rjust(10) for lo, hi in BINS)
    print("\n(1) CENTERED cosine(pred,true) — anisotropy floor removed:")
    print(f"{'method':6} {'overall':>8} {hdr}")
    for k in ("mean", "near", "pred"):
        print(f"{k:6} {overall_cos(k):8.3f} " + " ".join(f"{cm(cosb[k][b]):10.3f}" for b in BINS), flush=True)

    print("\n(2) SEG pixel-acc of decoded latent — % of UNOBSERVED cells correctly classified:")
    print(f"{'method':6} {'overall':>8} {hdr}")
    for k in ("true", "mean", "near", "pred"):
        tag = "  (oracle=observed latent)" if k == "true" else ""
        print(f"{k:6} {overall_acc(k):8.3f} " + " ".join(f"{am(k, b):10.3f}" for b in BINS) + tag, flush=True)

    pf, nf2, tf = overall_acc("pred"), overall_acc("near"), overall_acc("true")
    print(f"\n=== predict unobserved seg: pred {pf:.3f} vs nearest {nf2:.3f} (oracle {tf:.3f}) "
          f"Δ={pf-nf2:+.3f} | far-bin pred {am('pred', BINS[-1]):.3f} vs near {am('near', BINS[-1]):.3f} ===", flush=True)


if __name__ == "__main__":
    main()
