"""Spatial-JEPA gate: is the panorama's latent structure PREDICTABLE from partial observation?

Deliverable target: accurate partial observation -> predict the FULL panorama (latent space, like
V-JEPA but over the SPATIAL axis). Before building the full predictor, test the core premise on the
merged 64x128 field: mask a contiguous block (~one tile footprint), predict its latent from the rest
with a small circular-pad U-Net, and check it beats trivial fills (global-mean, nearest-visible).
Also report cosine vs distance-to-observed-boundary — how far into the unknown is predictable.

Gate: predictor cosine > nearest-fill by a clear margin AND near-boundary cosine high
      => spatial latent structure is predictable => build the full spatial-JEPA. Else reconsider.

Run: CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pano python scripts/spatial_jepa_gate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import anyres_e2p as a2p  # noqa: E402
from viz_merged_field import build_field, cell_ids, HF, WF, TILE, PATCH, ERP_W, ERP_H  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = "cuda"
SEED = 0
MODEL = os.environ.get("MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m")  # ViT-B for gate speed
EPOCHS = int(os.environ.get("EPOCHS", 40))
TR = int(os.environ.get("TR", 120))
VA = int(os.environ.get("VA", 30))


# --------------------------------------------------------------------------- predictor
def cconv(c_in, c_out):
    return nn.Conv2d(c_in, c_out, 3, padding=0)


def cpad(x):
    x = F.pad(x, (1, 1, 0, 0), mode="circular")       # wrap longitude
    return F.pad(x, (0, 0, 1, 1), mode="replicate")   # clamp latitude


class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.c1 = cconv(ci, co); self.c2 = cconv(co, co)
        self.n1 = nn.GroupNorm(8, co); self.n2 = nn.GroupNorm(8, co)

    def forward(self, x):
        x = F.gelu(self.n1(self.c1(cpad(x))))
        return F.gelu(self.n2(self.c2(cpad(x))))


class FieldUNet(nn.Module):
    """Predict the masked field latent from the visible field (+ mask channel). Circular in lon."""

    def __init__(self, D, h=192):
        super().__init__()
        self.inp = nn.Conv2d(D + 1, h, 1)
        self.d1 = Block(h, h); self.d2 = Block(h, 2 * h)
        self.mid = Block(2 * h, 2 * h)
        self.u2 = Block(2 * h + 2 * h, h); self.u1 = Block(h + h, h)
        self.out = nn.Conv2d(h, D, 1)

    def forward(self, x, m):                            # x:(B,D,H,W) visible(zeroed in mask), m:(B,1,H,W)
        x0 = self.inp(torch.cat([x, m], 1))
        a = self.d1(x0)
        b = self.d2(F.avg_pool2d(a, 2))
        c = self.mid(F.avg_pool2d(b, 2))
        c = F.interpolate(c, scale_factor=2, mode="bilinear", align_corners=False)
        b = self.u2(torch.cat([c, b], 1))
        b = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        a = self.u1(torch.cat([b, a], 1))
        return self.out(a)


# --------------------------------------------------------------------------- masks / baselines
def random_block(g):
    bh = int(torch.randint(12, 28, (1,), generator=g)); bw = int(torch.randint(20, 56, (1,), generator=g))
    r0 = int(torch.randint(0, HF - bh + 1, (1,), generator=g)); c0 = int(torch.randint(0, WF, (1,), generator=g))
    m = np.zeros((HF, WF), bool)
    cols = (c0 + np.arange(bw)) % WF
    m[r0:r0 + bh][:, cols] = True
    return m


def nearest_fill(field, valid):                        # field:(H,W,D) valid:(H,W) -> filled at all cells
    v3 = np.concatenate([valid] * 3, 1); f3 = np.concatenate([field] * 3, 1)
    idx = distance_transform_edt(~v3, return_distances=False, return_indices=True)
    return f3[tuple(idx)][:, WF:2 * WF]


def dist_to_observed(mask):                             # distance (cells) from each masked cell to nearest visible
    m3 = np.concatenate([mask] * 3, 1)
    d = distance_transform_edt(m3)
    return d[:, WF:2 * WF]


def cos_np(a, b):                                       # row-wise cosine, (N,D)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a * b).sum(1)


# --------------------------------------------------------------------------- data
def cache_fields(enc, files, plan, ids):
    out = []
    for f in files:
        erp = np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H), Image.BILINEAR))
        fs, cov = build_field(enc, erp, plan, ids)
        out.append((fs.reshape(HF, WF, enc.dim).astype(np.float32), cov.reshape(HF, WF)))
    return out


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    enc = PanoEncoder(model_id=MODEL, lora_rank=0).to(DEVICE).eval()
    D = enc.dim
    plan = a2p.plan_tiles("full_sphere", 65.0, 65.0, 0.25)
    ids = cell_ids(plan, TILE // PATCH)
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR]
    va = [f for f in files if "5" in area(f)][:VA]
    print(f"spatial-JEPA gate | enc={MODEL.split('/')[-1]} D={D} field={HF}×{WF} tr={len(tr)} va={len(va)} ep={EPOCHS}", flush=True)
    ctr = cache_fields(enc, tr, plan, ids); cva = cache_fields(enc, va, plan, ids)
    scale = float(np.std([c[0][c[1]] for c in ctr][0])) + 1e-6        # rough global feature scale

    # fixed val masks
    gv = torch.Generator().manual_seed(123)
    vmasks = [random_block(gv) for _ in cva]

    net = FieldUNet(D).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(SEED)

    def to_in(field, m):
        x = torch.from_numpy(field).permute(2, 0, 1)[None].to(DEVICE) / scale
        mt = torch.from_numpy(m.astype(np.float32))[None, None].to(DEVICE)
        return x * (1 - mt), mt

    for ep in range(EPOCHS):
        net.train(); order = torch.randperm(len(ctr), generator=g).tolist()
        for i in order:
            field, cov = ctr[i]; m = random_block(g)
            ev = m & cov
            if ev.sum() < 5:
                continue
            xin, mt = to_in(field, m)
            pred = net(xin, mt)[0].permute(1, 2, 0) * scale
            tgt = torch.from_numpy(field).to(DEVICE)
            sel = torch.from_numpy(ev).to(DEVICE)
            p = F.normalize(pred[sel], dim=1); t = F.normalize(tgt[sel], dim=1)
            loss = (1 - (p * t).sum(1)).mean() + 0.1 * F.mse_loss(pred[sel] / scale, tgt[sel] / scale)
            opt.zero_grad(); loss.backward(); opt.step()

    # ---- eval: predictor vs baselines, + distance bins
    net.eval()
    bins = [(0, 2), (2, 4), (4, 7), (7, 100)]
    acc = {k: {b: [] for b in bins} for k in ("pred", "near", "mean")}
    tot = {k: [] for k in ("pred", "near", "mean")}
    with torch.no_grad():
        for (field, cov), m in zip(cva, vmasks):
            ev = m & cov
            if ev.sum() < 5:
                continue
            valid = (~m) & cov
            xin, mt = to_in(field, m)
            pr = (net(xin, mt)[0].permute(1, 2, 0) * scale).cpu().numpy()
            nf = nearest_fill(field, valid)
            mn = np.broadcast_to(field[valid].mean(0), field.shape)
            d = dist_to_observed(m)
            true = field[ev]
            preds = {"pred": pr[ev], "near": nf[ev], "mean": mn[ev]}
            de = d[ev]
            for k, pv in preds.items():
                c = cos_np(pv, true); tot[k].append(c)
                for (lo, hi) in bins:
                    sel = (de >= lo) & (de < hi)
                    if sel.any():
                        acc[k][(lo, hi)].append(c[sel])

    def m_(lst): return float(np.concatenate(lst).mean()) if lst else float("nan")
    print("\ncosine(predicted, true) on masked&covered cells — higher=more predictable:")
    print(f"{'method':8} {'overall':>8} " + " ".join(f"d∈[{lo},{hi})".rjust(10) for lo, hi in bins))
    for k in ("mean", "near", "pred"):
        row = " ".join(f"{m_(acc[k][b]):10.3f}" for b in bins)
        print(f"{k:8} {m_(tot[k]):8.3f} {row}", flush=True)
    pred_o, near_o = m_(tot["pred"]), m_(tot["near"])
    gate = "✅ predictable (build spatial-JEPA)" if pred_o - near_o > 0.03 else "❌ ~nearest-fill (reconsider)"
    print(f"\n=== predictor {pred_o:.3f} vs nearest-fill {near_o:.3f}  Δ={pred_o-near_o:+.3f}  {gate} ===", flush=True)


if __name__ == "__main__":
    main()
