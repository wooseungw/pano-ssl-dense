"""Decisive objective-aligned gate: is there LEARNABLE semantic structure to predict unobserved
panorama regions, BEYOND trivial continuity?

The latent-regression predictor (spatial_jepa_metric) lost to nearest-copy on seg because it was
trained for cosine, not for the deliverable. Here we train a completion U-Net DIRECTLY for the
metric: visible latent field -> predict the CLASS of masked cells (cross-entropy). This is the
BEST CASE for "predict unobserved semantics". Compare to nearest-copy-then-decode, per distance bin.

  beats nearest  -> learnable scene structure exists -> spatial-JEPA worth building
  ties/loses     -> panorama content is CONTINUITY-BOUND (nearest-copy is the ceiling) -> pivot

Run: CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pano python scripts/spatial_jepa_objgate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import anyres_e2p as a2p  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
from spatial_jepa_gate import Block, cpad, random_block, nearest_fill, dist_to_observed  # noqa: E402
from spatial_jepa_metric import cache  # noqa: E402  (field+cov+seg per pano)
from viz_merged_field import cell_ids, HF, WF, TILE, PATCH  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = "cuda"
SEED = 0
MODEL = os.environ.get("MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m")
EPOCHS = int(os.environ.get("EPOCHS", 40))
TR = int(os.environ.get("TR", 120))
VA = int(os.environ.get("VA", 30))
BINS = [(0, 2), (2, 4), (4, 7), (7, 100)]


class CompletionUNet(nn.Module):
    """Visible latent field (+mask ch) -> per-cell class logits. Trained DIRECTLY for seg-acc."""

    def __init__(self, D, ncls, h=192):
        super().__init__()
        self.inp = nn.Conv2d(D + 1, h, 1)
        self.d1 = Block(h, h); self.d2 = Block(h, 2 * h); self.mid = Block(2 * h, 2 * h)
        self.u2 = Block(2 * h + 2 * h, h); self.u1 = Block(h + h, h)
        self.out = nn.Conv2d(h, ncls, 1)

    def forward(self, x, m):
        x0 = self.inp(torch.cat([x, m], 1))
        a = self.d1(x0); b = self.d2(F.avg_pool2d(a, 2)); c = self.mid(F.avg_pool2d(b, 2))
        c = F.interpolate(c, scale_factor=2, mode="bilinear", align_corners=False)
        b = self.u2(torch.cat([c, b], 1)); b = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        return self.out(self.u1(torch.cat([b, a], 1)))


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    P.configure("stanford2d3d"); P.TILE = TILE
    enc = PanoEncoder(model_id=MODEL, lora_rank=0).to(DEVICE).eval(); P.enc_patch = enc.patch
    D = enc.dim
    plan = a2p.plan_tiles("full_sphere", 65.0, 65.0, 0.25); ids = cell_ids(plan, TILE // PATCH)
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR]; va = [f for f in files if "5" in area(f)][:VA]
    print(f"objective-aligned gate (semantic completion) | enc={MODEL.split('/')[-1]} field={HF}×{WF} "
          f"tr={len(tr)} va={len(va)} ep={EPOCHS}", flush=True)
    ctr = cache(enc, tr, plan, ids); cva = cache(enc, va, plan, ids)
    scale = float(np.std(ctr[0][0][ctr[0][1]])) + 1e-6

    # seg head on true latent (for nearest/oracle/mean decode baselines)
    Xtr = np.concatenate([fs[cov] for fs, cov, _ in ctr]); ytr = np.concatenate([sg[cov] for _, cov, sg in ctr])
    keep = ytr != P.IGNORE; sel = np.random.RandomState(SEED).permutation(int(keep.sum()))[:300000]
    Xt = torch.from_numpy(Xtr[keep][sel]).to(DEVICE); yt = torch.from_numpy(ytr[keep][sel]).to(DEVICE)
    head = nn.Linear(D, P.N_CLASS).to(DEVICE); ho = torch.optim.Adam(head.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(600):
        ho.zero_grad(); F.cross_entropy(head(Xt), yt, ignore_index=P.IGNORE).backward(); ho.step()

    # completion U-Net trained DIRECTLY for masked-cell class
    net = CompletionUNet(D, P.N_CLASS).to(DEVICE); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(SEED)

    def to_in(field, m):
        x = torch.from_numpy(field).permute(2, 0, 1)[None].to(DEVICE) / scale
        mt = torch.from_numpy(m.astype(np.float32))[None, None].to(DEVICE)
        return x * (1 - mt), mt

    for _ in range(EPOCHS):
        net.train()
        for i in torch.randperm(len(ctr), generator=g).tolist():
            field, cov, seg = ctr[i]; m = random_block(g); tr_cells = m & cov & (seg != P.IGNORE)
            if tr_cells.sum() < 5:
                continue
            xin, mt = to_in(field, m)
            logit = net(xin, mt)[0].permute(1, 2, 0)              # (H,W,ncls)
            s = torch.from_numpy(tr_cells).to(DEVICE)
            y = torch.from_numpy(seg).to(DEVICE)[s]
            F.cross_entropy(logit[s], y).backward(); opt.step(); opt.zero_grad()

    # eval
    gv = torch.Generator().manual_seed(123); vmasks = [random_block(gv) for _ in cva]
    acc = {k: {b: [0, 0] for b in BINS} for k in ("oracle", "near", "mean", "complete")}

    @torch.no_grad()
    def dec(arr):
        return head(torch.from_numpy(arr).float().to(DEVICE)).argmax(1).cpu().numpy()

    net.eval()
    with torch.no_grad():
        for (field, cov, seg), m in zip(cva, vmasks):
            ev = m & cov & (seg != P.IGNORE); valid = (~m) & cov
            if ev.sum() < 5:
                continue
            xin, mt = to_in(field, m)
            comp = net(xin, mt)[0].permute(1, 2, 0).argmax(2).cpu().numpy()   # (H,W)
            nf = nearest_fill(field, valid); mu = field[valid].mean(0)
            d = dist_to_observed(m); de = d[ev]; ge = seg[ev]
            cls = {"oracle": dec(field[ev]), "near": dec(nf[ev]),
                   "mean": dec(np.broadcast_to(mu, (ev.sum(), D)).copy()), "complete": comp[ev]}
            for lo, hi in BINS:
                bm = (de >= lo) & (de < hi)
                if bm.any():
                    for k in cls:
                        acc[k][(lo, hi)][0] += int((cls[k][bm] == ge[bm]).sum()); acc[k][(lo, hi)][1] += int(bm.sum())

    def am(k, b): n = acc[k][b][1]; return acc[k][b][0] / n if n else float("nan")
    def ov(k):
        c = sum(acc[k][b][0] for b in BINS); n = sum(acc[k][b][1] for b in BINS); return c / n if n else float("nan")
    hdr = " ".join(f"d∈[{lo},{hi})".rjust(10) for lo, hi in BINS)
    print("\nSEG pixel-acc of UNOBSERVED cells — completion U-Net trained DIRECTLY for class:")
    print(f"{'method':9} {'overall':>8} {hdr}")
    for k in ("oracle", "near", "mean", "complete"):
        tag = "  (ceiling)" if k == "oracle" else ("  ⭐trained-for-metric" if k == "complete" else "")
        print(f"{k:9} {ov(k):8.3f} " + " ".join(f"{am(k, b):10.3f}" for b in BINS) + tag, flush=True)
    cp, nr = ov("complete"), ov("near")
    far = (am("complete", BINS[-1]), am("near", BINS[-1]))
    verdict = "✅ learnable structure (build JEPA)" if cp - nr > 0.02 else "❌ continuity-bound (nearest is ceiling -> pivot)"
    print(f"\n=== completion {cp:.3f} vs nearest {nr:.3f}  Δ={cp-nr:+.3f} | far-bin {far[0]:.3f} vs {far[1]:.3f}  {verdict} ===", flush=True)


if __name__ == "__main__":
    main()
