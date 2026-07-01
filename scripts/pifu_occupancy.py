"""PIFu-style implicit OCCUPANCY head for the 3D/pointmap task vs direct depth regression.

E2P tiles share one optical center (NO parallax) -> every 3D point on a ray projects to the SAME
pixel -> the pixel-aligned feature is constant along the ray; only radius r varies. So a PIFu
occupancy decoder is MLP(pixel_feat, posenc(r)) -> occupancy(r), and the surface is the 0->1
transition. Extract sub-resolution surface depth by NeRF-style expected value of the density
(d occupancy / d r). Compare to a direct depth-regression MLP on the SAME PIFu field.

  occupancy ~ regression  => use it (gives an implicit 3D surface, the committed DUSt3R deliverable)
  occupancy >  regression  => the implicit decoder is a stronger geometric head
Run: OPENCV_IO_ENABLE_OPENEXR=1 CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pano python scripts/pifu_occupancy.py
"""
from __future__ import annotations

import math
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
import probe_normal as PN  # noqa: E402
import multitask_eval as M  # noqa: E402
import tiling_decode_compare as TDC  # noqa: E402
import pifu_alltasks as PF  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = "cuda"
SEED = 0
MODEL = os.environ.get("MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m")
ERP_W, ERP_H = 2048, 1024
HQ, WQ = PF.HQ, PF.WQ
TR = int(os.environ.get("TR", 80))
VA = int(os.environ.get("VA", 30))
EPOCHS = int(os.environ.get("EPOCHS", 20))
K = int(os.environ.get("K", 32))                       # radii samples along each ray
LOG125 = math.log(1.25)


class OccMLP(nn.Module):
    """PIFu occupancy: (pixel feature, posenc(log-radius)) -> occupancy logit."""

    def __init__(self, d, nfreq=10):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(nfreq)) * math.pi)
        self.net = nn.Sequential(nn.Linear(d + 2 * nfreq, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))

    def posenc(self, lr):                              # lr (K,) -> (K, 2*nfreq)
        a = lr[:, None] * self.freqs[None]
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

    def forward(self, feat, lr):                       # feat (N,d), lr (K,) -> (N,K) logits
        pe = self.posenc(lr); N, K_ = feat.shape[0], lr.shape[0]
        x = torch.cat([feat[:, None, :].expand(N, K_, -1), pe[None].expand(N, K_, -1)], -1)
        return self.net(x)[..., 0]


def expected_depth(logits, lr):                        # (N,K),(K,) -> (N,) NeRF-style surface log-depth
    sig = torch.sigmoid(logits)
    w = (sig[:, 1:] - sig[:, :-1]).clamp(0)            # density peaks at the 0->1 surface
    rmid = 0.5 * (lr[1:] + lr[:-1])
    return (w * rmid[None]).sum(1) / w.sum(1).clamp_min(1e-6)


def cache(enc, files, plan, per_tile, pw):
    out = []
    for f in files:
        erp = np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H), Image.BILINEAR))
        pf, _ = PF.build_fields(enc, erp, plan, per_tile, pw)
        _, lab = P.load_rgb_label(f); nrm, nval = PN.load_rgb_normal(f)[1:]; dn, dval = M.load_depth(f)
        gt = TDC.gt_dict(lab, nrm, nval, dn, dval, HQ, WQ, warp=None)
        out.append((pf.reshape(HQ * WQ, enc.dim), gt["dlog"].reshape(-1), gt["dval"].reshape(-1)))
    return out


def evaluate(pred_fn, cva, cov):
    se, sd, n = 0.0, 0.0, 0
    with torch.no_grad():
        for feat, dlog, dval in cva:
            m = (dval & cov)
            if not m.any():
                continue
            x = torch.from_numpy(feat).float()[m].to(DEVICE)
            pred = pred_fn(x); gt = dlog[m].to(DEVICE)
            e = (pred - gt).abs()
            se += e.sum().item(); sd += (e < LOG125).float().sum().item(); n += int(m.sum())
    return se / max(n, 1), sd / max(n, 1)


def main():
    torch.manual_seed(SEED)
    P.configure("stanford2d3d"); P.TILE = PF.TILE
    enc = PanoEncoder(model_id=MODEL, lora_rank=0).to(DEVICE).eval(); P.enc_patch = enc.patch
    plan = a2p.plan_tiles("full_sphere", PF.FOV, PF.FOV, 0.25)
    per_tile, anycov, pw = PF.precompute(plan, PF.TILE // enc.patch)
    cov = torch.from_numpy(anycov.reshape(-1)); D = enc.dim
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR]; va = [f for f in files if "5" in area(f)][:VA]
    print(f"PIFu occupancy vs direct depth | enc={MODEL.split('/')[-1]} field {HQ}×{WQ} K={K} tr={len(tr)} va={len(va)} ep={EPOCHS}", flush=True)
    ctr = cache(enc, tr, plan, per_tile, pw); cva = cache(enc, va, plan, per_tile, pw)

    allr = np.concatenate([dl[(dv & cov)].numpy() for _, dl, dv in ctr])
    lr = torch.linspace(float(allr.min()) - 0.3, float(allr.max()) + 0.3, K).to(DEVICE)
    print(f"  log-depth range [{lr[0]:.2f},{lr[-1]:.2f}]", flush=True)
    g = torch.Generator().manual_seed(SEED)

    # ---- direct depth regression baseline
    torch.manual_seed(SEED); reg = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1)).to(DEVICE)
    opt = torch.optim.AdamW(reg.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(EPOCHS):
        for i in torch.randperm(len(ctr), generator=g).tolist():
            feat, dlog, dval = ctr[i]; m = (dval & cov)
            if not m.any():
                continue
            x = torch.from_numpy(feat).float()[m].to(DEVICE)
            ls = F.l1_loss(reg(x)[:, 0], dlog[m].to(DEVICE))
            opt.zero_grad(); ls.backward(); opt.step()
    reg.eval()
    reg_res = evaluate(lambda x: reg(x)[:, 0], cva, cov)

    # ---- PIFu occupancy head
    torch.manual_seed(SEED); occ = OccMLP(D).to(DEVICE)
    opt = torch.optim.AdamW(occ.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(EPOCHS):
        for i in torch.randperm(len(ctr), generator=g).tolist():
            feat, dlog, dval = ctr[i]; m = (dval & cov)
            if not m.any():
                continue
            x = torch.from_numpy(feat).float()[m].to(DEVICE); gt = dlog[m].to(DEVICE)
            logits = occ(x, lr)                                   # (n,K)
            occ_gt = (lr[None, :] >= gt[:, None]).float()         # 1 behind the surface
            ls = F.binary_cross_entropy_with_logits(logits, occ_gt)
            opt.zero_grad(); ls.backward(); opt.step()
    occ.eval()
    occ_res = evaluate(lambda x: expected_depth(occ(x, lr), lr), cva, cov)

    print(f"\n  [direct-regress] |Δlog|={reg_res[0]:.4f}  δ<1.25={reg_res[1]:.3f}", flush=True)
    print(f"  [PIFu-occupancy] |Δlog|={occ_res[0]:.4f}  δ<1.25={occ_res[1]:.3f}", flush=True)
    d = reg_res[0] - occ_res[0]
    verdict = "✅ occupancy better" if d > 0.003 else ("≈ tie (use occupancy for implicit surface)" if abs(d) <= 0.003 else "❌ regression better")
    print(f"\n=== Δ|Δlog|(occ better by)={d:+.4f}  δ Δ={occ_res[1]-reg_res[1]:+.3f}  {verdict} ===", flush=True)


if __name__ == "__main__":
    main()
