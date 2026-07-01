"""Mask2Former-style QUERY head on the PIFu field — MVP validating the mask-classification
architecture on our unified representation (semantic seg first; instance/panoptic/grounding add
GT + matching on the SAME head).

N learnable queries cross-attend to the PIFu field tokens (transformer decoder) -> each query emits
(class logits, mask embedding); mask = query_embed · pixel_embed. Trained with Hungarian matching
(class + dice cost) + class CE + mask BCE/dice. Semantic inference: argmax_c Σ_q softmax(cls)_qc·σ(mask)_q.
Compared to the per-pixel seg head (same PIFu field) to check the query architecture matches it.

Run: OPENCV_IO_ENABLE_OPENEXR=1 CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pano python scripts/mask_query_head.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment

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
EPOCHS = int(os.environ.get("EPOCHS", 40))
NQ = int(os.environ.get("NQ", 100))
DIM = 256


class QueryHead(nn.Module):
    def __init__(self, D, n_cls, nq=NQ, dim=DIM, layers=3):
        super().__init__()
        self.n_cls = n_cls
        self.in_proj = nn.Conv2d(D, dim, 1)
        self.pix_proj = nn.Conv2d(dim, dim, 1)
        self.pos = nn.Parameter(torch.randn(1, dim, HQ, WQ) * 0.02)
        self.query = nn.Parameter(torch.randn(nq, dim) * 0.02)
        layer = nn.TransformerDecoderLayer(dim, 8, dim * 4, batch_first=True, dropout=0.0)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.cls_head = nn.Linear(dim, n_cls + 1)            # +1 = no-object
        self.mask_embed = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, field):                                 # field (1,D,H,W) -> cls (Nq,C+1), mask (Nq,H,W)
        tok = self.in_proj(field)
        pix = self.pix_proj(tok)[0]                           # (dim,H,W)
        mem = (tok + self.pos).flatten(2).transpose(1, 2)     # (1,HW,dim)
        q = self.dec(self.query[None], mem)[0]                # (Nq,dim)
        cls = self.cls_head(q)
        mask = torch.einsum("qd,dhw->qhw", self.mask_embed(q), pix)
        return cls, mask


def gt_masks(seg, cov):
    """present-class binary masks (on cov) for semantic targets. seg,(H,W) tensor; cov (H,W)."""
    classes, masks = [], []
    valid = cov & (seg != P.IGNORE)
    for c in torch.unique(seg[valid]).tolist():
        if c == 0:                                            # treat 0 as void (mIoU is over 1..N-1)
            continue
        classes.append(c); masks.append(((seg == c) & cov).float())
    if not classes:
        return None
    return torch.tensor(classes, device=DEVICE), torch.stack(masks).to(DEVICE)   # (T,), (T,H,W)


def dice_cost(p, g, eps=1.0):                                 # p (Nq,N), g (T,N) -> (Nq,T)
    inter = p @ g.t()
    return 1 - (2 * inter + eps) / (p.sum(1, keepdim=True) + g.sum(1)[None] + eps)


def loss_one(cls, mask, tgt_cls, tgt_mask, covflat):
    pm = mask.reshape(NQ, -1)[:, covflat]; gm = tgt_mask.reshape(len(tgt_cls), -1)[:, covflat]
    with torch.no_grad():
        cprob = cls.softmax(-1)                               # (Nq,C+1)
        c_cost = -cprob[:, tgt_cls]                           # (Nq,T)
        m_cost = dice_cost(pm.sigmoid(), gm)
        qi, ti = linear_sum_assignment((c_cost + 5.0 * m_cost).cpu().numpy())
    qi = torch.as_tensor(qi, device=DEVICE); ti = torch.as_tensor(ti, device=DEVICE)
    tgt = torch.full((NQ,), self_no_obj(cls), device=DEVICE, dtype=torch.long)
    tgt[qi] = tgt_cls[ti]
    l_cls = F.cross_entropy(cls, tgt)
    pm_m, gm_m = pm[qi], gm[ti]
    l_mask = F.binary_cross_entropy_with_logits(pm_m, gm_m) + dice_cost(pm_m.sigmoid(), gm_m).diag().mean()
    return l_cls + l_mask


def self_no_obj(cls):
    return cls.shape[1] - 1                                   # last index = no-object


@torch.no_grad()
def infer_seg(cls, mask):
    scores = cls.softmax(-1)[:, :-1]                          # (Nq,C) drop no-object
    return torch.einsum("qc,qhw->chw", scores, mask.sigmoid()).argmax(0).cpu()   # (H,W) labels


def miou(preds, segs, cov):
    inter = torch.zeros(P.N_CLASS); union = torch.zeros(P.N_CLASS)
    for pr, y in zip(preds, segs):
        mm = (y != P.IGNORE) & cov
        for c in range(1, P.N_CLASS):
            inter[c] += ((pr == c) & (y == c) & mm).sum(); union[c] += (((pr == c) | (y == c)) & mm).sum()
    return float(np.mean([(inter[c] / union[c]).item() for c in range(1, P.N_CLASS) if union[c] > 0]))


def main():
    torch.manual_seed(SEED)
    P.configure("stanford2d3d"); P.TILE = PF.TILE
    enc = PanoEncoder(model_id=MODEL, lora_rank=0).to(DEVICE).eval(); P.enc_patch = enc.patch
    plan = a2p.plan_tiles("full_sphere", PF.FOV, PF.FOV, 0.25)
    per_tile, anycov, pw = PF.precompute(plan, PF.TILE // enc.patch)
    cov = torch.from_numpy(anycov.reshape(HQ, WQ)); covflat = cov.reshape(-1).to(DEVICE); D = enc.dim
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR]; va = [f for f in files if "5" in area(f)][:VA]
    print(f"Mask2Former query head vs per-pixel seg | enc={MODEL.split('/')[-1]} field {HQ}×{WQ} "
          f"Nq={NQ} tr={len(tr)} va={len(va)} ep={EPOCHS}", flush=True)

    cache = {"tr": [], "va": []}
    for sp, fl in [("tr", tr), ("va", va)]:
        for f in fl:
            erp = np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H), Image.BILINEAR))
            pf, _ = PF.build_fields(enc, erp, plan, per_tile, pw)
            _, lab = P.load_rgb_label(f); nrm, nval = PN.load_rgb_normal(f)[1:]; dn, dval = M.load_depth(f)
            seg = TDC.gt_dict(lab, nrm, nval, dn, dval, HQ, WQ, warp=None)["seg"]
            cache[sp].append((pf, seg))

    # ---- per-pixel baseline
    torch.manual_seed(SEED); base = PF.MLP(D, P.N_CLASS).to(DEVICE)
    opt = torch.optim.AdamW(base.parameters(), 1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        for i in torch.randperm(len(cache["tr"]), generator=g).tolist():
            pf, seg = cache["tr"][i]
            x = torch.from_numpy(pf).permute(2, 0, 1)[None].float().to(DEVICE)
            y = seg.to(DEVICE); m = (y != P.IGNORE) & cov.to(DEVICE)
            ls = F.cross_entropy(base(x).permute(0, 2, 3, 1)[0][m], y[m])
            opt.zero_grad(); ls.backward(); opt.step()
    base.eval()
    with torch.no_grad():
        bp = [base(torch.from_numpy(pf).permute(2, 0, 1)[None].float().to(DEVICE)).argmax(1)[0].cpu() for pf, _ in cache["va"]]
    base_miou = miou(bp, [s for _, s in cache["va"]], cov)

    # ---- query head
    torch.manual_seed(SEED); qh = QueryHead(D, P.N_CLASS).to(DEVICE)
    opt = torch.optim.AdamW(qh.parameters(), 1e-4, weight_decay=1e-4)
    for ep in range(EPOCHS):
        qh.train()
        for i in torch.randperm(len(cache["tr"]), generator=g).tolist():
            pf, seg = cache["tr"][i]; tg = gt_masks(seg.to(DEVICE), cov.to(DEVICE))
            if tg is None:
                continue
            x = torch.from_numpy(pf).permute(2, 0, 1)[None].float().to(DEVICE)
            cls, mask = qh(x)
            ls = loss_one(cls, mask, tg[0], tg[1], covflat)
            opt.zero_grad(); ls.backward(); opt.step()
    qh.eval()
    with torch.no_grad():
        qp = []
        for pf, _ in cache["va"]:
            x = torch.from_numpy(pf).permute(2, 0, 1)[None].float().to(DEVICE)
            cls, mask = qh(x); qp.append(infer_seg(cls, mask))
    q_miou = miou(qp, [s for _, s in cache["va"]], cov)

    print(f"\n  [per-pixel seg head] mIoU = {base_miou:.3f}", flush=True)
    print(f"  [Mask2Former query ] mIoU = {q_miou:.3f}", flush=True)
    d = q_miou - base_miou
    verdict = "✅ query head matches/beats" if d > -0.015 else "❌ query head underperforms (data/tuning?)"
    print(f"\n=== query {q_miou:.3f} vs per-pixel {base_miou:.3f}  Δ={d:+.3f}  {verdict} ===", flush=True)


if __name__ == "__main__":
    main()
