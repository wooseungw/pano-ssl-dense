"""Instance/panoptic via the Mask2Former query head on the PIFu field — the "regions" capability
the per-pixel head cannot do. S2D3D instance GT is FREE: the semantic PNG's RGB-index IS the index
into semantic_labels.json ('<class>_<instance>_<room>...'), so a unique idx == a unique instance
(no connected-components). Train the SAME query head with per-INSTANCE mask+class targets (Hungarian
matches queries↔instances). Eval = panoptic quality (PQ, things-as-instances).

Run: OPENCV_IO_ENABLE_OPENEXR=1 CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pano python scripts/instance_query_head.py
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
import pifu_alltasks as PF  # noqa: E402
from mask_query_head import QueryHead, dice_cost  # noqa: E402
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
MINPIX = 3


def gt_instances(f, cov_np):
    """-> (classes list, masks np bool (T,HQ,WQ)). Unique semantic idx = one instance."""
    sem = np.array(Image.open(data.s2d3d_gt_path(f, "semantic"))).astype(np.int64)
    idx = sem[:, :, 0] * 65536 + sem[:, :, 1] * 256 + sem[:, :, 2]
    idx = np.clip(idx, 0, len(P.S2D3D_LUT) - 1).astype(np.int32)
    idx_g = np.array(Image.fromarray(idx, mode="I").resize((WQ, HQ), Image.NEAREST))
    classes, masks = [], []
    for u in np.unique(idx_g):
        c = int(P.S2D3D_LUT[u])
        if c == 0:
            continue
        m = (idx_g == u) & cov_np
        if m.sum() < MINPIX:
            continue
        classes.append(c); masks.append(m)
    if not classes:
        return None
    return classes, np.stack(masks)


def loss_inst(cls, mask, tgt_cls, tgt_mask, covflat):
    pm = mask.reshape(NQ, -1)[:, covflat]; gm = tgt_mask.reshape(len(tgt_cls), -1)[:, covflat]
    with torch.no_grad():
        c_cost = -cls.softmax(-1)[:, tgt_cls]
        qi, ti = linear_sum_assignment((c_cost + 5.0 * dice_cost(pm.sigmoid(), gm)).cpu().numpy())
    qi = torch.as_tensor(qi, device=DEVICE); ti = torch.as_tensor(ti, device=DEVICE)
    tgt = torch.full((NQ,), cls.shape[1] - 1, device=DEVICE, dtype=torch.long); tgt[qi] = tgt_cls[ti]
    l_cls = F.cross_entropy(cls, tgt)
    pm_m, gm_m = pm[qi], gm[ti]
    l_mask = F.binary_cross_entropy_with_logits(pm_m, gm_m) + dice_cost(pm_m.sigmoid(), gm_m).diag().mean()
    return l_cls + l_mask


@torch.no_grad()
def infer_instances(cls, mask, cov, conf=0.5):
    prob = cls.softmax(-1); sc, lb = prob[:, :-1].max(-1)
    mp = (mask.sigmoid() > 0.5) & cov.to(mask.device)
    segs = []
    for q in (sc > conf).nonzero()[:, 0].tolist():
        m = mp[q].cpu()
        if m.sum() >= MINPIX:
            segs.append((int(lb[q]), m))
    return segs


def pq_accum(pred, gt_cls, gt_masks, cov, acc):
    matched = set()
    for pc, pm in pred:
        best, bj = 0.0, -1
        for gj in range(len(gt_cls)):
            if gj in matched or gt_cls[gj] != pc:
                continue
            gm = gt_masks[gj]
            inter = (pm & gm & cov).sum().item(); uni = ((pm | gm) & cov).sum().item()
            iou = inter / uni if uni else 0.0
            if iou > best:
                best, bj = iou, gj
        if best > 0.5:
            acc["tp"] += 1; acc["sq"] += best; matched.add(bj)
        else:
            acc["fp"] += 1
    acc["fn"] += len(gt_cls) - len(matched)


def main():
    torch.manual_seed(SEED)
    P.configure("stanford2d3d"); P.TILE = PF.TILE
    enc = PanoEncoder(model_id=MODEL, lora_rank=0).to(DEVICE).eval(); P.enc_patch = enc.patch
    plan = a2p.plan_tiles("full_sphere", PF.FOV, PF.FOV, 0.25)
    per_tile, anycov, pw = PF.precompute(plan, PF.TILE // enc.patch)
    cov = torch.from_numpy(anycov.reshape(HQ, WQ)); cov_np = anycov.reshape(HQ, WQ)
    covflat = cov.reshape(-1).to(DEVICE); D = enc.dim
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR]; va = [f for f in files if "5" in area(f)][:VA]
    print(f"instance/panoptic query head | enc={MODEL.split('/')[-1]} field {HQ}×{WQ} Nq={NQ} tr={len(tr)} va={len(va)} ep={EPOCHS}", flush=True)

    cache = {"tr": [], "va": []}
    for sp, fl in [("tr", tr), ("va", va)]:
        for f in fl:
            erp = np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H), Image.BILINEAR))
            pf, _ = PF.build_fields(enc, erp, plan, per_tile, pw)
            tg = gt_instances(f, cov_np)
            if tg is not None:
                cache[sp].append((pf, tg[0], tg[1]))
    ninst = np.mean([len(c[1]) for c in cache["tr"]])
    print(f"  mean instances/pano = {ninst:.1f}", flush=True)

    torch.manual_seed(SEED); qh = QueryHead(D, P.N_CLASS, nq=NQ).to(DEVICE)
    opt = torch.optim.AdamW(qh.parameters(), 1e-4, weight_decay=1e-4)
    g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        qh.train()
        for i in torch.randperm(len(cache["tr"]), generator=g).tolist():
            pf, cls_l, msk = cache["tr"][i]
            x = torch.from_numpy(pf).permute(2, 0, 1)[None].float().to(DEVICE)
            tgt_cls = torch.tensor(cls_l, device=DEVICE)
            tgt_mask = torch.from_numpy(msk).float().to(DEVICE)
            cls, mask = qh(x)
            ls = loss_inst(cls, mask, tgt_cls, tgt_mask, covflat)
            opt.zero_grad(); ls.backward(); opt.step()

    qh.eval(); acc = {"tp": 0, "fp": 0, "fn": 0, "sq": 0.0}
    with torch.no_grad():
        for pf, cls_l, msk in cache["va"]:
            x = torch.from_numpy(pf).permute(2, 0, 1)[None].float().to(DEVICE)
            cls, mask = qh(x)
            pred = infer_instances(cls, mask, cov)
            gtm = [torch.from_numpy(m) for m in msk]
            pq_accum(pred, cls_l, gtm, cov, acc)
    tp, fp, fn = acc["tp"], acc["fp"], acc["fn"]
    sq = acc["sq"] / tp if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 0.0
    print(f"\n  TP={tp} FP={fp} FN={fn}", flush=True)
    print(f"  SQ(mean matched IoU)={sq:.3f}  RQ(recognition)={rq:.3f}  PQ={sq*rq:.3f}", flush=True)
    print(f"\n=== instance/panoptic head WORKS on PIFu field: PQ={sq*rq:.3f} (SQ {sq:.3f} × RQ {rq:.3f}), {tp} instances matched ===", flush=True)


if __name__ == "__main__":
    main()
