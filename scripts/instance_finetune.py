"""END-TO-END FINE-TUNE for instance/panoptic: LoRA-adapt the encoder jointly with the query head,
so gradients flow into the FEATURES (the only accuracy lever). Differentiable PIFu merge in torch
(grid_sample + index_add) connects loss -> field -> tile features -> encoder LoRA.

Tile pixels are cached once (no re-render); the encoder runs WITH grad every step (24 tiles/pano,
batched). Frozen-feature instance was capacity-bound (PQ ~.08); this tests whether unfreezing the
encoder (full data) lifts recognition.

Run: OPENCV_IO_ENABLE_OPENEXR=1 CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pano python scripts/instance_finetune.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import anyres_e2p as a2p  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import pifu_alltasks as PF  # noqa: E402
from mask_query_head import QueryHead  # noqa: E402
from instance_query_head import gt_instances, loss_inst, infer_instances, pq_accum, NQ  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = "cuda"
SEED = 0
MODEL = os.environ.get("MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m")
ERP_W, ERP_H = 2048, 1024
HQ, WQ, TILE = PF.HQ, PF.WQ, PF.TILE
TR = int(os.environ.get("TR", 600))
VA = int(os.environ.get("VA", 60))
EPOCHS = int(os.environ.get("EPOCHS", 6))
LR = float(os.environ.get("LR", 1e-4))


def geom_tensors(plan, patch):
    per_tile, anycov, _ = PF.precompute(plan, TILE // patch)
    geom = []
    for uv, cov, oblw, _ in per_tile:
        idx = np.where(cov)[0]
        geom.append((torch.from_numpy(uv[idx]).view(1, 1, -1, 2).to(DEVICE),
                     torch.from_numpy(idx).long().to(DEVICE),
                     torch.from_numpy(oblw[idx]).float().to(DEVICE)))
    return geom, anycov


def merge_torch(enc, tiles_batch, geom, D):
    """Differentiable PIFu merge: (24,3,H,W) -> (HQ,WQ,D) with grad to the encoder."""
    feat = enc(tiles_batch)                                   # (T,D,gh,gw) WITH grad
    field = torch.zeros(PF.NC, D, device=DEVICE)
    wsum = torch.zeros(PF.NC, device=DEVICE)
    for i, (uvg, cidx, w) in enumerate(geom):
        samp = F.grid_sample(feat[i:i + 1], uvg, mode="bilinear", align_corners=False)[0, :, 0, :].t()
        field = field.index_add(0, cidx, w[:, None] * samp)
        wsum = wsum.index_add(0, cidx, w)
    return (field / wsum.clamp_min(1e-6)[:, None]).reshape(HQ, WQ, D)


def render_tiles(erp, plan):
    return np.stack([np.asarray(a2p.erp_to_pinhole_tile(erp, tp.yaw_deg, tp.pitch_deg, PF.FOV, TILE))
                     for tp in plan])                          # (T,H,W,3) uint8


def main():
    torch.manual_seed(SEED)
    P.configure("stanford2d3d"); P.TILE = TILE
    enc = PanoEncoder(model_id=MODEL, lora_rank=16).to(DEVICE).train()
    try:
        enc.backbone.gradient_checkpointing_enable()
        print("  gradient checkpointing ON", flush=True)
    except Exception as e:
        print(f"  (no grad checkpoint: {e})", flush=True)
    D = enc.dim
    plan = a2p.plan_tiles("full_sphere", PF.FOV, PF.FOV, 0.25)
    geom, anycov = geom_tensors(plan, enc.patch)
    cov = torch.from_numpy(anycov.reshape(HQ, WQ)); cov_np = anycov.reshape(HQ, WQ)
    covflat = cov.reshape(-1).to(DEVICE)
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR]; va = [f for f in files if "5" in area(f)][:VA]
    print(f"instance FINE-TUNE | enc={MODEL.split('/')[-1]}+LoRA field {HQ}×{WQ} Nq={NQ} "
          f"tr={len(tr)} va={len(va)} ep={EPOCHS} trainable={sum(p.numel() for p in enc.trainable_parameters())/1e6:.2f}M+head", flush=True)

    def cache(fl):
        out = []
        for f in fl:
            erp = np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H), Image.BILINEAR))
            tg = gt_instances(f, cov_np)
            if tg is not None:
                out.append((render_tiles(erp, plan), tg[0], tg[1]))
        return out
    ctr, cva = cache(tr), cache(va)
    print(f"  cached tr={len(ctr)} va={len(cva)} (tile pixels)", flush=True)

    torch.manual_seed(SEED); qh = QueryHead(D, P.N_CLASS, nq=NQ).to(DEVICE).train()
    opt = torch.optim.AdamW(list(enc.trainable_parameters()) + list(qh.parameters()), LR, weight_decay=1e-4)
    g = torch.Generator().manual_seed(SEED); t0 = time.time()
    for ep in range(EPOCHS):
        for s, i in enumerate(torch.randperm(len(ctr), generator=g).tolist()):
            tiles_np, cls_l, msk = ctr[i]
            tb = normalize_tiles(torch.from_numpy(tiles_np).float().permute(0, 3, 1, 2).to(DEVICE) / 255.0)
            field = merge_torch(enc, tb, geom, D)
            cls, mask = qh(field.permute(2, 0, 1)[None])
            ls = loss_inst(cls, mask, torch.tensor(cls_l, device=DEVICE),
                           torch.from_numpy(msk).float().to(DEVICE), covflat)
            opt.zero_grad(); ls.backward(); opt.step()
        print(f"  ep{ep} done ({(time.time()-t0)/60:.1f} min, loss {ls.item():.3f})", flush=True)

    enc.eval(); qh.eval(); acc = {"tp": 0, "fp": 0, "fn": 0, "sq": 0.0}
    with torch.no_grad():
        for tiles_np, cls_l, msk in cva:
            tb = normalize_tiles(torch.from_numpy(tiles_np).float().permute(0, 3, 1, 2).to(DEVICE) / 255.0)
            cls, mask = qh(merge_torch(enc, tb, geom, D).permute(2, 0, 1)[None])
            pq_accum(infer_instances(cls, mask, cov), cls_l, [torch.from_numpy(m) for m in msk], cov, acc)
    tp, fp, fn = acc["tp"], acc["fp"], acc["fn"]
    sq = acc["sq"] / tp if tp else 0.0; rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 0.0
    print(f"\n  TP={tp} FP={fp} FN={fn}", flush=True)
    print(f"\n=== FINE-TUNE instance: PQ={sq*rq:.3f} (SQ {sq:.3f} × RQ {rq:.3f}), {tp} matched "
          f"[vs frozen PQ .083] ===", flush=True)


if __name__ == "__main__":
    main()
