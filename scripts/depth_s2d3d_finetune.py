"""Supervised LoRA FINE-TUNE on Stanford2D3D depth — the lever the frozen probes never pulled.

All prior experiments (SSL adapters, rank/module scaling, fusion) kept the encoder FROZEN and
trained only a head → capped at DINOv3's information (the "can't beat DINOv3" ceiling). This
UNFREEZES a LoRA adapter and trains it end-to-end with the DEPTH task loss (supervised), so the
encoder itself adapts to the task — how the fully-fine-tuned SOTA baselines actually get their
numbers. Question: does supervised param-efficient fine-tuning beat the frozen probe (AbsRel 0.117)?

Same protocol as depth_s2d3d_bench (fold, GPU E2P render, coverage-mean stitch, metric depth), but:
  * encoder LoRA is TRAINABLE (FT_RANK / FT_TARGETS); no feature cache (re-encode every step);
  * head + LoRA optimized together on L1(log-depth); per-tile CHUNK keeps the ViT-in-graph memory bounded.
Init: fresh LoRA on DINOv3 (FT_INIT unset) or continue an SSL adapter (FT_INIT=<dir>).

Run: FT_RANK=16 FT_TARGETS=qv EPOCHS=8 TR_PANOS=400 CUDA_VISIBLE_DEVICES=1 python scripts/depth_s2d3d_finetune.py
"""
from __future__ import annotations

import os
import sys
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_common as B  # noqa: E402
import data  # noqa: E402
import depth_s2d3d_bench as D  # noqa: E402  (load_depth_m, metrics, SOTA, MIN_DEPTH, DEPTH_CAP)
import runlog  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

FT_RANK = int(os.environ.get("FT_RANK", 16))
FT_ALPHA = int(os.environ.get("FT_ALPHA", 2 * FT_RANK))
FT_TARGETS = os.environ.get("FT_TARGETS", "qv")                 # qv | all
FT_INIT = os.environ.get("FT_INIT", "").strip()                # "" -> fresh LoRA on DINOv3
FT_LR = float(os.environ.get("FT_LR", 3e-4))                    # LoRA + head lr
_TARGETS = {"qv": ["q_proj", "v_proj"],
            "all": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]}


def build_trainable_encoder() -> PanoEncoder:
    """Encoder with a TRAINABLE LoRA adapter (fresh on DINOv3, or continued from FT_INIT)."""
    if FT_INIT:
        enc = PanoEncoder(model_id=B.MODEL, adapter_path=FT_INIT, adapter_trainable=True)
    else:
        enc = PanoEncoder(model_id=B.MODEL, lora_rank=FT_RANK, lora_alpha=FT_ALPHA,
                          lora_targets=_TARGETS[FT_TARGETS])
    return enc.to(B.DEVICE).train()


def encode_grad(enc: PanoEncoder, rgb: np.ndarray, s: int, e: int) -> torch.Tensor:
    """GPU-render tiles [s:e] and encode WITH grad (LoRA in the graph). Returns (b,D,32,32) on DEVICE."""
    erp = torch.from_numpy(rgb).float().permute(2, 0, 1)[None].to(B.DEVICE) / 255.0
    g = B.GRIDS[s:e]
    tiles = F.grid_sample(erp.expand(g.shape[0], -1, -1, -1), g, mode="bilinear", align_corners=False)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=B.DEVICE == "cuda"):
        return enc(normalize_tiles(tiles)).float()


def load_gt_tiles(f: str, plan) -> List:
    """Per-tile (log-depth, valid) GT at HEAD_OUT — cached once (GT is fixed; only features re-encode)."""
    d_m, valid = D.load_depth_m(f, (B.EH, B.EW))
    dg = B.warp_gt_gpu(d_m[:, :, None], "nearest").numpy()[:, :, :, 0]
    mv = B.warp_gt_gpu(valid[:, :, None], "nearest").numpy()[:, :, :, 0] > 0.5
    out = []
    for ti in range(len(plan)):
        dlog = np.log(np.clip(dg[ti], D.MIN_DEPTH, None))
        out.append((torch.from_numpy(dlog).float(), torch.from_numpy(mv[ti] & (dg[ti] > D.MIN_DEPTH))))
    return out


def finetune(enc: PanoEncoder, head: nn.Module, tr: List[str], plan) -> None:
    """End-to-end supervised fine-tune of LoRA + head on L1(log-depth). Encoder is in the graph;
    per-tile CHUNK bounds activation memory. GT tiles cached; RGB re-encoded (with grad) every step."""
    gt_cache = {f: load_gt_tiles(f, plan) for f in tr}           # GT fixed -> cache once
    rgb_cache = {f: np.array(Image.open(f).convert("RGB").resize((B.EW, B.EH), Image.BILINEAR)) for f in tr}
    params = list(head.parameters()) + [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, FT_LR, weight_decay=1e-4)
    g = torch.Generator().manual_seed(B.SEED)
    n_tiles = len(plan)
    for ep in range(B.EPOCHS):
        enc.train(); head.train()
        tot, nb = 0.0, 0
        for i in torch.randperm(len(tr), generator=g).tolist():
            f = tr[i]
            gt = gt_cache[f]
            opt.zero_grad()
            for s in range(0, n_tiles, B.CHUNK):
                e = min(s + B.CHUNK, n_tiles)
                sub = gt[s:e]
                y = torch.stack([t[0] for t in sub]).to(B.DEVICE)
                m = torch.stack([t[1] for t in sub]).to(B.DEVICE)
                if not m.any():
                    continue
                feat = encode_grad(enc, rgb_cache[f], s, e)
                p = F.interpolate(head(feat), (B.HEAD_OUT, B.HEAD_OUT), mode="bilinear", align_corners=False)[:, 0]
                loss = F.l1_loss(p[m], y[m]) * ((e - s) / n_tiles)
                loss.backward()                                  # accumulate over tile chunks
                tot += loss.item(); nb += 1
            opt.step()
        print(f"  ep{ep+1}/{B.EPOCHS} loss={tot/max(nb,1):.4f}", flush=True)


@torch.no_grad()
def eval_ft(enc: PanoEncoder, head: nn.Module, va: List[str], plan, cids) -> dict:
    """Stitched metric-depth eval with the fine-tuned encoder (frozen for eval)."""
    enc.eval(); head.eval()
    agg = {k: 0.0 for k in D.STAT_KEYS}
    cov_sum = 0.0
    for f in va:
        rgb = np.array(Image.open(f).convert("RGB").resize((B.EW, B.EH), Image.BILINEAR))
        feat = B.encode_erp(enc, rgb)                            # no-grad GPU render+encode
        field, cov, covered = B.stitch_field(head, feat, cids, 1)
        cov_sum += cov
        pred = field[0].exp().clamp(D.MIN_DEPTH, D.DEPTH_CAP).numpy()
        gt, valid = D.load_depth_m(f, (B.EH, B.EW))
        v = (valid > 0.5) & covered & np.isfinite(pred)
        s = D.depth_pixel_stats(pred, gt, v)
        for k in D.STAT_KEYS:
            agg[k] += s[k]
    r = D.finalize_depth(agg)
    r["coverage"] = cov_sum / max(len(va), 1)
    return r


def main() -> None:
    torch.manual_seed(B.SEED)
    enc = build_trainable_encoder()
    plan = B.build_plan()
    B.build_sample_grids(plan)
    cids = [B.coord_map(tp.yaw_deg, tp.pitch_deg) for tp in plan]
    files = data.list_erps("stanford2d3d")
    tr, va = B.split_files(files, B.FOLD)
    ntr = sum(p.numel() for p in enc.parameters() if p.requires_grad) / 1e6
    tag = f"ft_{FT_TARGETS}_r{FT_RANK}" + (f"_init-{os.path.basename(FT_INIT)}" if FT_INIT else "")
    print(f"Depth-FINETUNE fold{B.FOLD} {tag} | LoRA {ntr:.3f}M trainable (+head) | tr={len(tr)} va={len(va)} "
          f"eval={B.EH}x{B.EW} ep={B.EPOCHS} lr={FT_LR}", flush=True)

    torch.manual_seed(B.SEED)
    head = B.make_head(enc.dim, 1).to(B.DEVICE)                   # DECODER=conv|deform
    t0 = time.time()
    finetune(enc, head, tr, plan)
    r = eval_ft(enc, head, va, plan, cids)
    head_M = sum(p.numel() for p in head.parameters()) / 1e6

    print(f"\n=== Depth-FINETUNE fold{B.FOLD} | {tag} | {ntr:.2f}M LoRA + {head_M:.2f}M head | "
          f"{time.time()-t0:.0f}s ===", flush=True)
    print(f"  AbsRel={r['AbsRel']:.4f}  SqRel={r['SqRel']:.4f}  RMSE={r['RMSE']:.4f}  RMSE_log={r['RMSE_log']:.4f}", flush=True)
    print(f"  d1={r['d1']*100:.1f}  d2={r['d2']*100:.1f}  d3={r['d3']*100:.1f}  (SI-d1={r['d1_SI']*100:.1f})  "
          f"| coverage {r['coverage']*100:.1f}%", flush=True)
    print("\n  vs frozen-PROBE headline: AbsRel 0.1173 / d1 85.9  (does supervised FT beat the probe?)", flush=True)
    print("  vs SOTA (full FT): UniFuse 0.112/87.1 · SGFormer 0.104/90.0", flush=True)

    run = runlog.create_run(f"depth_finetune_{tag}_f{B.FOLD}", {
        "benchmark": "Stanford2D3D depth SUPERVISED LoRA fine-tune", "fold": B.FOLD, "ft_rank": FT_RANK,
        "ft_alpha": FT_ALPHA, "ft_targets": FT_TARGETS, "ft_init": FT_INIT or "fresh-dinov3",
        "lora_M": round(ntr, 3), "head_M": round(head_M, 3), "epochs": B.EPOCHS, "lr": FT_LR,
        "eval_hw": [B.EH, B.EW], "tr_panos": len(tr), "va_panos": len(va), "metrics": r,
        "vs_frozen_probe": {"AbsRel": 0.1173, "d1": 85.9}})
    torch.save({"head": head.state_dict()}, os.path.join(run, "weights", "head.pt"))
    enc.backbone.save_pretrained(os.path.join(run, "weights", "lora"))
    print(f"saved -> {run}", flush=True)


if __name__ == "__main__":
    main()
