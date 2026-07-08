"""Streaming supervised LoRA fine-tune on Stanford2D3D for ALL 3 dense tasks — the go-forward
downstream protocol (see memory/pano-ssl-eval-protocol): TASK in {depth, normal, seg}, ALL train
images, NO feature cache (encoder is trainable -> re-encode every step = "skip pre-encoding").

FT_INIT continues an SSL adapter (e.g. ckpt_ssl_all from the all-image SSL run) or, unset, starts a
fresh LoRA on DINOv3 — so we can ask "does SSL-all pretraining improve fine-tuning vs fresh?".
GPU E2P render (bench_common), per-tile CHUNK bounds the ViT-in-graph memory, coverage-mean stitch.

Run: TASK=depth FT_INIT=ckpt_ssl_all EPOCHS=6 CUDA_VISIBLE_DEVICES=1 python scripts/finetune_s2d3d.py
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
import depth_s2d3d_bench as D  # noqa: E402
import normal_s2d3d_bench as N  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import runlog  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

TASK = os.environ.get("TASK", "depth")                         # depth | normal | seg
FT_RANK = int(os.environ.get("FT_RANK", 16))
FT_ALPHA = int(os.environ.get("FT_ALPHA", 2 * FT_RANK))
FT_TARGETS = os.environ.get("FT_TARGETS", "qv")
FT_INIT = os.environ.get("FT_INIT", "").strip()               # "" -> fresh LoRA on DINOv3
FT_LR = float(os.environ.get("FT_LR", 3e-4))
_TG = {"qv": ["q_proj", "v_proj"], "all": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]}
CH = {"depth": 1, "normal": 3, "seg": None}                    # seg set after P.configure


def build_encoder() -> PanoEncoder:
    if FT_INIT:
        enc = PanoEncoder(model_id=B.MODEL, adapter_path=FT_INIT, adapter_trainable=True)
    else:
        enc = PanoEncoder(model_id=B.MODEL, lora_rank=FT_RANK, lora_alpha=FT_ALPHA, lora_targets=_TG[FT_TARGETS])
    return enc.to(B.DEVICE).train()


def encode_grad(enc: PanoEncoder, rgb: np.ndarray, s: int, e: int) -> torch.Tensor:
    """GPU-render tiles [s:e] and encode WITH grad (LoRA in the graph)."""
    erp = torch.from_numpy(rgb).float().permute(2, 0, 1)[None].to(B.DEVICE) / 255.0
    g = B.GRIDS[s:e]
    tiles = F.grid_sample(erp.expand(g.shape[0], -1, -1, -1), g, mode="bilinear", align_corners=False)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=B.DEVICE == "cuda"):
        return enc(normalize_tiles(tiles)).float()


# ---------------------------------------------------------------- per-task GT / loss / metric
def gt_tiles(f: str, plan) -> List:
    """Per-tile supervision at HEAD_OUT (GPU-warped, task-specific)."""
    if TASK == "depth":
        d_m, val = D.load_depth_m(f, (B.EH, B.EW))
        dg = B.warp_gt_gpu(d_m[:, :, None], "nearest").numpy()[:, :, :, 0]
        mv = B.warp_gt_gpu(val[:, :, None], "nearest").numpy()[:, :, :, 0] > 0.5
        return [(torch.from_numpy(np.log(np.clip(dg[i], D.MIN_DEPTH, None))).float(),
                 torch.from_numpy(mv[i] & (dg[i] > D.MIN_DEPTH))) for i in range(len(plan))]
    if TASK == "normal":
        nrm, val = N.load_normal(f, (B.EH, B.EW))
        ng = B.warp_gt_gpu(nrm, "nearest").numpy()
        mv = B.warp_gt_gpu(val[:, :, None], "nearest").numpy()[:, :, :, 0] > 0.5
        out = []
        for i in range(len(plan)):
            u = ng[i] / np.clip(np.linalg.norm(ng[i], axis=2, keepdims=True), 1e-6, None)
            out.append((torch.from_numpy(u).float(), torch.from_numpy(mv[i])))
        return out
    lab = P.load_rgb_label(f)[1].astype(np.float32)                              # seg (EH,EW) class ids
    lg = B.warp_gt_gpu(lab[:, :, None], "nearest").numpy()[:, :, :, 0]
    return [(torch.from_numpy(np.round(lg[i]).astype(np.int64)), None) for i in range(len(plan))]


def loss_of(out: torch.Tensor, sub: List) -> torch.Tensor:
    """out: head output (b,C,128,128) at HEAD_OUT. Task loss over valid pixels."""
    if TASK == "depth":
        y = torch.stack([t[0] for t in sub]).to(B.DEVICE)
        m = torch.stack([t[1] for t in sub]).to(B.DEVICE)
        return F.l1_loss(out[:, 0][m], y[m]) if m.any() else out.sum() * 0.0
    if TASK == "normal":
        y = torch.stack([t[0] for t in sub]).permute(0, 3, 1, 2).to(B.DEVICE)
        m = torch.stack([t[1] for t in sub]).to(B.DEVICE)
        cos = (F.normalize(out, dim=1) * y).sum(1)
        return (1 - cos)[m].mean() if m.any() else out.sum() * 0.0
    y = torch.stack([t[0] for t in sub]).to(B.DEVICE)
    return F.cross_entropy(out, y, ignore_index=P.IGNORE)


def finetune(enc: PanoEncoder, head: nn.Module, tr: List[str], plan) -> None:
    gt_cache = {f: gt_tiles(f, plan) for f in tr}
    rgb_cache = {f: np.array(Image.open(f).convert("RGB").resize((B.EW, B.EH), Image.BILINEAR)) for f in tr}
    params = list(head.parameters()) + [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, FT_LR, weight_decay=1e-4)
    g = torch.Generator().manual_seed(B.SEED)
    nt = len(plan)
    for ep in range(B.EPOCHS):
        enc.train(); head.train()
        tot, nb = 0.0, 0
        for i in torch.randperm(len(tr), generator=g).tolist():
            f = tr[i]; gt = gt_cache[f]; opt.zero_grad()
            for s in range(0, nt, B.CHUNK):
                e = min(s + B.CHUNK, nt)
                sub = gt[s:e]
                out = F.interpolate(head(encode_grad(enc, rgb_cache[f], s, e)),
                                    (B.HEAD_OUT, B.HEAD_OUT), mode="bilinear", align_corners=False)
                loss = loss_of(out, sub) * ((e - s) / nt)
                loss.backward(); tot += float(loss); nb += 1
            opt.step()
        print(f"  ep{ep+1}/{B.EPOCHS} loss={tot/max(nb,1):.4f}", flush=True)


@torch.no_grad()
def evaluate(enc: PanoEncoder, head: nn.Module, va: List[str], plan, cids) -> dict:
    enc.eval(); head.eval()
    C = P.N_CLASS if TASK == "seg" else CH[TASK]
    dstat = {k: 0.0 for k in D.STAT_KEYS}
    nstat = {k: 0.0 for k in N.STAT_KEYS}; nhist = np.zeros(N.NBINS)
    inter = torch.zeros(P.N_CLASS); union = torch.zeros(P.N_CLASS)
    cov_sum = 0.0
    for f in va:
        rgb = np.array(Image.open(f).convert("RGB").resize((B.EW, B.EH), Image.BILINEAR))
        field, cov, covered = B.stitch_field(head, B.encode_erp(enc, rgb), cids, C)
        cov_sum += cov
        if TASK == "depth":
            pred = field[0].exp().clamp(D.MIN_DEPTH, D.DEPTH_CAP).numpy()
            gt, val = D.load_depth_m(f, (B.EH, B.EW))
            s = D.depth_pixel_stats(pred, gt, (val > 0.5) & covered & np.isfinite(pred))
            for k in D.STAT_KEYS:
                dstat[k] += s[k]
        elif TASK == "normal":
            pred = F.normalize(field.permute(1, 2, 0), dim=-1).numpy()
            gt, val = N.load_normal(f, (B.EH, B.EW))
            s = N.normal_pixel_stats(pred, gt, (val > 0.5) & covered)
            for k in N.STAT_KEYS:
                nstat[k] += s[k]
            nhist += s["hist"]
        else:
            pred = field.argmax(0).numpy()
            gt = P.load_rgb_label(f)[1]
            m = (gt != P.IGNORE) & covered
            for c in range(1, P.N_CLASS):
                inter[c] += ((pred == c) & (gt == c) & m).sum()
                union[c] += (((pred == c) | (gt == c)) & m).sum()
    cov = cov_sum / max(len(va), 1)
    if TASK == "depth":
        r = D.finalize_depth(dstat)
    elif TASK == "normal":
        r = N.finalize_normal(nstat, nhist)
    else:
        ious = {P.S2D3D_CLASSES[c - 1]: ((inter[c] / union[c]).item() if union[c] > 0 else 0.0)
                for c in range(1, P.N_CLASS)}
        r = {"mIoU": float(np.mean(list(ious.values()))), "per_class": ious}
    r["coverage"] = cov
    return r


@torch.no_grad()
def save_viz(enc: PanoEncoder, head: nn.Module, va: List[str], plan, cids, run: str, C: int) -> None:
    """GT-alongside prediction PNGs for 3 fixed val samples (input/gt/pred + compare panel)."""
    enc.eval(); head.eval()
    pal = runlog.seg_palette(P.N_CLASS) if TASK == "seg" else None
    for j, i in enumerate(runlog.spread_indices(len(va), 3)):
        f = va[i]
        rgb = np.array(Image.open(f).convert("RGB").resize((B.EW, B.EH), Image.BILINEAR))
        field, _, covered = B.stitch_field(head, B.encode_erp(enc, rgb), cids, C)
        rgb01 = rgb.astype(np.float32) / 255.0
        if TASK == "depth":
            pred = field[0].exp().clamp(D.MIN_DEPTH, D.DEPTH_CAP).numpy()
            gt, val = D.load_depth_m(f, (B.EH, B.EW))
            v = (val > 0.5) & covered
            runlog.save_depth_sample(run, "depth", j, rgb01, np.log(np.clip(gt, D.MIN_DEPTH, None)),
                                     {"ft": np.log(np.clip(pred, D.MIN_DEPTH, None))}, v, scale=1)
        elif TASK == "normal":
            pred = F.normalize(field.permute(1, 2, 0), dim=-1).numpy()
            gt, val = N.load_normal(f, (B.EH, B.EW))
            runlog.save_normal_sample(run, "normal", j, rgb01, gt, {"ft": pred}, (val > 0.5) & covered, scale=1)
        else:
            pred = field.argmax(0).numpy()
            gt = P.load_rgb_label(f)[1]
            runlog.save_seg_sample(run, "seg", j, rgb01, gt, {"ft": pred}, pal, scale=1)


def main() -> None:
    torch.manual_seed(B.SEED)
    if TASK == "seg":
        P.configure("stanford2d3d"); P.TILE = B.TILE; P.WORK_HW = (B.EH, B.EW)
        CH["seg"] = P.N_CLASS
    else:
        P.configure("stanford2d3d"); P.WORK_HW = (B.EH, B.EW)                    # for seg GT loader path parity
    enc = build_encoder()
    plan = B.build_plan(); B.build_sample_grids(plan)
    cids = [B.coord_map(tp.yaw_deg, tp.pitch_deg) for tp in plan]
    files = data.list_erps("stanford2d3d")
    tr, va = B.split_files(files, B.FOLD)                                        # TR_PANOS default -> ALL
    ntr = sum(p.numel() for p in enc.parameters() if p.requires_grad) / 1e6
    c_out = P.N_CLASS if TASK == "seg" else CH[TASK]
    tag = f"{TASK}_ft_{FT_TARGETS}_r{FT_RANK}_" + (os.path.basename(FT_INIT) if FT_INIT else "fresh")
    print(f"FINETUNE {TASK} fold{B.FOLD} {tag} | LoRA {ntr:.3f}M | tr={len(tr)} va={len(va)} "
          f"eval={B.EH}x{B.EW} ep={B.EPOCHS} decoder={os.environ.get('DECODER','conv')}", flush=True)

    torch.manual_seed(B.SEED)
    head = B.make_head(enc.dim, c_out).to(B.DEVICE)
    t0 = time.time()
    finetune(enc, head, tr, plan)
    r = evaluate(enc, head, va, plan, cids)
    print(f"\n=== FINETUNE {TASK} | {tag} | {time.time()-t0:.0f}s | coverage {r['coverage']*100:.1f}% ===", flush=True)
    if TASK == "depth":
        print(f"  AbsRel={r['AbsRel']:.4f} RMSE={r['RMSE']:.4f} d1={r['d1']*100:.1f} (SI-d1={r['d1_SI']*100:.1f})"
              f"  | vs frozen-probe 0.117/85.9 · fresh-FT 0.104/90.0", flush=True)
    elif TASK == "normal":
        print(f"  mean={r['mean']:.2f} median={r['median']:.2f} <11.25={r['pct_11']:.1f}%  | vs frozen-probe 42.0/50.4", flush=True)
    else:
        print(f"  mIoU={r['mIoU']*100:.2f}  | vs frozen-probe 57.7", flush=True)

    run = runlog.create_run(f"finetune_{tag}_f{B.FOLD}", {
        "task": TASK, "fold": B.FOLD, "ft_init": FT_INIT or "fresh-dinov3", "ft_rank": FT_RANK,
        "ft_targets": FT_TARGETS, "decoder": os.environ.get("DECODER", "conv"), "lora_M": round(ntr, 3),
        "epochs": B.EPOCHS, "lr": FT_LR, "eval_hw": [B.EH, B.EW], "tr_panos": len(tr), "va_panos": len(va),
        "metrics": {k: v for k, v in r.items() if k != "per_class"}})
    torch.save({"head": head.state_dict()}, os.path.join(run, "weights", "head.pt"))
    enc.backbone.save_pretrained(os.path.join(run, "weights", "lora"))
    save_viz(enc, head, va, plan, cids, run, c_out)
    print(f"saved -> {run} (config + weights + viz)", flush=True)


if __name__ == "__main__":
    main()
