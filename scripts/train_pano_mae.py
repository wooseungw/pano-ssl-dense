"""PANO-MAE — a REAL information-injecting MIM: unfreeze last-N DINOv3 blocks + raw-PIXEL reconstruction
+ L2-SP anti-forgetting + warmup/cosine, on E2P pano tiles.

Motivation (docs/FAILURE_ANALYSIS.md + this session): every prior SSL here was either consistency (I(T;Y|F)=0)
or a masked-prediction whose TARGET was the frozen feature (PANO-iBOT, capped at frozen). This is the honest
test the user asked for: a masked-prediction that reconstructs a signal F LACKS — raw PIXELS (SimMIM/MAE) — with
real capacity (unfreeze last-N blocks, not just LoRA), run with a proper SSL schedule, while preventing
catastrophic forgetting WITHOUT capping via L2-SP (weight-space anchor to the original last-N weights; freezing
blocks 0..(11-N) structurally preserves the general features).

  L = L1( pixel_head(student[mask]) , per-patch-normalized input pixels[mask] )   # SimMIM raw-pixel recon
    + L2SP_LAMBDA * || theta_unfrozen - theta_unfrozen_original ||^2               # anti-forgetting, no cap

Run: CAP_S3D=1500 CAP_S2D=300 EPOCHS=30 UNFREEZE_LAST=4 CUDA_VISIBLE_DEVICES=0 \
     OPENCV_IO_ENABLE_OPENEXR=1 conda run -n pano python scripts/train_pano_mae.py
Knobs: UNFREEZE_LAST(4) MASK_RATIO(0.6) LR(1e-4) L2SP_LAMBDA(1e-3) EPOCHS(30) WARMUP_FRAC(0.05)
       CKPT_EVERY(10) CAP_S3D/CAP_S2D. SMOKE_STEPS for a dry run.
"""
from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_ssl as TS  # data pipeline: build_pool / load_erp / render_tiles / DOMAINS  # noqa: E402
import anyres_e2p as a2p  # noqa: E402  (for GEO specs)
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEVICE = "cuda"
TILE = 512
CKPT = os.environ.get("CKPT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "ckpt_pano_mae"))
EPOCHS = int(os.environ.get("EPOCHS", 30))
UNFREEZE_LAST = int(os.environ.get("UNFREEZE_LAST", 4))
MASK_RATIO = float(os.environ.get("MASK_RATIO", 0.6))
LR = float(os.environ.get("LR", 1e-4))
L2SP_LAMBDA = float(os.environ.get("L2SP_LAMBDA", 1e-3))
WARMUP_FRAC = float(os.environ.get("WARMUP_FRAC", 0.05))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", 10))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 6))
SMOKE = int(os.environ.get("SMOKE_STEPS", 0))


def build_specs(hfov, pitches):
    yaws = a2p.make_yaw_centers_closed_loop(hfov, 0.25, start_deg=-180.0)
    return [(y, p) for p in pitches for y in yaws]


GEO = {k: (hf, build_specs(hf, pc)) for k, (hf, pc) in TS.DOMAINS.items()}


def find_block_list(backbone):
    """Return (name, ModuleList) of the transformer blocks (the longest ModuleList of layers)."""
    best = (None, None)
    for n, m in backbone.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) >= 6:
            if best[1] is None or len(m) > len(best[1]):
                best = (n, m)
    return best


def block_mask(T, gh, gw, ratio, gen, block=4):
    bh, bw = gh // block, gw // block
    nb = bh * bw
    k = max(1, int(round(ratio * nb)))
    m = torch.zeros(T, nb, dtype=torch.bool)
    for t in range(T):
        m[t, torch.randperm(nb, generator=gen)[:k]] = True
    return m.reshape(T, bh, bw).repeat_interleave(block, 1).repeat_interleave(block, 2).reshape(T, gh * gw)


def patchify(x, p):                                  # (T,3,H,W) in [0,1] -> (T, gh*gw, 3*p*p)
    Tn, C, H, W = x.shape
    gh, gw = H // p, W // p
    return x.reshape(Tn, C, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).reshape(Tn, gh * gw, C * p * p)


def main():
    torch.manual_seed(0)
    enc = PanoEncoder(model_id=MODEL, lora_rank=0).to(DEVICE)     # fully frozen backbone
    bname, blocks = find_block_list(enc.backbone)
    if blocks is None:
        raise RuntimeError("could not locate transformer block ModuleList")
    n_blocks = len(blocks)
    unfreeze_idx = set(range(n_blocks - UNFREEZE_LAST, n_blocks))
    for i in unfreeze_idx:
        for p in blocks[i].parameters():
            p.requires_grad = True
    for n, p in enc.backbone.named_parameters():                 # learnable mask token
        if "mask_token" in n:
            p.requires_grad = True
    enc.train()

    D, patch = enc.dim, enc.patch
    pix_head = nn.Linear(D, patch * patch * 3).to(DEVICE)

    trainable = [p for p in enc.parameters() if p.requires_grad] + list(pix_head.parameters())
    l2sp_pairs = [(p, p.detach().clone()) for p in enc.parameters() if p.requires_grad]  # anti-forget anchor
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.05)
    pool = TS.build_pool()
    total_steps = SMOKE if SMOKE else EPOCHS * len(pool)
    warmup = max(1, int(WARMUP_FRAC * total_steps))
    n_enc = sum(p.numel() for p in enc.parameters() if p.requires_grad) / 1e6
    print(f"PANO-MAE: blocks='{bname}'({n_blocks}), unfreeze last {UNFREEZE_LAST} -> {sorted(unfreeze_idx)} | "
          f"enc_trainable={n_enc:.2f}M + pixhead | pool={len(pool)} mask={MASK_RATIO} lr={LR} l2sp={L2SP_LAMBDA} "
          f"epochs={EPOCHS} warmup={warmup}/{total_steps}", flush=True)

    gh = gw = TILE // patch
    g0 = torch.Generator().manual_seed(0)

    def prep(i):
        f, kind = pool[i]
        try:
            erp = TS.load_erp(f, kind)
        except Exception:
            return None
        hf, specs = GEO[kind]
        return kind, TS.render_tiles(erp, specs, hf)

    step, t0, agg = 0, time.time(), {}
    ex = ThreadPoolExecutor(max_workers=NUM_WORKERS)
    for ep in range(EPOCHS):
        order = torch.randperm(len(pool), generator=g0).tolist()
        it = iter(order)
        inflight = deque()
        for _ in range(NUM_WORKERS * 2):
            try:
                inflight.append(ex.submit(prep, next(it)))
            except StopIteration:
                break
        while inflight:
            fut = inflight.popleft()
            try:
                inflight.append(ex.submit(prep, next(it)))
            except StopIteration:
                pass
            item = fut.result()
            if item is None:
                continue
            _, tiles_cpu = item                                  # (T,3,TILE,TILE) in [0,1]
            tiles01 = tiles_cpu.to(DEVICE)
            Tn = tiles01.shape[0]
            mask = block_mask(Tn, gh, gw, MASK_RATIO, g0).to(DEVICE)   # (T, N)
            student = enc.forward_masked(normalize_tiles(tiles01), mask)     # (T,D,gh,gw)
            N = gh * gw
            pred = pix_head(student.permute(0, 2, 3, 1).reshape(Tn, N, D))   # (T,N,3pp)
            with torch.no_grad():
                targ = patchify(tiles01, patch)                             # (T,N,3pp)
                targ = (targ - targ.mean(-1, keepdim=True)) / (targ.std(-1, keepdim=True) + 1e-6)
            mflat = mask.bool()
            recon = F.l1_loss(pred[mflat], targ[mflat])
            l2sp = sum((p - o).pow(2).sum() for p, o in l2sp_pairs)
            loss = recon + L2SP_LAMBDA * l2sp

            for ggrp in opt.param_groups:                                   # warmup + cosine
                ggrp["lr"] = LR * (step / warmup if step < warmup else
                                   0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))))
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            agg["recon"] = agg.get("recon", 0.0) + recon.item()
            agg["l2sp"] = agg.get("l2sp", 0.0) + float(l2sp)
            if step % 50 == 0:
                m = min(50, step)
                print(f"ep{ep} step{step}/{total_steps} lr={opt.param_groups[0]['lr']:.2e} "
                      f"recon={agg['recon']/m:.4f} l2sp={agg['l2sp']/m:.1f} "
                      f"({(time.time()-t0)/step:.2f}s/it)", flush=True)
                agg = {}
            if SMOKE and step >= SMOKE:
                break
        if SMOKE and step >= SMOKE:
            break
        if CKPT_EVERY and (ep + 1) % CKPT_EVERY == 0 and not SMOKE:
            d = f"{CKPT}_ep{ep+1}"
            os.makedirs(d, exist_ok=True)
            enc.backbone.save_pretrained(d)
            print(f"[ckpt] saved -> {d}", flush=True)

    os.makedirs(CKPT, exist_ok=True)
    enc.backbone.save_pretrained(CKPT)
    print(f"saved final -> {CKPT}", flush=True)


if __name__ == "__main__":
    main()
