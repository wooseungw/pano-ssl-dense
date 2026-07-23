"""PANO-iBOT (Term A): light iBOT-style masked-patch continued-pretrain of DINOv3 on E2P pano tiles.
Design docs/PANO_MIM_DESIGN.md; losses verified in scratchpad/verify_loss.py.

Per tile: block-mask ~45% of the 32x32 patch tokens; the student encodes with the (learnable)
mask_token at masked positions (bool_masked_pos -> no leakage) and predicts, in COSINE-prototype
space, the EMA-teacher's codes of the full tile at those positions; a frozen Gram anchor + KoLeo
prevent collapse/erosion (the two guards that verified as binding).

  L = ibot_loss(student[mask], EMA_teacher[mask])         # masked-patch self-distill (sinkhorn target)
    + L_GRAM · gram_anchor(student, FROZEN)               # dense collapse + erosion guard (verified)
    + L_KOLEO · koleo(student CLS-proxy)                   # dimensional-collapse guard
Student = DINOv3 + LoRA + trainable mask_token; teacher = EMA(student); frozen = adapter-off.
Monitor: erank (the true collapse detector; perplexity is blind — verified).

Run:  CAP_S3D=300 CAP_S2D=300 EPOCHS=1 CUDA_VISIBLE_DEVICES=1 python scripts/train_pano_ibot.py
Smoke: SMOKE_STEPS=8 CAP_S3D=8 CAP_S2D=8 ... python scripts/train_pano_ibot.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anyres_e2p as a2p  # noqa: E402
import train_ssl as TS  # noqa: E402  (reuse load_erp, render_tiles, build_pool, erank, ram_avail_gb, DOMAINS)
from encoder import PanoEncoder, normalize_tiles, ema_update  # noqa: E402
from losses import ibot_loss, gram_anchor, koleo  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEVICE = "cuda"
TILE = 512
CKPT = os.environ.get("CKPT_DIR", os.path.join(ROOT, "ckpt_pano_ibot"))
EPOCHS = int(os.environ.get("EPOCHS", 1))
LR = float(os.environ.get("LR", 1e-4))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 6))
MASK_RATIO = float(os.environ.get("MASK_RATIO", 0.45))
K_PROTO = int(os.environ.get("K_PROTO", 256))
TAU_S, TAU_T = 0.1, float(os.environ.get("TAU_T", 0.05))
L_GRAM = float(os.environ.get("L_GRAM", 1.0))
L_KOLEO = float(os.environ.get("L_KOLEO", 0.1))
EMA_M = float(os.environ.get("EMA_M", 0.996))
TARGET = os.environ.get("TARGET", "proto")           # "feat"=frozen-feature (I-JEPA/data2vec), "proto"=iBOT prototypes
SMOKE = int(os.environ.get("SMOKE_STEPS", 0))
SNAP_EVERY = int(os.environ.get("SNAP_EVERY", 0))    # save adapter snapshot every N epochs (0=off)


def build_specs(hfov, pitches):
    yaws = a2p.make_yaw_centers_closed_loop(hfov, 0.25, start_deg=-180.0)
    return [(y, p) for p in pitches for y in yaws]


GEO = {k: (hf, build_specs(hf, pc)) for k, (hf, pc) in TS.DOMAINS.items()}


def erank_flat(x):                                   # effective rank of (M,D) feats (collapse detector)
    x = x - x.mean(0, keepdim=True)
    ev = torch.linalg.eigvalsh((x.T @ x) / (x.shape[0] - 1)).clamp_min(1e-9)
    p = ev / ev.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def block_mask(T, gh, gw, ratio, gen, block=4):
    """Block-wise mask (~ratio) over the (gh,gw) patch grid -> (T, gh*gw) bool. Contiguous
    `block`x`block` cells (iBOT/BEiT block masking beats scattered for dense features)."""
    bh, bw = gh // block, gw // block
    nb = bh * bw
    k = max(1, int(round(ratio * nb)))
    m = torch.zeros(T, nb, dtype=torch.bool)
    for t in range(T):
        m[t, torch.randperm(nb, generator=gen)[:k]] = True
    m = m.reshape(T, bh, bw).repeat_interleave(block, 1).repeat_interleave(block, 2)
    return m.reshape(T, gh * gw)


def main():
    torch.manual_seed(0)
    enc = PanoEncoder(model_id=MODEL, lora_rank=16).to(DEVICE).train()
    for n, p in enc.named_parameters():                 # trainable: LoRA + the iBOT mask_token
        if "mask_token" in n:
            p.requires_grad = True
    tea = PanoEncoder(model_id=MODEL, lora_rank=16).to(DEVICE)
    tea.load_state_dict(enc.state_dict())               # EMA teacher init = student
    for p in tea.parameters():
        p.requires_grad = False
    tea.eval()

    D = enc.dim
    proj = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 256)).to(DEVICE)
    protos = nn.Linear(256, K_PROTO, bias=False).to(DEVICE)

    def score(x):                                       # (M,D) -> (M,K) COSINE prototype scores
        z = F.normalize(proj(x), dim=-1)
        w = F.normalize(protos.weight, dim=1)
        return z @ w.t()

    params = list(enc.trainable_parameters()) + list(proj.parameters()) + list(protos.parameters())
    opt = torch.optim.AdamW(params, lr=LR)
    pool = TS.build_pool()
    n_tr = sum(p.numel() for p in enc.trainable_parameters()) / 1e6
    print(f"PANO-iBOT enc_trainable={n_tr:.3f}M head={sum(p.numel() for p in proj.parameters())/1e6:.2f}M "
          f"pool={len(pool)} mask={MASK_RATIO} K={K_PROTO} L_gram={L_GRAM} L_koleo={L_KOLEO} ema={EMA_M}", flush=True)

    gh = gw = TILE // enc.patch
    g0 = torch.Generator().manual_seed(0)
    total_steps = SMOKE if SMOKE else EPOCHS * len(pool)

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
        for _ in range(NUM_WORKERS * 2):                # bounded prefetch (RAM-flat)
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
            kind, tiles_cpu = item
            tiles = normalize_tiles(tiles_cpu.to(DEVICE))
            T = tiles.shape[0]
            mask = block_mask(T, gh, gw, MASK_RATIO, g0).to(DEVICE)          # (T, N) bool
            student = enc.forward_masked(tiles, mask)                        # (T,D,gh,gw)
            with torch.no_grad():
                frozen = enc.teacher(tiles)                                  # adapter-off frozen
                teacher = tea(tiles) if TARGET == "proto" else None          # EMA full (proto target only)
            N = gh * gw
            mflat = mask.reshape(-1)
            sflat = student.permute(0, 2, 3, 1).reshape(T * N, D)
            if TARGET == "feat":                                            # predict FROZEN feature at masked
                tfz = frozen.permute(0, 2, 3, 1).reshape(T * N, D)          # patches (no lossy clustering)
                l_ibot = (1.0 - F.cosine_similarity(sflat[mflat], tfz[mflat].detach(), dim=-1)).mean()
            else:                                                           # iBOT learned-prototype codes
                tflat = teacher.permute(0, 2, 3, 1).reshape(T * N, D)
                l_ibot = ibot_loss(score(sflat[mflat]), score(tflat[mflat]), TAU_S, TAU_T)
            l_gram = gram_anchor(student, frozen)
            l_kol = koleo(student.mean(dim=(2, 3)))
            loss = l_ibot + L_GRAM * l_gram + L_KOLEO * l_kol
            opt.zero_grad(); loss.backward(); opt.step()
            if TARGET == "proto":
                ema_update(enc, tea, EMA_M)

            for kk, vv in [("ibot", l_ibot), ("gram", l_gram), ("koleo", l_kol)]:
                agg[kk] = agg.get(kk, 0.0) + vv.item()
            step += 1
            if step % 50 == 0 or (SMOKE and step <= 8):
                svis = student.detach().permute(0, 2, 3, 1).reshape(T * N, D)[~mflat]  # visible only
                er = erank_flat(svis); erf = TS.erank(frozen)
                msg = " ".join(f"{k}={v/max(1,min(50,step)):.3f}" for k, v in agg.items())
                print(f"ep{ep} step{step}/{total_steps} erank={er:.1f}/{erf:.1f} {msg} "
                      f"({(time.time()-t0)/step:.2f}s/it ram={TS.ram_avail_gb():.0f}GB)", flush=True)
                agg = {}
            if SMOKE and step >= SMOKE:
                break
        if SNAP_EVERY and (ep + 1) % SNAP_EVERY == 0 and not SMOKE:
            snap = os.path.join(CKPT, f"ep{ep + 1}")
            os.makedirs(snap, exist_ok=True)
            enc.backbone.save_pretrained(snap)
            print(f"[snap ep{ep + 1}] adapter -> {snap}", flush=True)
        if SMOKE and step >= SMOKE:
            break

    os.makedirs(CKPT, exist_ok=True)
    enc.backbone.save_pretrained(CKPT)                  # LoRA adapter (mask_token unused at eval)
    print(f"saved adapter -> {CKPT}", flush=True)
    slug = os.environ.get("RUN_SLUG")
    if slug:
        import runlog
        run = runlog.create_run(slug, {"model": MODEL, "mask_ratio": MASK_RATIO, "K": K_PROTO,
            "tau_s": TAU_S, "tau_t": TAU_T, "l_gram": L_GRAM, "l_koleo": L_KOLEO, "ema_m": EMA_M,
            "epochs": EPOCHS, "lr": LR, "pool": len(pool), "final_step": step, "ckpt": CKPT})
        enc.backbone.save_pretrained(os.path.join(run, "weights"))
        print(f"saved run -> {run}", flush=True)
        try:                                       # default train-time viz (opt out: TRAIN_VIZ=0)
            import train_viz
            train_viz.emit_train_viz(run, CKPT)
        except Exception as e:
            print(f"train-viz skipped: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
