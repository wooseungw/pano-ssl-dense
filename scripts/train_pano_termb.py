"""Term B — cross-view masked completion (docs/PANO_WHEREWHAT_SPEC.md).

The live SSL lever after the position pretext was diagnosed null (diag_position_headroom.py):
frozen DINOv3 has POOR cross-view correspondence (ret@1 0.21), so learning to complete a masked
A-patch from the overlapping B-tile's evidence at the warp location injects the B->A distortion
transform the planar prior lacks. TC3-safe: the target is the FROZEN A feature (no erosion).

Per pool item, per overlapping pair (a,b) with WarpField (grid,valid,weight):
  b_ev   = grid_sample(student_full[b], grid)            # LoRA B-evidence warped to A cells
  a_ctx  = student_masked[a]                             # A context (masked cells = mask_token)
  pred   = P(a_ctx, b_ev)                                # zero-init residual on b_ev
  target = frozen_teacher[a]                             # A's own FROZEN feature (de-overlap rule)
  L_comp = obliquity-weighted (1 - cos(pred, target)) over cells masked-in-A AND visible-in-B
  L = L_comp + L_GRAM*gram_anchor(student_masked, frozen) + L_KOLEO*koleo(student CLS-proxy)

Run:  CAP_S3D=300 CAP_S2D=300 EPOCHS=1 CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/train_pano_termb.py
Smoke: SMOKE_STEPS=8 CAP_S3D=8 CAP_S2D=8 CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/train_pano_termb.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_ssl as TS  # noqa: E402  (build_geometry/build_pool/load_erp/render_tiles/erank/ram_avail_gb/DOMAINS)
from encoder import PanoEncoder, normalize_tiles, CrossViewPredictor  # noqa: E402
from losses import cross_view_completion_loss, gram_anchor, koleo  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = TS.MODEL
DEVICE = "cuda"
CKPT = os.environ.get("CKPT_DIR", os.path.join(ROOT, "runs", "ckpt_pano_termb"))
EPOCHS = int(os.environ.get("EPOCHS", 1))
LR = float(os.environ.get("LR", 1e-4))
LORA_RANK = int(os.environ.get("LORA_RANK", 16))
MASK_RATIO = float(os.environ.get("MASK_RATIO", 0.5))
L_GRAM = float(os.environ.get("L_GRAM", 1.0))
L_KOLEO = float(os.environ.get("L_KOLEO", 0.1))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 6))
SMOKE = int(os.environ.get("SMOKE_STEPS", 0))
# FIXES (audit 2026-07-08): obliquity weight was BACKWARDS for a completion objective (it
# down-weighted the oblique cells that carry the informative B->A distortion transform); default
# to UNIFORM so learning concentrates on the hard/informative cells. And anchor the Gram on the
# UNMASKED student (the eval-time representation) to avoid the mask_token/frozen mismatch.
COMP_WEIGHT = os.environ.get("COMP_WEIGHT", "uniform")   # "uniform" (fix) | "obliquity" (old)
GRAM_ON = os.environ.get("GRAM_ON", "full")              # "full" (fix) | "masked" (old)


def block_mask(T, gh, gw, ratio, gen, block=4):
    """Block-wise mask (~ratio) over the (gh,gw) grid -> (T, gh*gw) bool."""
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
    enc = PanoEncoder(model_id=MODEL, lora_rank=LORA_RANK).to(DEVICE).train()
    for n, p in enc.named_parameters():                      # trainable: LoRA + iBOT mask_token
        if "mask_token" in n:
            p.requires_grad = True
    pred = CrossViewPredictor(enc.dim).to(DEVICE).train()
    geom = {k: TS.build_geometry(enc, hf, pc) for k, (hf, pc) in TS.DOMAINS.items()}
    for k, g in geom.items():
        print(f"geom[{k}] hfov={g['hfov']} tiles={len(g['specs'])} pairs={len(g['pairs'])}", flush=True)

    params = list(enc.trainable_parameters()) + list(pred.parameters())
    opt = torch.optim.AdamW(params, lr=LR)
    pool = TS.build_pool()
    n_tr = sum(p.numel() for p in enc.trainable_parameters()) / 1e6
    print(f"TERM-B enc_trainable={n_tr:.3f}M predictor={sum(p.numel() for p in pred.parameters())/1e6:.2f}M "
          f"pool={len(pool)} mask={MASK_RATIO} L_gram={L_GRAM} L_koleo={L_KOLEO} "
          f"comp_w={COMP_WEIGHT} gram_on={GRAM_ON}", flush=True)

    gh = gw = TS.TILE // enc.patch
    N = gh * gw
    g0 = torch.Generator().manual_seed(0)
    total_steps = SMOKE if SMOKE else EPOCHS * len(pool)

    def prep(i):
        f, kind = pool[i]
        try:
            erp = TS.load_erp(f, kind)
        except Exception:
            return None
        return kind, TS.render_tiles(erp, geom[kind]["specs"], geom[kind]["hfov"])

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
            kind, tiles_cpu = item
            g = geom[kind]
            tiles = normalize_tiles(tiles_cpu.to(DEVICE))
            T = tiles.shape[0]
            mask = block_mask(T, gh, gw, MASK_RATIO, g0).to(DEVICE)          # (T, N) bool
            student_full = enc(tiles)                                        # LoRA, unmasked (B-evidence)
            student_masked = enc.forward_masked(tiles, mask)                 # LoRA, masked  (A-context)
            with torch.no_grad():
                frozen = enc.teacher(tiles)                                  # adapter-off target

            l_comp, npairs = 0.0, 0
            for (a, b), (grid, valid, weight) in zip(g["pairs"], g["warps"]):
                m_a = mask[a] & valid                                        # (N,) masked-in-A AND visible-in-B
                if int(m_a.sum()) == 0:
                    continue
                gg = grid.view(1, 1, N, 2)
                b_ev = F.grid_sample(student_full[b:b + 1], gg, mode="bilinear",
                                     align_corners=False)[0, :, 0, :].t()    # (N, D) B evidence at A cells
                a_ctx = student_masked[a].reshape(enc.dim, N).t()            # (N, D)
                tgt = frozen[a].reshape(enc.dim, N).t().detach()             # (N, D) frozen A target
                w_sel = weight[m_a] if COMP_WEIGHT == "obliquity" else torch.ones_like(weight[m_a])
                p_hat = pred(a_ctx[m_a], b_ev[m_a])                          # (n, D)
                l_comp = l_comp + cross_view_completion_loss(p_hat, tgt[m_a], w_sel)
                npairs += 1
            if npairs == 0:
                continue
            l_comp = l_comp / npairs
            l_gram = gram_anchor(student_full if GRAM_ON == "full" else student_masked, frozen)
            l_kol = koleo(student_masked.mean(dim=(2, 3)))
            loss = l_comp + L_GRAM * l_gram + L_KOLEO * l_kol
            opt.zero_grad(); loss.backward(); opt.step()

            for kk, vv in [("comp", l_comp), ("gram", l_gram), ("koleo", l_kol)]:
                agg[kk] = agg.get(kk, 0.0) + float(vv.detach())
            step += 1
            if step % 50 == 0 or (SMOKE and step <= 8):
                er = TS.erank(student_full.detach()); erf = TS.erank(frozen)
                msg = " ".join(f"{k}={v/max(1,min(50,step if step%50 else 50)):.3f}" for k, v in agg.items())
                print(f"ep{ep} step{step}/{total_steps} erank={er:.1f}/{erf:.1f} {msg} "
                      f"({(time.time()-t0)/step:.2f}s/it ram={TS.ram_avail_gb():.0f}GB)", flush=True)
                agg = {}
            if SMOKE and step >= SMOKE:
                break
        if SMOKE and step >= SMOKE:
            break

    os.makedirs(CKPT, exist_ok=True)
    enc.backbone.save_pretrained(CKPT)                          # LoRA adapter (predictor unused at eval)
    torch.save(pred.state_dict(), os.path.join(CKPT, "predictor.pt"))
    print(f"saved adapter -> {CKPT}", flush=True)
    slug = os.environ.get("RUN_SLUG")
    if slug:
        import runlog
        run = runlog.create_run(slug, {"model": MODEL, "mask_ratio": MASK_RATIO, "l_gram": L_GRAM,
            "l_koleo": L_KOLEO, "epochs": EPOCHS, "lr": LR, "lora_rank": LORA_RANK,
            "pool": len(pool), "final_step": step, "ckpt": CKPT})
        enc.backbone.save_pretrained(os.path.join(run, "weights"))
        print(f"saved run -> {run}", flush=True)


if __name__ == "__main__":
    main()
