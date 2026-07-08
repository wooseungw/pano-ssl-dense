"""V1: consolidated 3-role VICReg SSL (docs §2 redesign, user-directed 2026-07-03).

Stops the loss-term pile-up (warp+token+relational+var+cov+tc3+pred) and returns to the
canonical three roles, one term each:
  geometric robustness = INVARIANCE over the E2P overlap (standard augmentation-SSL term)
  anti-collapse        = VARIANCE + COVARIANCE, ACTIVE on an expander (fixes F-3 flaw #2)
  semantic similarity  = deliberately OMITTED for now ("먼저 기하강건성") — added later iff needed

No teacher, no distillation: a fresh LoRA on frozen DINOv3 adapted by pure VICReg. This
is the direct test of the user's thesis — that ACTIVE VICReg anti-collapse (var+cov on an
expander, gamma=1, canonical weights) holds erosion where F-3's dormant floor could not.

  L = 25·inv + 25·var + 1·cov            (canonical VICReg weights)

Run:   CUDA_VISIBLE_DEVICES=<n> conda run -n pano python scripts/train_ssl_vicreg.py
Smoke: SMOKE_STEPS=20 BATCH=2 CUDA_VISIBLE_DEVICES=<n> ... python scripts/train_ssl_vicreg.py
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_ssl as T  # noqa: E402
import train_ssl_m1 as M  # noqa: E402  (bidirectional warp geometry)
import runlog  # noqa: E402
from encoder import Expander, PanoEncoder, normalize_tiles  # noqa: E402
from losses import distill_loss, overlap_invariance, vicreg_vc  # noqa: E402

DEVICE = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.environ.get("VICREG_CKPT", os.path.join(ROOT, "runs", "ckpt_ssl_vicreg"))

EPOCHS = int(os.environ.get("EPOCHS", 3))
SMOKE_STEPS = int(os.environ.get("SMOKE_STEPS", 0))
BATCH = int(os.environ.get("BATCH", 4))                 # panos per optimizer step (grad accum)
WORKERS = int(os.environ.get("NUM_WORKERS", BATCH))
AMP = os.environ.get("AMP", "1") == "1"
LR = 1e-4
PROJ_DIM = int(os.environ.get("PROJ_DIM", 1024))
GAMMA = 1.0                                             # REAL variance target on the expander
L_INV = float(os.environ.get("L_INV", 25.0))           # canonical VICReg weights
L_VAR = float(os.environ.get("L_VAR", 25.0))
L_COV = float(os.environ.get("L_COV", 1.0))
# semantic-similarity role (role 2): distill-to-teacher anchors DINOv3 semantics — the
# PROVEN anti-erosion guard (TC3 kept it=no erosion; M1/F-3 dropped it=erosion). Default
# ON. SEM=none is the clean ablation of the user's thesis "active var+cov alone suffices".
SEM = os.environ.get("SEM", "distill")                 # distill | none
L_SEM = float(os.environ.get("L_SEM", 1.0))
LOG_EVERY = 5 if SMOKE_STEPS else 50

os.environ.setdefault("POOL_PIN", os.path.join(ROOT, "configs", "pool_pin_20260702.tsv"))


def pano_forward(enc, expander, tiles, geom):
    """Forward ONE pano -> (z all-tiles, per-pano invariance, semantic, feat). var/cov are
    NOT computed here — the caller stacks z across the whole BATCH and computes them once,
    so the anti-collapse statistic sees DIFFERENT scenes (canonical batch-level VICReg),
    not one scene's spatially-correlated tiles."""
    x = normalize_tiles(tiles.to(DEVICE))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
        feat = enc(x)                                               # (T,D,gh,gw) student
    ff = feat.float()
    z = expander(ff)                                                # (T,P,gh,gw)
    inv = z.new_zeros(())
    for (a, b), warp in zip(geom["pairs"], geom["warps"]):
        inv = inv + overlap_invariance(z[a:a + 1], z[b:b + 1], *warp)   # geo robustness (per pair)
    inv = inv / len(geom["pairs"])
    sem = z.new_zeros(())
    if SEM == "distill":                                            # semantic-similarity role
        with torch.no_grad():
            teacher = enc.teacher(x).float()
        tok, rel = distill_loss(ff, teacher)                        # anchor DINOv3 semantics
        sem = tok + rel
    return z, inv, sem, feat.detach()


def main() -> None:
    torch.manual_seed(0)
    enc = PanoEncoder(model_id=T.MODEL, lora_rank=16).to(DEVICE).train()   # fresh LoRA, no teacher
    expander = Expander(enc.dim, proj_dim=PROJ_DIM).to(DEVICE).train()
    geom = {k: M.build_geometry_bidir(enc, hf, pc) for k, (hf, pc) in T.DOMAINS.items()}
    pool = T.build_pool()
    n_lora = sum(p.numel() for p in enc.trainable_parameters())
    n_exp = sum(p.numel() for p in expander.parameters())
    total_steps = SMOKE_STEPS if SMOKE_STEPS else EPOCHS * ((len(pool) + BATCH - 1) // BATCH)
    warmup = max(1, total_steps // 10)
    print(f"VICReg-3role lora={n_lora/1e6:.3f}M expander={n_exp/1e6:.2f}M(P={PROJ_DIM}) "
          f"pool={len(pool)} batch={BATCH} w=[inv {L_INV} var {L_VAR} cov {L_COV} sem {L_SEM}] "
          f"sem={SEM} gamma={GAMMA} amp={AMP} steps={total_steps}", flush=True)

    opt = torch.optim.AdamW(list(enc.trainable_parameters()) + list(expander.parameters()), lr=LR)

    def prep(i):
        f, kind = pool[i]
        try:
            erp = T.load_erp(f, kind)
        except Exception:
            return None
        g = geom[kind]
        return T.render_tiles(erp, g["specs"], g["hfov"]), g

    ex = ThreadPoolExecutor(max_workers=WORKERS)
    g0 = torch.Generator().manual_seed(0)
    step, t0, agg, done, last = 0, time.time(), {}, False, None
    pbar = tqdm(total=total_steps, desc="vicreg", mininterval=10, file=sys.stdout, dynamic_ncols=True)
    for ep in range(EPOCHS):
        if done:
            break
        order = torch.randperm(len(pool), generator=g0).tolist()
        for bs in range(0, len(order), BATCH):
            opt.zero_grad()
            warm = min(1.0, step / warmup)
            zs, invs, sems = [], [], []
            for item in ex.map(prep, order[bs:bs + BATCH]):
                if item is None:
                    continue
                tiles, g = item
                z, inv_p, sem_p, feat = pano_forward(enc, expander, tiles, g)
                zs.append(z); invs.append(inv_p); sems.append(sem_p); last = feat
            if not zs:
                continue
            var, cov = vicreg_vc(torch.cat(zs, dim=0), gamma=GAMMA)  # ONCE over the whole BATCH
            inv = torch.stack(invs).mean()
            sem = torch.stack(sems).mean()
            total = warm * L_INV * inv + L_VAR * var + L_COV * cov + L_SEM * sem
            total.backward()
            opt.step()
            step += 1
            for kk, vv in (("inv", inv), ("var", var), ("cov", cov), ("sem", sem)):
                agg[kk] = agg.get(kk, 0.0) + float(vv.detach())
            pbar.update(1); pbar.set_postfix(inv=f"{float(inv.detach()):.3f}", refresh=False)
            if step % LOG_EVERY == 0:
                er = T.erank(last)                          # backbone erank (erosion monitor)
                msg = " ".join(f"{kk}={vv/LOG_EVERY:.3f}" for kk, vv in agg.items())
                print(f"ep{ep} step{step}/{total_steps} warm={warm:.2f} sem={SEM} erank={er:.1f} "
                      f"{msg} ({(time.time()-t0)/step:.2f}s/it)", flush=True)
                agg = {}
            if step >= total_steps:
                done = True
                break
    pbar.close()

    os.makedirs(CKPT, exist_ok=True)
    enc.backbone.save_pretrained(CKPT)
    torch.save({"state_dict": expander.state_dict(), "dim": enc.dim, "proj_dim": PROJ_DIM},
               os.path.join(CKPT, "expander.pt"))
    run = runlog.create_run(f"vicreg_3role_{SEM}", {
        "roles": f"geo-invariance + var + cov + semantic({SEM})", "lora_M": n_lora / 1e6,
        "proj_dim": PROJ_DIM, "weights": {"inv": L_INV, "var": L_VAR, "cov": L_COV, "sem": L_SEM},
        "sem": SEM, "gamma": GAMMA, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "pool": len(pool),
        "pool_pin": os.environ.get("POOL_PIN"), "amp_bf16": AMP, "steps": total_steps})
    enc.backbone.save_pretrained(os.path.join(run, "weights", "adapter"))
    torch.save({"state_dict": expander.state_dict(), "dim": enc.dim, "proj_dim": PROJ_DIM},
               os.path.join(run, "weights", "expander.pt"))
    print(f"saved -> {CKPT} and {run}", flush=True)
    try:                                           # default train-time viz (opt out: TRAIN_VIZ=0)
        import train_viz
        train_viz.emit_train_viz(run, CKPT)
    except Exception as e:
        print(f"train-viz skipped: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
