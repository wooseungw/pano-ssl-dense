"""TC3: Teacher-Code Cross-view Consistency — v2 of the semantic-identity SSL.

M1 post-mortem (docs/SEMANTIC_IDENTITY_SSL.md §10): self-generated sinkhorn codes +
annealed token anchor opened an EROSION channel — cross-view code agreement rose
(ARI 0.495→0.677) by discarding semantic content (purity 0.838→0.730, mIoU −0.05…−0.13).
TC3 keeps the geo recipe fully intact and welds both identities into one term whose
erosion channel is closed BY CONSTRUCTION:

  * targets = FIXED teacher semantic codes: prototypes C_t are k-means centroids of
    FROZEN teacher features (built once, never trained). Blurred features cannot hit
    sharp teacher codes — erosion RAISES this loss instead of lowering it.
  * zero new parameters: student is scored against the same fixed C_t directly on
    backbone F — no projector to absorb the loss (§2.1 propagation by construction).
  * geo recipe intact: warp + token distill at FULL weight (no anneal!) + relational
    + VICReg floor. TC3 adds the semantic-quotient cross-view term on top.
  * init from the geo adapter (runs/ckpt_ssl_lora): warp/distill already settled, TC3
    presses only the semantic-identity axis.

  L = warp + tok + rel + VICReg + w_tc3 · CE( p_stu_A(p), q_tea_B(Hp) )   [symmetric]
      p_stu = softmax(F_stu·C_t / τ_s),  q_tea = softmax(F_tea·C_t / τ_t),  τ_t < τ_s

Success signature (vs geo): D-B code ARI ↑ beyond 0.495 WITHOUT the D-C purity drop
(geo 0.854). Honest null: ties geo (student≈teacher ⇒ small gradients).

Run:   CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/train_ssl_tc3.py
Smoke: SMOKE_STEPS=30 CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/train_ssl_tc3.py
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_ssl as T  # noqa: E402
import train_ssl_m1 as M  # noqa: E402  (bidirectional warp geometry)
from encoder import PanoEncoder, normalize_tiles  # noqa: E402
from losses import code_swap_loss, distill_loss, vicreg_var_cov, warp_equivariance_loss  # noqa: E402

DEVICE = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.environ.get("TC3_CKPT", os.path.join(ROOT, "runs", "ckpt_ssl_tc3"))   # sweep: one dir per config
GEO = os.environ.get("GEO_INIT", os.path.join(ROOT, "runs", "ckpt_ssl_lora"))

EPOCHS = int(os.environ.get("EPOCHS", 2))               # continuation run — shorter than scratch
SMOKE_STEPS = int(os.environ.get("SMOKE_STEPS", 0))
LR = 5e-5                                               # continuation LR (geo trained at 1e-4)
K_PROTO = int(os.environ.get("K_PROTO", 512))
TAU_S = 0.1                                             # student temperature (soft)
TAU_T = float(os.environ.get("TAU_T", 0.05))            # teacher target temperature (sharp)
W_TC3 = float(os.environ.get("W_TC3", 0.3))
GAMMA = 0.04
LOG_EVERY = 10 if SMOKE_STEPS else 50


def build_teacher_prototypes(enc: PanoEncoder, geom: dict, pool: list,
                             n_panos: int = 60, n_tokens: int = 200_000,
                             seed: int = 0) -> torch.Tensor:
    """(K, D) unit-norm k-means centroids of FROZEN teacher features — the fixed
    semantic reference frame. Cached in CKPT so eval and reruns share it."""
    # sweep configs share one prototype set (same teacher+K -> same C_t; removes a variance source)
    cache = os.environ.get("PROTOS", os.path.join(CKPT, "teacher_protos.pt"))
    if os.path.exists(cache):
        c = torch.load(cache, map_location="cpu")["protos"]
        print(f"loaded prototypes {tuple(c.shape)} <- {cache}", flush=True)
        return c.to(DEVICE)
    g = torch.Generator().manual_seed(seed)
    ins = [p for p in pool if p[1] == "in"]
    outs = list(dict.fromkeys(p for p in pool if p[1] == "out"))     # undo x3 oversample
    sample = ([ins[i] for i in torch.randperm(len(ins), generator=g)[:n_panos * 2 // 3]]
              + [outs[i] for i in torch.randperm(len(outs), generator=g)[:n_panos // 3]])
    toks = []
    for f, kind in sample:
        try:
            erp = T.load_erp(f, kind)
        except Exception:
            continue
        gm = geom[kind]
        tiles = normalize_tiles(T.render_tiles(erp, gm["specs"], gm["hfov"]).to(DEVICE))
        with torch.no_grad():
            t = enc.teacher(tiles)                                   # (T,D,Gh,Gw) frozen
        toks.append(F.normalize(t.permute(0, 2, 3, 1).reshape(-1, t.shape[1]), dim=1).cpu())
    x = torch.cat(toks)
    x = x[torch.randperm(x.shape[0], generator=g)[:n_tokens]]
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=K_PROTO, random_state=seed, batch_size=8192,
                         n_init=5, max_iter=300).fit(x.numpy())
    c = F.normalize(torch.from_numpy(km.cluster_centers_).float(), dim=1)
    os.makedirs(CKPT, exist_ok=True)
    torch.save({"protos": c, "k": K_PROTO}, cache)
    print(f"built prototypes {tuple(c.shape)} from {x.shape[0]} teacher tokens -> {cache}", flush=True)
    return c.to(DEVICE)


def code_scores(feat: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
    """(T,D,Gh,Gw) features -> (T,K,Gh,Gw) cosine scores vs the fixed prototypes."""
    return torch.einsum("tdhw,kd->tkhw", F.normalize(feat, dim=1), protos)


def main() -> None:
    torch.manual_seed(0)
    enc = PanoEncoder(model_id=T.MODEL, adapter_path=GEO, adapter_trainable=True).to(DEVICE).train()
    n_tr = sum(p.numel() for p in enc.trainable_parameters())
    assert n_tr > 0, "geo adapter loaded frozen — adapter_trainable flag not honored"
    geom = {k: M.build_geometry_bidir(enc, hf, pc) for k, (hf, pc) in T.DOMAINS.items()}
    pool = T.build_pool()
    protos = build_teacher_prototypes(enc, geom, pool)
    total_steps = SMOKE_STEPS if SMOKE_STEPS else EPOCHS * len(pool)
    print(f"init<-{GEO} trainable={n_tr/1e6:.3f}M pool={len(pool)} K={K_PROTO} "
          f"w_tc3={W_TC3} tau_s/t={TAU_S}/{TAU_T} steps={total_steps}", flush=True)

    opt = torch.optim.AdamW(enc.trainable_parameters(), lr=LR)
    g0 = torch.Generator().manual_seed(0)
    step, t0, agg = 0, time.time(), {}
    done = False
    for ep in range(EPOCHS):
        if done:
            break
        order = torch.randperm(len(pool), generator=g0).tolist()
        for i in order:
            f, kind = pool[i]
            try:
                erp = T.load_erp(f, kind)
            except Exception:
                continue
            g = geom[kind]
            tiles = normalize_tiles(T.render_tiles(erp, g["specs"], g["hfov"]).to(DEVICE))
            student = enc(tiles)
            teacher = enc.teacher(tiles)
            s_stu = code_scores(student, protos)                     # (T,K,Gh,Gw)
            with torch.no_grad():
                q = torch.softmax(code_scores(teacher, protos) / TAU_T, dim=1)

            w_tc3 = W_TC3 * min(1.0, step / max(1, total_steps // 10))   # short ramp-in
            l_warp = student.new_zeros(())
            l_tc3 = student.new_zeros(())
            for (a, b), warp in zip(g["pairs"], g["warps"]):
                l_warp = l_warp + warp_equivariance_loss(student[a:a + 1], student[b:b + 1], *warp)
                l_tc3 = l_tc3 + code_swap_loss(s_stu[a:a + 1], q[b:b + 1], *warp, tau_s=TAU_S)
            npair = len(g["pairs"])
            l_warp, l_tc3 = l_warp / npair, l_tc3 / npair
            tok, rel = distill_loss(student, teacher)
            var, cov = vicreg_var_cov(student, gamma=GAMMA)
            total = l_warp + tok + rel + 25.0 * var + cov + w_tc3 * l_tc3

            opt.zero_grad(); total.backward(); opt.step()
            comps = {"warp": l_warp, "tc3": l_tc3, "tok": tok, "rel": rel,
                     "var": var, "cov": cov, "total": total}
            for kk, vv in comps.items():
                agg[kk] = agg.get(kk, 0.0) + float(vv.detach())
            step += 1
            if step % LOG_EVERY == 0:
                er, ert = T.erank(student.detach()), T.erank(teacher.detach())
                conf = float(torch.softmax(s_stu.detach() / TAU_S, dim=1).max(dim=1).values.mean())
                msg = " ".join(f"{kk}={vv/LOG_EVERY:.3f}" for kk, vv in agg.items())
                print(f"ep{ep} step{step}/{total_steps} w_tc3={w_tc3:.2f} erank={er:.1f}/{ert:.1f} "
                      f"conf={conf:.2f} {msg} ({(time.time()-t0)/step:.2f}s/it)", flush=True)
                agg = {}
            if step >= total_steps:
                done = True
                break

    os.makedirs(CKPT, exist_ok=True)
    enc.backbone.save_pretrained(CKPT)
    print(f"saved adapter -> {CKPT} (prototypes already cached there)", flush=True)


if __name__ == "__main__":
    main()
