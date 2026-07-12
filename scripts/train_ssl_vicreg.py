"""Panorama VICReg: safe local, global, sub-token, and geometry-preserving SSL.

Each representation scale keeps the canonical VICReg roles: paired-view invariance plus
active variance and covariance regularization. Local positives use complete-patch spherical
footprints, RGB confidence, cycle-checked bidirectional warps, and independent appearance
views. A frozen-teacher anchor preserves the strong DINOv3 semantics while an auxiliary
geometry readout prevents the invariant branch from discarding latitude/direction structure.

  L = local-VICReg + global/tile-VICReg + sub-token-VICReg
      + geometry-readout + frozen-semantic preservation

Run:   CUDA_VISIBLE_DEVICES=<n> conda run -n pano python scripts/train_ssl_vicreg.py
Smoke: SMOKE_STEPS=20 BATCH=2 CUDA_VISIBLE_DEVICES=<n> ... python scripts/train_ssl_vicreg.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_ssl as T  # noqa: E402
import train_ssl_m1 as M  # noqa: E402  (bidirectional warp geometry)
import geometry as G  # noqa: E402
import runlog  # noqa: E402
from encoder import (Expander, GeometryHead, GlobalExpander, PanoEncoder,  # noqa: E402
                     SubtokenExpander, normalize_tiles)
from losses import (distill_loss, overlap_invariance, vicreg_pair_invariance,  # noqa: E402
                    vicreg_vc, vicreg_vc_vectors)

DEVICE = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.environ.get("VICREG_CKPT", os.path.join(ROOT, "runs", "ckpt_ssl_vicreg"))

EPOCHS = int(os.environ.get("EPOCHS", 300))
SMOKE_STEPS = int(os.environ.get("SMOKE_STEPS", 0))
BATCH = int(os.environ.get("BATCH", 4))                 # panos per optimizer step (grad accum)
WORKERS = int(os.environ.get("NUM_WORKERS", BATCH))
AMP = os.environ.get("AMP", "1") == "1"
LR = 1e-4
PROJ_DIM = int(os.environ.get("PROJ_DIM", 1024))
GLOBAL_DIM = int(os.environ.get("GLOBAL_DIM", 256))
SUB_DIM = int(os.environ.get("SUB_DIM", 128))
GAMMA = 1.0                                             # REAL variance target on the expander
L_INV = float(os.environ.get("L_INV", 25.0))           # canonical VICReg weights
L_VAR = float(os.environ.get("L_VAR", 25.0))
L_COV = float(os.environ.get("L_COV", 1.0))
# semantic-similarity role (role 2): distill-to-teacher anchors DINOv3 semantics — the
# PROVEN anti-erosion guard (TC3 kept it=no erosion; M1/F-3 dropped it=erosion). Default
# ON. SEM=none is the clean ablation of the user's thesis "active var+cov alone suffices".
SEM = os.environ.get("SEM", "distill")                 # distill | none
L_SEM = float(os.environ.get("L_SEM", 1.0))
L_GLOBAL = float(os.environ.get("L_GLOBAL", 0.25))
L_SUB = float(os.environ.get("L_SUB", 0.25))
L_GEO = float(os.environ.get("L_GEO", 0.1))
PHOTO_SIGMA = float(os.environ.get("PHOTO_SIGMA", 0.15))
YAW_ROLL = os.environ.get("YAW_ROLL", "1") == "1"
FOV_JITTER = float(os.environ.get("FOV_JITTER", 5.0))
SKIP_SAVE = os.environ.get("SKIP_SAVE", "0") == "1"
LOG_EVERY = 5 if SMOKE_STEPS else 50
VAL_PANOS = int(os.environ.get("VAL_PANOS", 16))
VAL_EVERY = int(os.environ.get("VAL_EVERY", 5))
VAL_VIZ = os.environ.get("VAL_VIZ", "1") == "1"
VIZ_EVERY = int(os.environ.get("VIZ_EVERY", VAL_EVERY))
VIZ_PANOS = int(os.environ.get("VIZ_PANOS", 1))
VIZ_ROOT = os.environ.get("VIZ_ROOT", os.path.join(CKPT, "viz"))
EARLY_STOP = os.environ.get("EARLY_STOP", "1") == "1"
EARLY_PATIENCE = int(os.environ.get("EARLY_PATIENCE", 5))
EARLY_MIN_DELTA = float(os.environ.get("EARLY_MIN_DELTA", 1e-3))
EARLY_EMA_ALPHA = float(os.environ.get("EARLY_EMA_ALPHA", 0.8))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 1))
RESUME = os.environ.get("RESUME", "")                  # dir with adapter/ + *.pt to continue from

os.environ.setdefault("POOL_PIN", os.path.join(ROOT, "configs", "pool_pin_20260702.tsv"))


def photometric_augment(tiles: torch.Tensor, seed: int) -> torch.Tensor:
    """Independent per-tile appearance augmentation; geometry remains unchanged."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    t = tiles.clone()
    n = t.shape[0]
    brightness = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, generator=gen)
    contrast = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, generator=gen)
    saturation = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, generator=gen)
    t = t * brightness
    mean = t.mean(dim=(2, 3), keepdim=True)
    t = (t - mean) * contrast + mean
    gray = t.mean(dim=1, keepdim=True)
    t = (t - gray) * saturation + gray
    if float(torch.rand((), generator=gen)) < 0.5:
        t = F.avg_pool2d(t, kernel_size=3, stride=1, padding=1)
    return t.clamp(0.0, 1.0)


def geometry_targets(specs, hfov: float, patch: int) -> torch.Tensor:
    """Per-token [sin(lat), cos(lat), sin(relative-lon), cos(relative-lon)]."""
    targets = []
    for yaw, pitch in specs:
        lat, dlon, _, _ = G.tile_position_labels(
            T.ERP_H, T.ERP_W, yaw, pitch, hfov, T.TILE, patch)
        lat = np.deg2rad(lat)
        dlon = np.deg2rad(dlon)
        targets.append(np.stack([np.sin(lat), np.cos(lat),
                                 np.sin(dlon), np.cos(dlon)], axis=0))
    return torch.from_numpy(np.stack(targets).astype(np.float32))


@torch.no_grad()
def confidence_warp(tiles: torch.Tensor, a: int, b: int, warp, gh: int, gw: int):
    """Down-weight mixed/boundary positives whose patch-average RGB does not correspond."""
    grid, valid, weight = warp
    rgb = F.adaptive_avg_pool2d(tiles, (gh, gw))
    g = grid.view(1, 1, gh * gw, 2)
    rb = F.grid_sample(rgb[b:b + 1], g, mode="bilinear", align_corners=False)[:, :, 0, :]
    ra = rgb[a:a + 1].reshape(1, 3, gh * gw)
    err = (ra - rb).abs().mean(dim=1).reshape(-1)
    confidence = torch.exp(-0.5 * (err / max(PHOTO_SIGMA, 1e-6)) ** 2)
    return grid, valid, weight * confidence


def pano_forward(enc, expander, global_expander, sub_expander, geo_head,
                 tiles, geom, seed: int):
    """Forward two augmented views of one panorama and return all VICReg roles."""
    raw = tiles.to(DEVICE)
    x1 = normalize_tiles(photometric_augment(tiles, seed * 2 + 1).to(DEVICE))
    x2 = normalize_tiles(photometric_augment(tiles, seed * 2 + 2).to(DEVICE))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
        feat1 = enc(x1)
        feat2 = enc(x2)
    ff1, ff2 = feat1.float(), feat2.float()
    z1, z2 = expander(ff1), expander(ff2)
    sub1, sub2 = sub_expander(ff1), sub_expander(ff2)
    glob1, glob2 = global_expander(ff1), global_expander(ff2)

    gh, gw = z1.shape[-2:]
    inv = z1.new_zeros(())
    sub_inv = z1.new_zeros(())
    for pair_idx, ((a, b), warp) in enumerate(zip(geom["pairs"], geom["warps"])):
        dynamic_warp = confidence_warp(raw, a, b, warp, gh, gw)
        # Cross the independently augmented branches in both directions. The geometry list
        # itself also contains A->B and B->A, preventing a fixed target-view bias.
        inv = inv + 0.5 * (
            overlap_invariance(z1[a:a + 1], z2[b:b + 1], *dynamic_warp)
            + overlap_invariance(z2[a:a + 1], z1[b:b + 1], *dynamic_warp))
        sub_warp = geom["sub_warps"][pair_idx]
        sub_inv = sub_inv + 0.5 * (
            overlap_invariance(sub1[a:a + 1], sub2[b:b + 1], *sub_warp)
            + overlap_invariance(sub2[a:a + 1], sub1[b:b + 1], *sub_warp))
    inv = inv / len(geom["pairs"])
    sub_inv = sub_inv / len(geom["pairs"])

    global_inv = vicreg_pair_invariance(glob1, glob2)
    geo_target = geom["geo_targets"]
    geo = 0.5 * (F.smooth_l1_loss(geo_head(ff1), geo_target)
                 + F.smooth_l1_loss(geo_head(ff2), geo_target))
    sem = z1.new_zeros(())
    if SEM == "distill":                                            # semantic-similarity role
        with torch.no_grad():
            teacher1 = enc.teacher(x1).float()
            teacher2 = enc.teacher(x2).float()
        tok1, rel1 = distill_loss(ff1, teacher1)
        tok2, rel2 = distill_loss(ff2, teacher2)
        sem = 0.5 * (tok1 + rel1 + tok2 + rel2)
    return {"z": (z1, z2), "sub": (sub1, sub2), "global": (glob1, glob2),
            "inv": inv, "sub_inv": sub_inv, "global_inv": global_inv,
            "geo": geo, "sem": sem, "feat": feat1.detach()}


def batch_objective(outputs, warm: float):
    """Combine per-panorama outputs and return total plus every logged component."""
    zs, subs, globals_ = [], [], []
    for out in outputs:
        zs.extend(out["z"])
        subs.extend(out["sub"])
        globals_.extend(out["global"])
    var, cov = vicreg_vc(torch.cat(zs, dim=0), gamma=GAMMA)
    sub_var, sub_cov = vicreg_vc(torch.cat(subs, dim=0), gamma=GAMMA)
    global_var, global_cov = vicreg_vc_vectors(torch.cat(globals_, dim=0), gamma=GAMMA)
    inv = torch.stack([out["inv"] for out in outputs]).mean()
    sub_inv = torch.stack([out["sub_inv"] for out in outputs]).mean()
    global_inv = torch.stack([out["global_inv"] for out in outputs]).mean()
    geo = torch.stack([out["geo"] for out in outputs]).mean()
    sem = torch.stack([out["sem"] for out in outputs]).mean()
    local_loss = warm * L_INV * inv + L_VAR * var + L_COV * cov
    sub_loss = warm * L_INV * sub_inv + L_VAR * sub_var + L_COV * sub_cov
    global_loss = (warm * L_INV * global_inv
                   + L_VAR * global_var + L_COV * global_cov)
    total = (local_loss + L_SUB * sub_loss + L_GLOBAL * global_loss
             + L_GEO * geo + L_SEM * sem)
    metrics = {
        "total": total, "local": local_loss, "sub": sub_loss, "global": global_loss,
        "inv": inv, "var": var, "cov": cov,
        "sub_inv": sub_inv, "sub_var": sub_var, "sub_cov": sub_cov,
        "global_inv": global_inv, "global_var": global_var, "global_cov": global_cov,
        "geo": geo, "sem": sem,
    }
    return total, metrics


def main() -> None:
    torch.manual_seed(0)
    enc = PanoEncoder(model_id=T.MODEL, lora_rank=16,
                      adapter_path=(os.path.join(RESUME, "adapter") if RESUME else None),
                      adapter_trainable=bool(RESUME)).to(DEVICE).train()
    expander = Expander(enc.dim, proj_dim=PROJ_DIM).to(DEVICE).train()
    global_expander = GlobalExpander(enc.dim, proj_dim=GLOBAL_DIM).to(DEVICE).train()
    sub_expander = SubtokenExpander(enc.dim, proj_dim=SUB_DIM).to(DEVICE).train()
    geo_head = GeometryHead(enc.dim).to(DEVICE).train()
    geom = {}
    for kind, (hfov, pitches) in T.DOMAINS.items():
        fovs = [hfov] if FOV_JITTER <= 0 else [hfov - FOV_JITTER, hfov, hfov + FOV_JITTER]
        geom[kind] = []
        for aug_fov in fovs:
            g = M.build_geometry_bidir(
                enc, aug_fov, pitches, footprint_safe=True, sub_patch=enc.patch // 2)
            g["geo_targets"] = geometry_targets(g["specs"], aug_fov, enc.patch).to(DEVICE)
            geom[kind].append(g)
    full_pool = T.build_pool()
    split_gen = torch.Generator().manual_seed(1729)
    split_order = torch.randperm(len(full_pool), generator=split_gen).tolist()
    requested_val = min(VAL_PANOS, 2) if SMOKE_STEPS else VAL_PANOS
    n_val = min(max(0, requested_val), max(0, len(full_pool) - 1))
    val_ids = set(split_order[:n_val])
    val_pool = [item for i, item in enumerate(full_pool) if i in val_ids]
    pool = [item for i, item in enumerate(full_pool) if i not in val_ids]
    n_lora = sum(p.numel() for p in enc.trainable_parameters())
    aux_modules = (expander, global_expander, sub_expander, geo_head)
    n_exp = sum(p.numel() for module in aux_modules for p in module.parameters())
    total_steps = SMOKE_STEPS if SMOKE_STEPS else EPOCHS * ((len(pool) + BATCH - 1) // BATCH)
    warmup = max(1, total_steps // 10)
    print(f"Pano-VICReg lora={n_lora/1e6:.3f}M auxiliary={n_exp/1e6:.2f}M "
          f"(local={PROJ_DIM} global={GLOBAL_DIM} sub={SUB_DIM}) "
          f"pool={len(pool)} val={len(val_pool)} batch={BATCH} "
          f"w=[inv {L_INV} var {L_VAR} cov {L_COV} sem {L_SEM}] "
          f"branch=[global {L_GLOBAL} sub {L_SUB} geo {L_GEO}] sem={SEM} gamma={GAMMA} "
          f"footprint=safe yaw_roll={YAW_ROLL} fov_jitter=±{FOV_JITTER:g} "
          f"amp={AMP} steps={total_steps} early=[{EARLY_STOP} patience={EARLY_PATIENCE} "
          f"min_delta={EARLY_MIN_DELTA:g} ema={EARLY_EMA_ALPHA:g}]", flush=True)

    aux_params = [p for module in aux_modules for p in module.parameters()]
    opt = torch.optim.AdamW(list(enc.trainable_parameters()) + aux_params, lr=LR)

    start_epoch, resume_step, resume_best_ema, resume_plateau = 0, 0, None, 0
    if RESUME:
        aux = torch.load(os.path.join(RESUME, "vicreg_auxiliary.pt"), map_location=DEVICE)
        expander.load_state_dict(aux["local"])
        global_expander.load_state_dict(aux["global"])
        sub_expander.load_state_dict(aux["sub"])
        geo_head.load_state_dict(aux["geometry"])
        tr = torch.load(os.path.join(RESUME, "trainer.pt"), map_location=DEVICE)
        opt.load_state_dict(tr["optimizer"])
        start_epoch, resume_step = int(tr["epoch"]), int(tr["step"])
        resume_best_ema, resume_plateau = tr["best_ema"], int(tr["plateau_count"])
        print(f"RESUME {RESUME}: start_epoch={start_epoch} step={resume_step} "
              f"best_ema={resume_best_ema} plateau={resume_plateau}", flush=True)

    def prep(job):
        entry, sample_id, aug_epoch = job
        f, kind = entry
        try:
            erp = T.load_erp(f, kind)
        except Exception:
            return None
        if YAW_ROLL:
            # Circular ERP roll is a lossless random yaw augmentation. The fixed tile geometry
            # remains valid; only which scene content appears under each ray changes.
            shift = int((sample_id * 2654435761 + aug_epoch * 2246822519) % T.ERP_W)
            erp = np.roll(erp, shift, axis=1)
        variants = geom[kind]
        g = variants[(sample_id + aug_epoch) % len(variants)]
        seed = sample_id + aug_epoch * 1000003
        return T.render_tiles(erp, g["specs"], g["hfov"]), g, seed

    ex = ThreadPoolExecutor(max_workers=WORKERS)
    g0 = torch.Generator().manual_seed(0)
    modules = (enc, expander, global_expander, sub_expander, geo_head)

    def validate() -> dict:
        if not val_pool:
            return {}
        states = [module.training for module in modules]
        for module in modules:
            module.eval()
        sums, count = {}, 0
        with torch.no_grad():
            for start in range(0, len(val_pool), BATCH):
                entries = val_pool[start:start + BATCH]
                jobs = [(entry, 10000000 + start + j, 0) for j, entry in enumerate(entries)]
                outputs = []
                for item in ex.map(prep, jobs):
                    if item is None:
                        continue
                    tiles, g, seed = item
                    outputs.append(pano_forward(
                        enc, expander, global_expander, sub_expander, geo_head,
                        tiles, g, seed))
                if not outputs:
                    continue
                _, metrics = batch_objective(outputs, warm=1.0)
                weight = len(outputs)
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.0) + float(value) * weight
                count += weight
        for module, was_training in zip(modules, states):
            module.train(was_training)
        return {key: value / max(count, 1) for key, value in sums.items()}

    def auxiliary_state() -> dict:
        return {"local": expander.state_dict(), "global": global_expander.state_dict(),
                "sub": sub_expander.state_dict(), "geometry": geo_head.state_dict(),
                "dim": enc.dim, "proj_dim": PROJ_DIM,
                "global_dim": GLOBAL_DIM, "sub_dim": SUB_DIM}

    def save_snapshot(name: str, epoch: int, best_ema, plateau_count: int) -> None:
        if SKIP_SAVE:
            return
        root = os.path.join(CKPT, name)
        os.makedirs(root, exist_ok=True)
        enc.backbone.save_pretrained(os.path.join(root, "adapter"))
        torch.save(auxiliary_state(), os.path.join(root, "vicreg_auxiliary.pt"))
        torch.save({"epoch": epoch, "step": step, "optimizer": opt.state_dict(),
                    "best_ema": best_ema, "plateau_count": plateau_count},
                   os.path.join(root, "trainer.pt"))

    pca_bases, viz_history = {}, []

    def pca_image(feat: torch.Tensor, basis) -> np.ndarray:
        """Project one (D,H,W) feature map through a frozen-teacher PCA color frame."""
        mean, comps, lo, hi = basis
        d, gh, gw = feat.shape
        flat = feat.permute(1, 2, 0).reshape(-1, d)
        rgb = ((flat - mean) @ comps - lo) / (hi - lo).clamp_min(1e-6)
        return rgb.clamp(0, 1).reshape(gh, gw, 3).cpu().numpy()

    def overlap_cosine(feat: torch.Tensor, pair, warp) -> np.ndarray:
        a, b = pair
        grid, valid, _ = warp
        _, d, gh, gw = feat.shape
        g = grid.view(1, 1, gh * gw, 2)
        fb = F.grid_sample(feat[b:b + 1], g, mode="bilinear",
                           align_corners=False)[:, :, 0, :]
        fa = feat[a:a + 1].reshape(1, d, gh * gw)
        cosine = F.cosine_similarity(fa, fb, dim=1).reshape(gh, gw)
        cosine = cosine.masked_fill(~valid.reshape(gh, gw), float("nan"))
        return cosine.cpu().numpy()

    def emit_validation_viz(epoch: int, val_metrics: dict, ema_value) -> None:
        """Fixed-sample encoder diagnostics under a frozen PCA basis."""
        if not VAL_VIZ or not val_pool or (SKIP_SAVE and "VIZ_ROOT" not in os.environ):
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        states = [module.training for module in modules]
        for module in modules:
            module.eval()
        epoch_dir = os.path.join(VIZ_ROOT, f"epoch_{epoch:04d}")
        os.makedirs(epoch_dir, exist_ok=True)
        history_row = None
        with torch.no_grad():
            for sample_idx, entry in enumerate(val_pool[:max(1, VIZ_PANOS)]):
                item = prep((entry, 20000000 + sample_idx, 0))
                if item is None:
                    continue
                tiles, g, _ = item
                x = normalize_tiles(tiles.to(DEVICE))
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                    student = enc(x).float()
                    frozen = enc.teacher(x).float()
                flat_frozen = frozen.permute(0, 2, 3, 1).reshape(-1, enc.dim)
                if sample_idx not in pca_bases:
                    mean = flat_frozen.mean(0, keepdim=True)
                    _, _, vectors = torch.pca_lowrank(flat_frozen - mean, q=3, center=False)
                    comps = vectors[:, :3]
                    projected = (flat_frozen - mean) @ comps
                    lo = torch.quantile(projected, 0.02, dim=0)
                    hi = torch.quantile(projected, 0.98, dim=0)
                    pca_bases[sample_idx] = (mean, comps, lo, hi)
                basis = pca_bases[sample_idx]

                pair_idx = 0
                pair, warp = g["pairs"][pair_idx], g["warps"][pair_idx]
                a, b = pair
                frozen_cos = overlap_cosine(frozen, pair, warp)
                student_cos = overlap_cosine(student, pair, warp)
                drift = 1.0 - F.cosine_similarity(student, frozen, dim=1)
                norm = student.norm(dim=1)
                input_a = tiles[a].permute(1, 2, 0).numpy()
                input_b = tiles[b].permute(1, 2, 0).numpy()

                fig, axes = plt.subplots(3, 4, figsize=(15, 11))
                panels = [
                    (input_a, "input tile A", None, None),
                    (input_b, "input tile B", None, None),
                    (pca_image(frozen[a], basis), "frozen PCA A (fixed basis)", None, None),
                    (pca_image(frozen[b], basis), "frozen PCA B (fixed basis)", None, None),
                    (pca_image(student[a], basis), "student PCA A", None, None),
                    (pca_image(student[b], basis), "student PCA B", None, None),
                    (drift[a].cpu().numpy(), "semantic drift A: 1-cos(student,frozen)", "magma", (0, 0.3)),
                    (drift[b].cpu().numpy(), "semantic drift B: 1-cos(student,frozen)", "magma", (0, 0.3)),
                    (frozen_cos, "frozen overlap cosine", "viridis", (0, 1)),
                    (student_cos, "student overlap cosine", "viridis", (0, 1)),
                    (norm[a].cpu().numpy(), "student feature norm A", "turbo", None),
                    (norm[b].cpu().numpy(), "student feature norm B", "turbo", None),
                ]
                for ax, (image, title, cmap, limits) in zip(axes.flat, panels):
                    kwargs = {"interpolation": "nearest"}
                    if cmap:
                        kwargs["cmap"] = cmap
                    if limits:
                        kwargs["vmin"], kwargs["vmax"] = limits
                    im = ax.imshow(image, **kwargs)
                    ax.set_title(title, fontsize=9)
                    ax.axis("off")
                    if cmap:
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                valid_frozen = frozen_cos[np.isfinite(frozen_cos)]
                valid_student = student_cos[np.isfinite(student_cos)]
                flat_student = student.permute(0, 2, 3, 1).reshape(-1, enc.dim)
                centered = flat_student - basis[0]
                top3_ratio = float(((centered @ basis[1]).var(0).sum()
                                    / centered.var(0).sum().clamp_min(1e-8)).cpu())
                erank_student, erank_frozen = T.erank(student), T.erank(frozen)
                drift_mean = float(drift.mean().cpu())
                mean_frozen = float(valid_frozen.mean())
                mean_student = float(valid_student.mean())
                fig.suptitle(
                    f"epoch {epoch} | overlap cos {mean_frozen:.3f}→{mean_student:.3f} | "
                    f"erank {erank_frozen:.1f}→{erank_student:.1f} | drift {drift_mean:.3f} | "
                    f"PCA3 ratio {top3_ratio:.3f}", fontsize=12)
                fig.tight_layout(rect=(0, 0, 1, 0.97))
                fig.savefig(os.path.join(epoch_dir, f"sample_{sample_idx}_encoder.png"), dpi=120)
                plt.close(fig)
                if sample_idx == 0:
                    history_row = {"epoch": epoch, "val_total": val_metrics.get("total"),
                                   "val_ema": ema_value, "overlap_frozen": mean_frozen,
                                   "overlap_student": mean_student,
                                   "erank_frozen": erank_frozen, "erank_student": erank_student,
                                   "drift": drift_mean, "pca3_ratio": top3_ratio}

        for module, was_training in zip(modules, states):
            module.train(was_training)
        if history_row is None:
            return
        viz_history.append(history_row)
        viz_root = VIZ_ROOT
        with open(os.path.join(viz_root, "history.json"), "w") as fh:
            json.dump(viz_history, fh, indent=2)
        epochs = [row["epoch"] for row in viz_history]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        axes[0, 0].plot(epochs, [row["val_total"] for row in viz_history], "o-", label="val")
        axes[0, 0].plot(epochs, [row["val_ema"] for row in viz_history], "o-", label="EMA")
        axes[0, 0].set_title("validation objective"); axes[0, 0].legend()
        axes[0, 1].plot(epochs, [row["overlap_frozen"] for row in viz_history], "--", label="frozen")
        axes[0, 1].plot(epochs, [row["overlap_student"] for row in viz_history], "o-", label="student")
        axes[0, 1].set_title("overlap cosine"); axes[0, 1].legend()
        axes[1, 0].plot(epochs, [row["erank_frozen"] for row in viz_history], "--", label="frozen")
        axes[1, 0].plot(epochs, [row["erank_student"] for row in viz_history], "o-", label="student")
        axes[1, 0].set_title("effective rank"); axes[1, 0].legend()
        axes[1, 1].plot(epochs, [row["drift"] for row in viz_history], "o-", label="drift")
        axes[1, 1].plot(epochs, [row["pca3_ratio"] for row in viz_history], "o-", label="PCA3 ratio")
        axes[1, 1].set_title("semantic drift / concentration"); axes[1, 1].legend()
        for ax in axes.flat:
            ax.grid(alpha=0.25); ax.set_xlabel("epoch")
        fig.tight_layout()
        fig.savefig(os.path.join(viz_root, "training_diagnostics.png"), dpi=130)
        plt.close(fig)
        print(f"validation viz -> {epoch_dir}", flush=True)

    step, t0, agg, done, last = resume_step, time.time(), {}, False, None
    val_ema, best_ema, plateau_count = resume_best_ema, resume_best_ema, resume_plateau
    pbar = tqdm(total=total_steps, initial=resume_step, desc="vicreg",
                mininterval=10, file=sys.stdout, dynamic_ncols=True)
    for ep in range(start_epoch, EPOCHS):
        if done:
            break
        order = torch.randperm(len(pool), generator=g0).tolist()
        epoch_sum, epoch_count = {}, 0
        for bs in range(0, len(order), BATCH):
            opt.zero_grad()
            warm = min(1.0, step / warmup)
            batch_ids = order[bs:bs + BATCH]
            jobs = [(pool[i], i, ep) for i in batch_ids]
            outputs = []
            for item in ex.map(prep, jobs):
                if item is None:
                    continue
                tiles, g, seed = item
                outputs.append(pano_forward(
                    enc, expander, global_expander, sub_expander, geo_head,
                    tiles, g, seed))
                last = outputs[-1]["feat"]
            if not outputs:
                continue
            total, metrics = batch_objective(outputs, warm)
            total.backward()
            opt.step()
            step += 1
            batch_weight = len(outputs)
            for key, value in metrics.items():
                scalar = float(value.detach())
                agg[key] = agg.get(key, 0.0) + scalar
                epoch_sum[key] = epoch_sum.get(key, 0.0) + scalar * batch_weight
            epoch_count += batch_weight
            pbar.update(1)
            pbar.set_postfix(loss=f"{float(total.detach()):.3f}",
                             inv=f"{float(metrics['inv'].detach()):.3f}",
                             var=f"{float(metrics['var'].detach()):.3f}",
                             lr=f"{opt.param_groups[0]['lr']:.1e}", refresh=False)
            if step % LOG_EVERY == 0:
                er = T.erank(last)                          # backbone erank (erosion monitor)
                msg = " ".join(f"{key}={value/LOG_EVERY:.3f}" for key, value in agg.items())
                print(f"ep{ep} step{step}/{total_steps} warm={warm:.2f} sem={SEM} erank={er:.1f} "
                      f"{msg} ({(time.time()-t0)/max(1, step-resume_step):.2f}s/it)", flush=True)
                agg = {}
            if step >= total_steps:
                done = True
                break

        if epoch_count == 0:
            continue
        train_epoch = {key: value / epoch_count for key, value in epoch_sum.items()}
        print(f"[epoch {ep + 1}/{EPOCHS}] train_total={train_epoch['total']:.4f} "
              f"local={train_epoch['local']:.4f} sub={train_epoch['sub']:.4f} "
              f"global={train_epoch['global']:.4f}", flush=True)

        val_metrics = {}
        if val_pool and (ep + 1) % max(1, VAL_EVERY) == 0:
            val_metrics = validate()
            if val_metrics:
                value = val_metrics["total"]
                val_ema = value if val_ema is None else (
                    EARLY_EMA_ALPHA * val_ema + (1.0 - EARLY_EMA_ALPHA) * value)
                eligible = step >= warmup
                improved = eligible and (
                    best_ema is None or val_ema < best_ema * (1.0 - EARLY_MIN_DELTA))
                if improved:
                    best_ema = val_ema
                    plateau_count = 0
                    save_snapshot("best", ep + 1, best_ema, plateau_count)
                elif eligible:
                    plateau_count += 1
                phase = "monitor" if eligible else "warmup"
                print(f"[validation] total={value:.4f} ema={val_ema:.4f} "
                      f"best={best_ema if best_ema is not None else float('nan'):.4f} "
                      f"plateau={plateau_count}/{EARLY_PATIENCE} phase={phase}", flush=True)
                pbar.set_postfix(loss=f"{train_epoch['total']:.3f}",
                                 val=f"{value:.3f}", ema=f"{val_ema:.3f}",
                                 lr=f"{opt.param_groups[0]['lr']:.1e}", refresh=True)
                if (ep + 1) % max(1, VIZ_EVERY) == 0:
                    emit_validation_viz(ep + 1, val_metrics, val_ema)

        if SAVE_EVERY > 0 and (ep + 1) % SAVE_EVERY == 0:
            save_snapshot("last", ep + 1, best_ema, plateau_count)
        if (EARLY_STOP and step >= warmup and best_ema is not None
                and plateau_count >= EARLY_PATIENCE):
            print(f"early stop: validation EMA did not improve by {EARLY_MIN_DELTA:.2%} "
                  f"for {EARLY_PATIENCE} validation epochs", flush=True)
            done = True
    pbar.close()

    if SKIP_SAVE:
        print("smoke validation complete (SKIP_SAVE=1)", flush=True)
        return

    os.makedirs(CKPT, exist_ok=True)
    enc.backbone.save_pretrained(CKPT)
    auxiliary = auxiliary_state()
    torch.save(auxiliary, os.path.join(CKPT, "vicreg_auxiliary.pt"))
    run = runlog.create_run(f"vicreg_3role_{SEM}", {
        "roles": f"local+global+sub VICReg + geometry + semantic({SEM})", "lora_M": n_lora / 1e6,
        "proj_dim": PROJ_DIM, "global_dim": GLOBAL_DIM, "sub_dim": SUB_DIM,
        "weights": {"inv": L_INV, "var": L_VAR, "cov": L_COV, "sem": L_SEM,
                    "global": L_GLOBAL, "sub": L_SUB, "geometry": L_GEO},
        "sem": SEM, "gamma": GAMMA, "epochs": EPOCHS, "completed_epochs": ep + 1,
        "batch": BATCH, "lr": LR, "pool": len(pool), "val_panos": len(val_pool),
        "validation": {"every": VAL_EVERY, "ema_alpha": EARLY_EMA_ALPHA,
                       "patience": EARLY_PATIENCE, "min_delta": EARLY_MIN_DELTA,
                       "best_ema": best_ema, "plateau_count": plateau_count},
        "visualization": {"enabled": VAL_VIZ, "every": VIZ_EVERY,
                          "panos": VIZ_PANOS, "root": VIZ_ROOT},
        "pool_pin": os.environ.get("POOL_PIN"), "amp_bf16": AMP, "steps": total_steps,
        "footprint_safe": True, "yaw_roll": YAW_ROLL, "fov_jitter": FOV_JITTER,
        "photo_sigma": PHOTO_SIGMA})
    enc.backbone.save_pretrained(os.path.join(run, "weights", "adapter"))
    torch.save(auxiliary, os.path.join(run, "weights", "vicreg_auxiliary.pt"))
    print(f"saved -> {CKPT} and {run}", flush=True)
    try:                                           # default train-time viz (opt out: TRAIN_VIZ=0)
        import train_viz
        train_viz.emit_train_viz(run, CKPT)
    except Exception as e:
        print(f"train-viz skipped: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
