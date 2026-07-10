"""SSL losses for the E2P-overlap panorama encoder.

Design (verified by adversarial review — see memory/ssl-loss-recommendation.md):
  L = lambda_distill * (token + relational distillation to the frozen teacher)   # preserve planar semantics+relations
    + lambda_warp    * warp-equivariance on the ERODED, obliquity-weighted overlap  # E2P geometric consistency
    + lambda_reg     * VICReg (variance + COVARIANCE over FULL maps)               # anti-collapse (covariance is load-bearing)

Notes baked in from review:
  * warp loss compares WARPED coordinates F_A(p) vs F_B(Hp) (equivariance), not same-coords.
  * VICReg is computed over the FULL feature map, never overlap-only (overlap pixels are
    spatially correlated and would game the variance hinge).
  * cosine + obliquity weighting so the irreducible edge-stretch residual is not over-penalized.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def warp_equivariance_loss(feat_a: torch.Tensor, feat_b: torch.Tensor,
                           grid: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor,
                           stop_grad_target: bool = True) -> torch.Tensor:
    """Cosine equivariance loss: F_A(p) ~ F_B(H p) on the valid overlap, obliquity-weighted.

    feat_a, feat_b : (B, D, Gh, Gw) dense feature grids.
    grid           : (N, 2) normalized [-1,1] sample locations into B (row-major over A cells).
    valid, weight  : (N,) bool and (N,) float obliquity weights.
    """
    b, d, gh, gw = feat_a.shape
    n = gh * gw
    g = grid.view(1, 1, n, 2).expand(b, 1, n, 2)
    fb = F.grid_sample(feat_b, g, mode="bilinear", align_corners=False)[:, :, 0, :]  # (B, D, N)
    if stop_grad_target:
        fb = fb.detach()
    fa = feat_a.reshape(b, d, n)
    fa = F.normalize(fa, dim=1)
    fb = F.normalize(fb, dim=1)
    cos = (fa * fb).sum(dim=1)                       # (B, N)
    w = (weight * valid.float()).view(1, n)
    denom = w.sum().clamp_min(1.0)
    return ((1.0 - cos) * w).sum() / (denom * b)


def vicreg_var_cov(feat: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """VICReg variance + covariance over the FULL feature map (anti-collapse).

    feat: (B, D, Gh, Gw). Returns (variance_loss, covariance_loss); covariance is the
    load-bearing guard on high-D DINOv2 features.
    """
    b, d, gh, gw = feat.shape
    x = feat.permute(0, 2, 3, 1).reshape(-1, d)      # (B*N, D)
    x = x - x.mean(dim=0, keepdim=True)
    std = torch.sqrt(x.var(dim=0) + eps)
    var_loss = F.relu(gamma - std).mean()
    cov = (x.T @ x) / (x.shape[0] - 1)               # (D, D)
    off = cov - torch.diag(torch.diagonal(cov))
    # MEAN squared off-diagonal covariance (scale-stable on raw high-D backbone features;
    # canonical VICReg uses sum/D, which on raw DINOv2 D=768 dwarfs every other term).
    # Production: apply this on a projector/expander head, not the backbone directly.
    cov_loss = off.pow(2).sum() / (d * (d - 1))
    return var_loss, cov_loss


def vicreg_vc(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """VICReg variance + covariance (anti-collapse) on an expander map z (B,P,Gh,Gw).

    var = hinge to gamma (a REAL target on the expander, not a floor); cov = canonical
    sum-of-squared-off-diagonal / P. MUST be called over a DECORRELATED batch — a pano's
    FULL tile stack, never a single tile: single-tile patches are spatially correlated and
    game the variance hinge (this module's standing warning; the reason F-3/single-tile
    VICReg eroded despite being 'active')."""
    x = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])    # (B*N, P)
    x = x - x.mean(dim=0, keepdim=True)
    std = torch.sqrt(x.var(dim=0) + eps)
    var = F.relu(gamma - std).mean()
    cov = (x.T @ x) / (x.shape[0] - 1)
    off = cov - torch.diag(torch.diagonal(cov))
    return var, off.pow(2).sum() / z.shape[1]            # canonical sum/P (rank guard)


def vicreg_vc_vectors(z: torch.Tensor, gamma: float = 1.0,
                      eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Canonical VICReg variance/covariance for a batch of vectors ``(N, D)``."""
    if z.ndim != 2 or z.shape[0] < 2:
        raise ValueError("vicreg_vc_vectors expects (N,D) with N >= 2")
    x = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(x.var(dim=0) + eps)
    var = F.relu(gamma - std).mean()
    cov = (x.T @ x) / (x.shape[0] - 1)
    off = cov - torch.diag(torch.diagonal(cov))
    return var, off.pow(2).sum() / z.shape[1]


def vicreg_pair_invariance(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """VICReg invariance for aligned vector batches, with canonical element-wise mean."""
    if z_a.shape != z_b.shape:
        raise ValueError(f"paired VICReg shapes differ: {z_a.shape} vs {z_b.shape}")
    return F.mse_loss(z_a, z_b)


def overlap_invariance(z_a: torch.Tensor, z_b: torch.Tensor, grid: torch.Tensor,
                       valid: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Geometric-robustness INVARIANCE: matched-patch MSE over the E2P overlap (the standard
    augmentation-SSL consistency term; E2P transform = the augmentation, warp = exact match).

    MEAN over the P expander channels (canonical VICReg F.mse_loss scale) so it does NOT
    outweigh var/cov by a factor of P. Obliquity-weighted, no stop-grad (VICReg is symmetric).
    z_a, z_b: (B,P,Gh,Gw) expander outputs. Pair with vicreg_vc over the full pano.
    """
    b, p, gh, gw = z_a.shape
    n = gh * gw
    g = grid.view(1, 1, n, 2).expand(b, 1, n, 2)
    zb = F.grid_sample(z_b, g, mode="bilinear", align_corners=False)[:, :, 0, :]   # (B,P,N)
    za = z_a.reshape(b, p, n)
    w = (weight * valid.float()).view(1, n)
    return ((((za - zb) ** 2).mean(dim=1)) * w).sum() / (w.sum().clamp_min(1.0) * b)


@torch.no_grad()
def sinkhorn(scores: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
    """SwAV Sinkhorn-Knopp: balanced soft assignment over a token batch.

    scores: (N, K) prototype scores (cosine in [-1,1]). Returns Q (N, K) with rows
    summing to 1 and prototype usage balanced to ~N/K (anti-collapse target).
    A global max-shift keeps exp() safe for arbitrary score scales.
    """
    q = torch.exp((scores - scores.max()) / eps).t()      # (K, N)
    q = q / q.sum()
    k, n = q.shape
    for _ in range(iters):
        q = q / q.sum(dim=1, keepdim=True) / k            # prototype marginal -> 1/K
        q = q / q.sum(dim=0, keepdim=True) / n            # token marginal -> 1/N
    return (q * n).t()                                    # rows sum to 1


def code_swap_loss(logits_a: torch.Tensor, q_b: torch.Tensor,
                   grid: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor,
                   tau_s: float = 0.1) -> torch.Tensor:
    """M1 swapped code prediction on the overlap: A's student assignment predicts B's
    balanced (sinkhorn) code at the warped location — semantic identity across views.

    logits_a : (B, K, Gh, Gw) student prototype scores at A cells.
    q_b      : (B, K, Gh, Gw) sinkhorn target codes of B (detached here regardless).
    grid, valid, weight: WarpField tensors, same convention as warp_equivariance_loss.
    """
    b, k, gh, gw = logits_a.shape
    n = gh * gw
    g = grid.view(1, 1, n, 2).expand(b, 1, n, 2)
    qb = F.grid_sample(q_b.detach(), g, mode="bilinear", align_corners=False)[:, :, 0, :]
    qb = qb.clamp_min(0)
    qb = qb / qb.sum(dim=1, keepdim=True).clamp_min(1e-8)  # re-normalize after bilinear mix
    logp = F.log_softmax(logits_a.reshape(b, k, n) / tau_s, dim=1)
    ce = -(qb * logp).sum(dim=1)                           # (B, N)
    w = (weight * valid.float()).view(1, n)
    return (ce * w).sum() / (w.sum().clamp_min(1.0) * b)


def distill_loss(student: torch.Tensor, teacher: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preserve the frozen teacher: per-token cosine + relational (Gram self-similarity).

    student, teacher: (B, D, Gh, Gw). teacher must be detached by the caller.
    Returns (token_loss, relational_loss). The relational term preserves inter-region
    structure ('상호관계') even if absolute features drift.
    """
    b, d, gh, gw = student.shape
    s = F.normalize(student, dim=1)
    t = F.normalize(teacher, dim=1)
    token = (1.0 - (s * t).sum(dim=1)).mean()

    sf = s.reshape(b, d, -1).transpose(1, 2)         # (B, N, D)
    tf = t.reshape(b, d, -1).transpose(1, 2)
    gs = torch.bmm(sf, sf.transpose(1, 2))           # (B, N, N) patch self-similarity
    gt = torch.bmm(tf, tf.transpose(1, 2))
    relational = F.mse_loss(gs, gt)
    return token, relational


def combined_loss(student_a: torch.Tensor, student_b: torch.Tensor,
                  teacher_a: torch.Tensor, teacher_b: torch.Tensor,
                  warp: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                  w_warp: float, w_distill: float = 1.0, w_reg: float = 1.0,
                  gamma: float = 1.0) -> Tuple[torch.Tensor, dict]:
    """Full objective. `warp` is (grid, valid, weight). `w_warp` is the warm-up-ramped weight.

    Teacher tensors must already be detached. Returns (total, components dict).
    """
    grid, valid, weight = warp
    l_warp = warp_equivariance_loss(student_a, student_b, grid, valid, weight)
    var_a, cov_a = vicreg_var_cov(student_a, gamma=gamma)
    var_b, cov_b = vicreg_var_cov(student_b, gamma=gamma)
    var, cov = 0.5 * (var_a + var_b), 0.5 * (cov_a + cov_b)
    tok_a, rel_a = distill_loss(student_a, teacher_a)
    tok_b, rel_b = distill_loss(student_b, teacher_b)
    token, relational = 0.5 * (tok_a + tok_b), 0.5 * (rel_a + rel_b)

    total = (w_warp * l_warp
             + w_distill * (token + relational)
             + w_reg * (25.0 * var + cov))           # VICReg-style weighting: var heavy, cov sum
    comps = {
        "warp": l_warp.detach(), "distill_token": token.detach(), "distill_rel": relational.detach(),
        "vic_var": var.detach(), "vic_cov": cov.detach(), "total": total.detach(),
    }
    return total, comps


# ---- PANO-iBOT (docs/PANO_MIM_DESIGN.md) — verified numerically in scratchpad/verify_loss.py ----

def koleo(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KoLeo: uniform-hypersphere spread on L2-normed embeddings (dimensional-collapse guard on
    CLS). z: (n, D). NOTE (verified): guards CLS/dimensional collapse, NOT patch representational
    collapse — the dense guard is gram_anchor. No in-place ops (autograd-safe)."""
    z = F.normalize(z, dim=-1)
    n = z.shape[0]
    d = torch.cdist(z, z) + torch.eye(n, device=z.device, dtype=z.dtype) * 1e6   # exclude self
    return -torch.log(d.min(dim=1).values + eps).mean()


def cross_view_completion_loss(pred: torch.Tensor, target: torch.Tensor,
                               weight: torch.Tensor) -> torch.Tensor:
    """Term B (docs/PANO_WHEREWHAT_SPEC.md §3): obliquity-weighted 1-cosine between the predicted
    and the FROZEN-teacher A feature over masked-and-visible cells.

    pred, target: (n, D) — target must be detached (frozen A feature) by the caller.
    weight: (n,) obliquity weight from the WarpField. Returns a scalar in [0, 2].
    """
    p = F.normalize(pred, dim=-1)
    t = F.normalize(target, dim=-1)
    cos = (p * t).sum(dim=-1)                         # (n,)
    return ((1.0 - cos) * weight).sum() / weight.sum().clamp_min(1.0)


def gram_anchor(student: torch.Tensor, frozen: torch.Tensor) -> torch.Tensor:
    """DINOv3 Gram anchoring: pin the student's per-image patch-similarity (Gram) matrix to the
    FROZEN prior's. Verified to BIND on both representational (0->0.31) and dimensional (0->0.08)
    collapse -> the dense-feature collapse + semantic-erosion guard. student, frozen: (B,D,Gh,Gw)."""
    b, d, gh, gw = student.shape
    xs = F.normalize(student.permute(0, 2, 3, 1).reshape(b, gh * gw, d), dim=-1)
    xf = F.normalize(frozen.permute(0, 2, 3, 1).reshape(b, gh * gw, d), dim=-1)
    return ((torch.bmm(xs, xs.transpose(1, 2)) - torch.bmm(xf, xf.transpose(1, 2))) ** 2).mean()


def ibot_loss(student_scores: torch.Tensor, teacher_scores: torch.Tensor,
              tau_s: float = 0.1, tau_t: float = 0.05, iters: int = 3) -> torch.Tensor:
    """iBOT masked-patch self-distillation on prototype COSINE scores (must be bounded [-1,1];
    unbounded scores or DOUBLE temperature NaN the sinkhorn — verified). Teacher target =
    Sinkhorn-balanced (anti cluster-collapse) at temperature tau_t applied ONCE via eps; student =
    softmax/tau_s. student_scores, teacher_scores: (M, K) over the M masked patches."""
    q = sinkhorn(teacher_scores.detach(), eps=tau_t, iters=iters)      # balanced, stop-grad target
    logp = F.log_softmax(student_scores / tau_s, dim=-1)
    return -(q * logp).sum(dim=1).mean()
