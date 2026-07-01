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
