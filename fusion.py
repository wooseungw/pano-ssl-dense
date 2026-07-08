"""F-2 learned set-fusion: per-ERP-cell attention over covering tile contributions.

Design (docs/SEMANTIC_IDENTITY_SSL.md §9.6): the encoder is FROZEN (per-tile features
preserved — user requirement); learning lives only in how contributions are combined.
Residual parameterization: fused = masked_mean + g(set), with g's output layer
ZERO-INITIALIZED — at init the module IS the validated uniform-mean baseline (F-1
winner), so training can only add on top of it, never regress below it at start.

What g can express that the mean cannot: set statistics beyond the first moment
(cross-view variance as uncertainty), channel-selective mixing, content-conditioned
correction of warp-resampling blur.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x (B, K, D), mask (B, K) bool -> (B, D)."""
    m = mask.float().unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


class SetFusion(nn.Module):
    """fused = mean(contributions) + zero-init attention correction."""

    def __init__(self, dim: int = 768, geo_dim: int = 4, d_model: int = 256,
                 n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.inp = nn.Linear(dim + geo_dim, d_model)
        # dropout=0.0: no train-time regularization intended, and eval must be
        # deterministic — default 0.1 would corrupt the paired attn-vs-mean metric
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                           dim_feedforward=2 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Linear(d_model, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feats: torch.Tensor, geo: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """feats (B, K, D), geo (B, K, G), mask (B, K) bool (>=1 True per row) -> (B, D)."""
        base = masked_mean(feats, mask)
        h = self.enc(self.inp(torch.cat([feats, geo], dim=-1)),
                     src_key_padding_mask=~mask)
        return base + self.out(masked_mean(h, mask))


def scatter_mean_field(cids: torch.Tensor, feats: torch.Tensor, ncell: int):
    """Flat contributions -> ((ncell, D) mean field, (ncell,) counts). Differentiable
    through feats — F-3 builds its context field from VISIBLE tiles with this."""
    field = feats.new_zeros(ncell, feats.shape[1])
    field.index_add_(0, cids, feats)
    counts = torch.bincount(cids, minlength=ncell).to(feats.dtype)
    return field / counts.clamp_min(1.0).unsqueeze(1), counts


def pack_sets(cid: torch.Tensor, feats: torch.Tensor, geo: torch.Tensor,
              ncell: int, max_cov: int):
    """Group flat contributions by ERP cell into padded sets.

    cid (M,) long cell ids; feats (M, D); geo (M, G).
    Returns (ncell, max_cov, D), (ncell, max_cov, G), mask (ncell, max_cov).
    Contributions beyond max_cov per cell are dropped IN PLAN ORDER — size max_cov
    above the true coverage maximum (44 for the F-2 band plans) to avoid truncation.
    """
    order = torch.argsort(cid, stable=True)
    cid_s = cid[order]
    counts = torch.bincount(cid_s, minlength=ncell)
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.arange(cid.numel(), device=cid.device) - starts[cid_s]
    keep = slot < max_cov
    out_f = feats.new_zeros(ncell, max_cov, feats.shape[1])
    out_g = geo.new_zeros(ncell, max_cov, geo.shape[1])
    mask = torch.zeros(ncell, max_cov, dtype=torch.bool, device=cid.device)
    out_f[cid_s[keep], slot[keep]] = feats[order][keep]
    out_g[cid_s[keep], slot[keep]] = geo[order][keep]
    mask[cid_s[keep], slot[keep]] = True
    return out_f, out_g, mask
