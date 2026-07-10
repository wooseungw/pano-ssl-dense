"""Cross-tile correspondence geometry for E2P-overlap SSL.

Two facts this module encodes (verified to machine precision against py360convert 1.0.4):
  * Adjacent AnyRes-E2P tiles share ONE optical center, so the overlap map is an
    exact rotation homography H = K_B R_ab K_A^-1 (no parallax, depth-independent).
  * A wrong-but-plausible analytic H is *catastrophic yet passes horizon smoke tests*.

Therefore TRAINING correspondence is built convention-free from render-time COORDINATE
MAPS (`warp_field_from_coordmaps`); the analytic homography (`tile_homography`) is kept
only for the metadata-only DEPLOYMENT path and is gated by tests/test_geometry.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import py360convert


# --------------------------------------------------------------------------- #
# Coordinate maps (convention-free ground-truth correspondence)               #
# --------------------------------------------------------------------------- #
def render_coordmap(erp_h: int, erp_w: int, yaw_deg: float, pitch_deg: float,
                    hfov_deg: float, out_size: int) -> np.ndarray:
    """Return (out_size, out_size, 2) = the ERP (x, y) each tile pixel sampled.

    Built by e2p-sampling an ERP whose pixels encode their own (x, y) index with
    nearest interpolation, so it is exact and independent of py360's axis convention.
    """
    yy, xx = np.mgrid[0:erp_h, 0:erp_w]
    coord = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).astype(np.float32)
    cmap = py360convert.e2p(coord, hfov_deg, yaw_deg, pitch_deg,
                            out_hw=(out_size, out_size), mode="nearest")
    return cmap[..., :2]


def tile_position_labels(erp_h: int, erp_w: int, yaw_deg: float, pitch_deg: float,
                         hfov_deg: float, out_size: int, patch: int
                         ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Per-patch spherical POSITION labels for the PWW pretext (docs/PANO_WHEREWHAT_SPEC.md).

    Derived (free, exact) from the render-time coordinate map: each patch-centre's ERP (x, y)
    -> gravity-referenced latitude and tile-relative longitude.

    Well-posedness: latitude is ABSOLUTE (ERP row is gravity-locked); longitude is a free gauge
    (roll), so we return longitude RELATIVE to the tile centre only — never absolute longitude.

    Returns (patch_lat (Gh,Gw) deg, patch_dlon (Gh,Gw) deg relative-to-centre,
             centre_lat float deg, hfov float deg). Gh=Gw=out_size//patch.
    """
    cmap = render_coordmap(erp_h, erp_w, yaw_deg, pitch_deg, hfov_deg, out_size)   # (out,out,2)=(x,y)
    grid_n = out_size // patch
    half = patch // 2
    gi, gj = np.mgrid[0:grid_n, 0:grid_n]
    xy = cmap[gi * patch + half, gj * patch + half]                                # (Gh,Gw,2)
    lat = 90.0 - (xy[..., 1] + 0.5) / erp_h * 180.0                                # absolute latitude
    lon = (xy[..., 0] + 0.5) / erp_w * 360.0 - 180.0
    centre_lon = lon[grid_n // 2, grid_n // 2]
    dlon = (lon - centre_lon + 180.0) % 360.0 - 180.0                              # wrap-safe, seam-safe
    return (lat.astype(np.float32), dlon.astype(np.float32),
            float(lat[grid_n // 2, grid_n // 2]), float(hfov_deg))


@dataclass(frozen=True)
class WarpField:
    """Geometry-only (image-independent) correspondence from tile A -> tile B features.

    grid   : (Gh*Gw, 2) normalized [-1,1] (x,y) sample locations into B's feature grid
             for F.grid_sample(align_corners=False); ordered row-major over A cells.
    valid  : (Gh*Gw,) bool — A cells that fall inside B's overlap.
    weight : (Gh*Gw,) float in (0,1] — min(cos theta_A, cos theta_B) obliquity weight.
    grid_hw: (Gh, Gw) feature-grid shape.
    """

    grid: np.ndarray
    valid: np.ndarray
    weight: np.ndarray
    grid_hw: Tuple[int, int]


def _offaxis_cos(px_col: np.ndarray, px_row: np.ndarray, out_size: int, hfov_deg: float) -> np.ndarray:
    """cos(theta) of each pixel's gnomonic off-axis angle (1 at center -> small at edge)."""
    c = (out_size - 1) / 2.0
    f = (out_size - 1) / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    r2 = (px_col - c) ** 2 + (px_row - c) ** 2
    return f / np.sqrt(r2 + f * f)


def warp_field_from_coordmaps(cmap_a: np.ndarray, cmap_b: np.ndarray, patch: int,
                              hfov_deg: float, erp_w: int, thresh_px: float = 4.0,
                              dst_stride: int = 2, erode_patches: float = 0.0,
                              mutual_nearest: bool = False,
                              confidence_sigma_px: float = 0.0) -> WarpField:
    """Build a feature-level WarpField (A->B) from two coordinate maps. Image-independent.

    Match each A patch-center to the nearest B pixel in ERP-coordinate space (x wraps with
    modulus erp_w for seam tiles), then express that B pixel as a normalized grid_sample
    location on B's feature grid.
    """
    out_size = cmap_a.shape[0]
    grid_n = out_size // patch
    half = patch // 2

    gi, gj = np.mgrid[0:grid_n, 0:grid_n]
    a_row = (gi * patch + half).ravel()
    a_col = (gj * patch + half).ravel()
    a_erp = cmap_a[a_row, a_col]                                  # (N,2) erp (x,y)

    b_row, b_col = np.mgrid[0:out_size:dst_stride, 0:out_size:dst_stride]
    b_row, b_col = b_row.ravel(), b_col.ravel()
    b_erp = cmap_b[b_row, b_col]                                  # (M,2)

    # nearest B pixel per A center (handles ERP x-wrap at the seam)
    dx = np.abs(a_erp[:, None, 0] - b_erp[None, :, 0])
    dx = np.minimum(dx, erp_w - dx)                              # wrap in x at the seam
    dy = a_erp[:, None, 1] - b_erp[None, :, 1]
    d = np.sqrt(dx * dx + dy * dy)
    nn = d.argmin(1)
    match_error = d[np.arange(len(a_erp)), nn]
    valid = match_error < thresh_px

    mb_row = b_row[nn].astype(np.float32)
    mb_col = b_col[nn].astype(np.float32)
    if erode_patches > 0:
        margin = float(erode_patches * patch)
        valid &= ((mb_col >= margin) & (mb_col <= out_size - 1 - margin)
                  & (mb_row >= margin) & (mb_row <= out_size - 1 - margin))
    if mutual_nearest:
        # A centre must also be the nearest A centre to its selected B sample. This removes
        # many-to-one matches at stretched tile edges without building a second correspondence.
        valid &= d[:, nn].argmin(axis=0) == np.arange(len(a_erp))
    nx = (mb_col + 0.5) / out_size * 2.0 - 1.0
    ny = (mb_row + 0.5) / out_size * 2.0 - 1.0
    grid = np.stack([nx, ny], axis=-1).astype(np.float32)

    w_a = _offaxis_cos(a_col.astype(np.float32), a_row.astype(np.float32), out_size, hfov_deg)
    w_b = _offaxis_cos(mb_col, mb_row, out_size, hfov_deg)
    weight = np.minimum(w_a, w_b).astype(np.float32)
    if confidence_sigma_px > 0:
        confidence = np.exp(-0.5 * (match_error / confidence_sigma_px) ** 2)
        weight *= confidence.astype(np.float32)

    return WarpField(grid=grid, valid=valid, weight=weight, grid_hw=(grid_n, grid_n))


def warp_field_from_homography(hfov_deg: float, out_size: int,
                               yaw_a: float, pitch_a: float,
                               yaw_b: float, pitch_b: float,
                               patch: int, footprint: bool = True,
                               erode_px: float = 1.0,
                               cycle_sigma_px: float = 1.0) -> WarpField:
    """Build an exact A->B token warp and validate the complete patch footprint.

    Unlike :func:`warp_field_from_coordmaps`, this path is O(number-of-tokens), so it is
    practical for the 2x sub-token grid. ``footprint=True`` accepts a positive only when
    all four corners of A's patch project inside B after erosion. The analytic homography
    is renderer-gated by ``tests/test_geometry.py``.
    """
    if patch <= 0 or out_size % patch:
        raise ValueError("patch must be positive and divide out_size")
    grid_n = out_size // patch
    half = patch / 2.0
    gi, gj = np.mgrid[0:grid_n, 0:grid_n]
    a_row = ((gi + 0.5) * patch - 0.5).ravel().astype(np.float64)
    a_col = ((gj + 0.5) * patch - 0.5).ravel().astype(np.float64)

    h_ab = tile_homography(hfov_deg, out_size, yaw_a, pitch_a, yaw_b, pitch_b)
    b_col, b_row = apply_homography(h_ab, a_col, a_row)
    valid = (np.isfinite(b_col) & np.isfinite(b_row)
             & (b_col >= erode_px) & (b_col <= out_size - 1 - erode_px)
             & (b_row >= erode_px) & (b_row <= out_size - 1 - erode_px))

    if footprint:
        # Patch edges, not just its centre, must be visible in B. This is the token-level
        # label contract missing from centre-only correspondence.
        for dc, dr in ((-half, -half), (half, -half), (-half, half), (half, half)):
            cc, rr = apply_homography(h_ab, a_col + dc, a_row + dr)
            valid &= (np.isfinite(cc) & np.isfinite(rr)
                      & (cc >= erode_px) & (cc <= out_size - 1 - erode_px)
                      & (rr >= erode_px) & (rr <= out_size - 1 - erode_px))

    nx = (b_col + 0.5) / out_size * 2.0 - 1.0
    ny = (b_row + 0.5) / out_size * 2.0 - 1.0
    grid = np.stack([nx, ny], axis=-1).astype(np.float32)

    w_a = _offaxis_cos(a_col.astype(np.float32), a_row.astype(np.float32), out_size, hfov_deg)
    w_b = _offaxis_cos(b_col.astype(np.float32), b_row.astype(np.float32), out_size, hfov_deg)
    weight = np.minimum(w_a, w_b).astype(np.float32)
    if cycle_sigma_px > 0:
        h_ba = tile_homography(hfov_deg, out_size, yaw_b, pitch_b, yaw_a, pitch_a)
        ac, ar = apply_homography(h_ba, b_col, b_row)
        cycle_error = np.sqrt((ac - a_col) ** 2 + (ar - a_row) ** 2)
        valid &= np.isfinite(cycle_error)
        weight *= np.exp(-0.5 * (cycle_error / cycle_sigma_px) ** 2).astype(np.float32)

    return WarpField(grid=grid, valid=valid, weight=weight, grid_hw=(grid_n, grid_n))


# --------------------------------------------------------------------------- #
# Analytic homography (DEPLOYMENT path only — gated by tests)                  #
# --------------------------------------------------------------------------- #
def _rodrigues(rad: float, axis: Tuple[float, float, float]) -> np.ndarray:
    """Match py360convert.utils.rotation_matrix exactly (row-vector convention)."""
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / np.sqrt((ax ** 2).sum())
    R = np.diag([math.cos(rad)] * 3).astype(np.float64)
    R = R + np.outer(ax, ax) * (1.0 - math.cos(rad))
    s = ax * math.sin(rad)
    R = R + np.array([[0, -s[2], s[1]], [s[2], 0, -s[0]], [-s[1], s[0], 0]])
    return R


def _tile_R(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """py360 e2p orientation: ray_row = d0_row @ (Rx(pitch) @ Ry(-yaw)), in_rot=0."""
    u = -math.radians(yaw_deg)
    v = math.radians(pitch_deg)
    return _rodrigues(v, (1.0, 0.0, 0.0)) @ _rodrigues(u, (0.0, 1.0, 0.0))


def tile_homography(hfov_deg: float, out_size: int,
                    yaw_a: float, pitch_a: float, yaw_b: float, pitch_b: float) -> np.ndarray:
    """H mapping tile-A pixels (col,row,1) -> tile-B pixels (col,row,1), homogeneous.

    Uses d0 = (col - c, c - row, f); M = R_A R_B^T; recovers B pixel by reprojection.
    Returned as a 3x3 acting on column vectors [col, row, 1]^T.
    """
    c = (out_size - 1) / 2.0
    f = (out_size - 1) / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    M = _tile_R(yaw_a, pitch_a) @ _tile_R(yaw_b, pitch_b).T          # row-vector: d0_B = d0_A @ M
    # d0_A = (col-c, c-row, f) = K_inv @ [col,row,1]; here build explicit 3x3.
    Kinv = np.array([[1.0, 0.0, -c], [0.0, -1.0, c], [0.0, 0.0, f]], dtype=np.float64)
    K = np.array([[f, 0.0, c], [0.0, -f, c], [0.0, 0.0, 1.0]], dtype=np.float64)
    # d0_A column = Kinv @ p ; d0_B column = M^T @ d0_A column ; p_B = K @ d0_B column
    return K @ M.T @ Kinv


def apply_homography(H: np.ndarray, cols: np.ndarray, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map pixel arrays through H. Returns (col_b, row_b)."""
    ones = np.ones_like(cols, dtype=np.float64)
    p = np.stack([cols.astype(np.float64), rows.astype(np.float64), ones], axis=0)  # (3,N)
    q = H @ p
    return q[0] / q[2], q[1] / q[2]
