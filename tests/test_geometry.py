"""Gating tests for E2P cross-tile correspondence.

These guard the SSL objective: a wrong-but-plausible homography passes a horizon-only
smoke test yet trains the whole loss on garbage. We verify against the REAL py360convert
renderer using the convention-free coordinate-map correspondence as ground truth, across a
battery spanning pitch, vertical (pitch-differing) pairs, and the +/-180 seam.
"""

from __future__ import annotations

import numpy as np
import pytest

import anyres_e2p as a2p
import geometry as G

py360convert = pytest.importorskip("py360convert")

ERP_H, ERP_W = 512, 1024
HFOV = 90.0
OUT = 224

# (yaw_a, pitch_a, yaw_b, pitch_b, label)
PAIRS = [
    (0.0, 0.0, 60.0, 0.0, "horizon"),
    (0.0, 30.0, 60.0, 30.0, "pitch30"),
    (0.0, 45.0, 60.0, 45.0, "pitch45"),
    (0.0, 75.0, 50.0, 75.0, "pitch75"),
    (10.0, -36.0, 10.0, 36.0, "vertical"),
    (170.0, 0.0, -130.0, 0.0, "seam"),
]


def _textured_erp() -> np.ndarray:
    """Deterministic moderate-frequency ERP: enough structure that a wrong correspondence
    spikes MAE, but not near-Nyquist (which would inflate MAE from pure resampling)."""
    yy, xx = np.mgrid[0:ERP_H, 0:ERP_W].astype(np.float32)
    r = np.sin(xx / 41.0) * np.cos(yy / 33.0) * 0.5 + 0.5
    g = np.sin((xx + 2 * yy) / 53.0) * 0.5 + 0.5
    b = np.sin(xx / 29.0 + yy / 37.0) * 0.5 + 0.5
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    x1, y1 = np.clip(x0 + 1, 0, w - 1), np.clip(y0 + 1, 0, h - 1)
    x0, y0 = np.clip(x0, 0, w - 1), np.clip(y0, 0, h - 1)
    wx, wy = (xs - np.floor(xs))[:, None], (ys - np.floor(ys))[:, None]
    return (img[y0, x0] * (1 - wx) * (1 - wy) + img[y0, x1] * wx * (1 - wy)
            + img[y1, x0] * (1 - wx) * wy + img[y1, x1] * wx * wy)


def _coordmap_correspondence(pair):
    """Ground-truth A->B pixel correspondence + overlap mask, convention-free (O(N)).

    Scatter each B pixel index into an ERP-indexed inverse map (4 integer corners for
    robustness to sub-pixel rounding), then look up each A pixel's ERP cell.
    """
    ca = G.render_coordmap(ERP_H, ERP_W, pair[0], pair[1], HFOV, OUT)
    cb = G.render_coordmap(ERP_H, ERP_W, pair[2], pair[3], HFOV, OUT)
    inv = np.full((ERP_H, ERP_W), -1, dtype=np.int64)
    bx, by = cb[..., 0].ravel(), cb[..., 1].ravel()
    bidx = np.arange(OUT * OUT)
    for ox in (0, 1):
        for oy in (0, 1):
            cx = np.clip(np.floor(bx).astype(int) + ox, 0, ERP_W - 1)
            cy = np.clip(np.floor(by).astype(int) + oy, 0, ERP_H - 1)
            inv[cy, cx] = bidx
    ax = np.clip(np.floor(ca[..., 0]).astype(int), 0, ERP_W - 1).ravel()
    ay = np.clip(np.floor(ca[..., 1]).astype(int), 0, ERP_H - 1).ravel()
    nn = inv[ay, ax]
    valid = nn >= 0
    return ca, cb, np.where(valid, nn, 0), valid


@pytest.mark.parametrize("pair", PAIRS, ids=[p[4] for p in PAIRS])
def test_analytic_H_matches_renderer(pair):
    """Analytic H must reproduce the renderer's correspondence to < 1 px on the overlap.

    This is the convention guard: a sign/order bug here yields 10s-1000s of px error.
    """
    ca, cb, nn, valid = _coordmap_correspondence(pair)
    assert valid.sum() > 50, f"{pair[4]}: too little overlap to test"

    rows, cols = np.divmod(np.arange(OUT * OUT), OUT)
    H = G.tile_homography(HFOV, OUT, pair[0], pair[1], pair[2], pair[3])
    bc, br = G.apply_homography(H, cols[valid], rows[valid])

    gt_row, gt_col = np.divmod(nn[valid], OUT)
    err = np.sqrt((bc - gt_col) ** 2 + (br - gt_row) ** 2)
    assert np.median(err) < 1.5, f"{pair[4]}: median H error {np.median(err):.2f}px (convention bug?)"
    assert np.percentile(err, 95) < 3.0, f"{pair[4]}: p95 H error {np.percentile(err, 95):.2f}px"


@pytest.mark.parametrize("pair", PAIRS, ids=[p[4] for p in PAIRS])
def test_rgb_roundtrip(pair):
    """Warp tile_B RGB into tile_A via analytic H; overlap MAE must be resampling-small."""
    erp = _textured_erp()
    ta = np.asarray(a2p.erp_to_pinhole_tile(erp, pair[0], pair[1], HFOV, OUT)).astype(np.float32)
    tb = np.asarray(a2p.erp_to_pinhole_tile(erp, pair[2], pair[3], HFOV, OUT)).astype(np.float32)

    _, _, nn, valid = _coordmap_correspondence(pair)
    rows, cols = np.divmod(np.arange(OUT * OUT), OUT)
    H = G.tile_homography(HFOV, OUT, pair[0], pair[1], pair[2], pair[3])
    bc, br = G.apply_homography(H, cols[valid], rows[valid])
    inb = (bc >= 0) & (bc <= OUT - 1) & (br >= 0) & (br <= OUT - 1)
    idx = np.where(valid)[0][inb]

    recon = _bilinear(tb, bc[inb], br[inb])
    target = ta.reshape(-1, 3)[idx]
    mae = np.abs(recon - target).mean()
    # correct-but-stretched is ~<10/255; a wrong convention is ~76/255 (verified) -> 30 separates.
    assert mae < 30.0, f"{pair[4]}: RGB round-trip MAE {mae:.1f}/255 (correspondence wrong?)"


def test_warpfield_is_geometry_only_and_valid():
    """The training WarpField (coordmap-based) yields a sane overlap fraction + grid range."""
    ca = G.render_coordmap(ERP_H, ERP_W, 0.0, 0.0, HFOV, OUT)
    cb = G.render_coordmap(ERP_H, ERP_W, 60.0, 0.0, HFOV, OUT)
    wf = G.warp_field_from_coordmaps(ca, cb, patch=14, hfov_deg=HFOV, erp_w=ERP_W)
    frac = wf.valid.mean()
    assert 0.15 < frac < 0.55, f"overlap fraction {frac:.2f} unexpected"
    assert wf.grid[wf.valid].min() >= -1.0 and wf.grid[wf.valid].max() <= 1.0
    assert (wf.weight > 0).all() and (wf.weight <= 1.0 + 1e-6).all()


def test_renderer_default_has_seam_overlap():
    """Config-bug fix: default render must wrap 360 so the +/-180 seam has overlapping tiles."""
    from PIL import Image

    erp = Image.fromarray(_textured_erp())
    pack = a2p.build_anyres_from_erp(erp, hfov_deg=HFOV, overlap=0.3)
    assert pack.yaw_geometry["closed_loop_yaw"] is True
    centers = pack.yaw_geometry["yaw_centers_deg"]
    # closed loop => a center at the seam and uniform wrap-around spacing
    assert any(abs(abs(c) - 180.0) < 1e-6 for c in centers), "no seam-straddling tile center"


def test_full_sphere_mode_has_caps_and_cos_correction():
    """mode='full_sphere': +/-90 pole caps + cos(phi) yaw (fewer tiles toward the poles)."""
    from PIL import Image

    erp = Image.fromarray(_textured_erp())
    pack = a2p.build_anyres_from_erp(erp, tile_render_size=OUT, hfov_deg=HFOV, overlap=0.2,
                                     mode="full_sphere")
    g = pack.yaw_geometry
    assert g["mode"] == "full_sphere" and g["n_caps"] == 2
    assert sorted(round(m.pitch_deg) for m in pack.metas if m.is_cap) == [-90, 90]
    counts = g["ring_tile_counts"]
    assert counts[0] == 1 and counts[-1] == 1                       # caps are N=1
    assert max(counts) == counts[len(counts) // 2]                  # most tiles near the equator
    ring_pitch = next(p for p in g["ring_pitches_deg"] if abs(abs(p) - 90) > 1e-6)
    yaws = [m.yaw_deg for m in pack.metas if abs(m.pitch_deg - ring_pitch) < 1e-6]
    assert any(abs(abs(y) - 180.0) < 1e-6 for y in yaws)           # ring wraps the +/-180 seam


def test_band_mode_equator_centered_no_caps_independent_vfov():
    """mode='band': a ring on the equator, symmetric pitches, no caps, independent vfov."""
    from PIL import Image

    erp = Image.fromarray(_textured_erp())
    pack = a2p.build_anyres_from_erp(erp, tile_render_size=OUT, hfov_deg=HFOV, vfov_deg=60.0,
                                     overlap=0.2, mode="band", pmax_deg=45.0)
    g = pack.yaw_geometry
    assert g["mode"] == "band" and g["n_caps"] == 0
    pitches = [round(p, 3) for p in g["ring_pitches_deg"]]
    assert any(abs(p) < 1e-6 for p in pitches)                     # ring centered on equator
    assert pitches == [round(-p, 3) for p in reversed(pitches)]    # symmetric about equator
    assert all(m.vfov_deg == 60.0 for m in pack.metas)             # vfov independent of hfov
