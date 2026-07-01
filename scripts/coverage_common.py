"""Shared library for the E2P multi-pinhole coverage experiments.

Tiling (full-sphere + equator-centered band), coverage/distortion measures, and the
shared plotting layout. Entry scripts (visualize_coverage / band_coverage /
compare_coverage / pinhole_distortion) import from here.

ERP defaults to DEFAULT_ERP; override on any entry script with `--erp PATH`.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import py360convert
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geometry as G  # validated py360 rotation convention

DEFAULT_ERP = "/data/1_personal/4_SWWOO/refer360/data/quic360_format/images/10185122034_f463786774_f.jpg"
CW, CH = 720, 360


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args(argv):
    """Positional args are FOV params; ERP defaults to DEFAULT_ERP (override: --erp PATH)."""
    erp, pos, i = DEFAULT_ERP, [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--erp":
            erp = argv[i + 1]; i += 2
        elif a.startswith("--erp="):
            erp = a.split("=", 1)[1]; i += 1
        else:
            pos.append(a); i += 1
    return erp, pos


# --------------------------------------------------------------------------- #
# Tiling                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tile:
    yaw: float
    pitch: float
    hfov: float
    vfov: float
    is_cap: bool = False


def _ring_yaws(N):
    """Closed-loop yaw centers: uniform 360/N starting at -180 (a tile centered on the
    +/-180 seam), so the last tile overlaps the first across the seam exactly like any
    interior pair -> the ring wraps seamlessly."""
    return [-180 + j * 360 / N for j in range(N)]


def build_tiles(hfov, vfov, r_h, r_v):
    """Full sphere: pitch interval [-90,90] with +/-90 caps; yaw circle per ring (cos phi)."""
    dphi = vfov * (1 - r_v)
    n = max(1, math.ceil(180 / dphi)); dphip = 180 / n
    tiles = []
    for phi in (-90 + i * dphip for i in range(n + 1)):
        if abs(abs(phi) - 90) < 1e-6:
            tiles.append(Tile(0.0, phi, hfov, vfov, True)); continue
        phw = max(0.0, abs(phi) - dphip / 2)
        N = max(1, math.ceil(360 * math.cos(math.radians(phw)) / (hfov * (1 - r_h))))
        tiles += [Tile(y, phi, hfov, vfov) for y in _ring_yaws(N)]
    return tiles


def build_band_tiles(hfov, vfov, r_h, r_v, pmax):
    """Equator-centered symmetric rings within [-pmax, pmax]; no caps."""
    dphi = vfov * (1 - r_v)
    m = max(1, math.ceil(pmax / dphi)); dphip = pmax / m
    tiles = []
    for phi in (i * dphip for i in range(-m, m + 1)):
        phw = max(0.0, abs(phi) - dphip / 2)
        N = max(1, math.ceil(360 * math.cos(math.radians(phw)) / (hfov * (1 - r_h))))
        tiles += [Tile(y, phi, hfov, vfov) for y in _ring_yaws(N)]
    return tiles


# --------------------------------------------------------------------------- #
# Coverage / distortion measures (exact, convention-free via geometry._tile_R) #
# --------------------------------------------------------------------------- #
def world_dirs():
    """Unit vector per ERP cell (py360 convention, verified vs geometry._tile_R)."""
    lon = ((np.arange(CW) + 0.5) / CW * 2 - 1) * np.pi
    lat = (0.5 - (np.arange(CH) + 0.5) / CH) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    cl = np.cos(lat)
    return np.stack([cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)], -1)


def _tile_coords(d, t):
    """Project world dirs into tile frame; return (q, inside-mask)."""
    q = d @ G._tile_R(t.yaw, t.pitch).T
    inside = ((q[..., 2] > 0)
              & (np.abs(np.arctan2(q[..., 0], q[..., 2])) <= math.radians(t.hfov / 2))
              & (np.abs(np.arctan2(q[..., 1], q[..., 2])) <= math.radians(t.vfov / 2)))
    return q, inside


def hitmap(tiles, d=None):
    """# tiles covering each ERP cell, via exact forward projection."""
    if d is None:
        d = world_dirs()
    hit = np.zeros(d.shape[:2], np.int32)
    for t in tiles:
        _, inside = _tile_coords(d, t)
        hit += inside
    return hit


def distortion_map(tiles, d=None):
    """Per ERP cell: min gnomonic area-magnification (1/cos^3 theta) over covering tiles.

    theta = angle from a tile's optical axis (cos theta = forward component). inf where
    uncovered. Lower = a less-distorted view of that region is available.
    """
    if d is None:
        d = world_dirs()
    best = np.full(d.shape[:2], np.inf)
    for t in tiles:
        q, inside = _tile_coords(d, t)
        dist = 1.0 / np.clip(q[..., 2], 1e-6, 1.0) ** 3
        sel = inside & (dist < best)
        best[sel] = dist[sel]
    return best


def tile_distortion(S, hfov, vfov):
    """Per-pixel gnomonic area-magnification 1/cos^3(theta) for an SxS pinhole tile."""
    fx = (S - 1) / (2 * math.tan(math.radians(hfov / 2)))
    fy = (S - 1) / (2 * math.tan(math.radians(vfov / 2)))
    c = (S - 1) / 2
    yy, xx = np.mgrid[0:S, 0:S]
    x, y = (xx - c) / fx, (yy - c) / fy
    cos = 1.0 / np.sqrt(x * x + y * y + 1.0)
    return cos ** -3


def load_erp(path, size=(CW, CH)):
    img = Image.open(path).convert("RGB") if path and os.path.exists(path) else Image.new("RGB", size)
    return np.asarray(img.resize(size, Image.BILINEAR))


# --------------------------------------------------------------------------- #
# Shared figure                                                               #
# --------------------------------------------------------------------------- #
def plot_coverage(erp_path, tiles, hit, suptitle, savepath):
    """4-section layout: ERP+centers / hit-count / best-view distortion / real crops.

    Ring pitches are cyan lines, equator (0) bold yellow (shows equator-centering).
    Bottom block = real pinhole crops as a cube-net (rows = ring pitches, north on top).
    """
    ext = [-180, 180, -90, 90]
    erp = load_erp(erp_path)
    erp_hi = load_erp(erp_path, (2048, 1024))
    pitches = sorted({round(t.pitch, 3) for t in tiles})
    has_eq = any(abs(p) < 1e-6 for p in pitches)
    ring_p = sorted(pitches, reverse=True)
    rows = {p: sorted([t for t in tiles if abs(t.pitch - p) < 1e-6], key=lambda t: t.yaw)
            for p in ring_p}
    ncol = max(len(v) for v in rows.values())

    dist = distortion_map(tiles)
    dm = np.where(np.isinf(dist), np.nan, dist)
    latw = np.cos(np.radians((0.5 - (np.arange(CH) + 0.5) / CH) * 180))[:, None] * np.ones((1, CW))
    cov = np.isfinite(dist)
    awd = float((dist[cov] * latw[cov]).sum() / latw[cov].sum())

    fig = plt.figure(figsize=(max(9, ncol * 1.25), 14 + len(ring_p) * 1.25), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[2.0, 2.0, 2.0, len(ring_p) * 0.9])
    a1, a2, a3 = (fig.add_subplot(gs[i]) for i in range(3))

    a1.imshow(erp, extent=ext, aspect="auto")
    for t in tiles:
        a1.plot(t.yaw, t.pitch, "*" if t.is_cap else "o", ms=14 if t.is_cap else 5,
                color="red" if t.is_cap else "yellow", mec="k", mew=0.6, clip_on=False)
    im2 = a2.imshow(hit, extent=ext, aspect="auto", cmap="viridis", vmin=0)
    cmap = plt.cm.inferno.copy(); cmap.set_bad("lightgray")
    vmax = float(np.nanpercentile(dm, 99)) if np.isfinite(dm).any() else 8.0
    im3 = a3.imshow(dm, extent=ext, aspect="auto", cmap=cmap, vmin=1.0, vmax=vmax)
    for ax in (a1, a2, a3):
        for p in pitches:
            ax.axhline(p, color="cyan", lw=0.8, alpha=0.6)
        ax.axhline(0, color="yellow", lw=1.6)
        ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="yaw", ylabel="pitch")
    eq = "ring AT equator -> centered" if has_eq else "no ring at equator -> NOT centered"
    a1.set_title(f"Tile centers — {len(tiles)} tiles | rings(cyan) at {pitches} | {eq}", fontsize=9)
    a2.set_title(f"Coverage hit-count — min={hit.min()} mean={hit.mean():.2f} max={hit.max()}")
    a3.set_title(f"Best-view distortion — min 1/cos³θ per cell, area-wt mean={awd:.2f}×  (gray=uncovered)")
    fig.colorbar(im2, ax=a2, location="bottom", shrink=0.6, label="# tiles")
    fig.colorbar(im3, ax=a3, location="bottom", shrink=0.6, label="area magnification ×")

    sub = gs[3].subgridspec(len(ring_p), ncol)
    for r, p in enumerate(ring_p):
        ts = rows[p]; start = (ncol - len(ts)) // 2
        for c in range(ncol):
            ax = fig.add_subplot(sub[r, c]); ax.axis("off")
            j = c - start
            if 0 <= j < len(ts):
                t = ts[j]
                crop = py360convert.e2p(erp_hi, (t.hfov, t.vfov), t.yaw, t.pitch,
                                        out_hw=(160, 160), mode="bilinear").astype(np.uint8)
                ax.imshow(crop)
                ax.set_title(f"y{t.yaw:+.0f} p{t.pitch:+.0f}", fontsize=6)
    fig.suptitle(suptitle)
    fig.savefig(savepath, dpi=120)
    print(f"saved {savepath} | area-wt best-view distortion = {awd:.2f}x")
