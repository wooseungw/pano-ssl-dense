"""Visualize the ACTUAL anyres_e2p production output (both modes) — pack.tiles/metas/yaw_geometry.

Needs the torch env (anyres_e2p imports torch). Renders one figure per mode:
  top  = ERP + tile centers (cyan ring pitches, yellow equator, red ★ caps)
  bottom = cube-net of the real rendered tiles from pack.tiles

Run:  <torch-python> scripts/visualize_anyres.py [hfov] [vfov] [overlap] [pmax] [--erp PATH]
"""

import math
import os
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
import anyres_e2p as a2p
import geometry as G  # validated tile<->world rotation
from coverage_common import world_dirs, hitmap, distortion_map, Tile


def tile_footprint_lonlat(yaw, pitch, hfov, vfov, S=64):
    """Tile border -> (lon, lat) degrees, via the verified tile->world rotation."""
    fx = (S - 1) / (2 * math.tan(math.radians(hfov / 2)))
    fy = (S - 1) / (2 * math.tan(math.radians(vfov / 2)))
    c = (S - 1) / 2
    cols, rows = np.arange(S), np.arange(S)
    per = np.concatenate([
        np.stack([cols, np.zeros(S)], 1),                 # top
        np.stack([np.full(S, S - 1), rows], 1),           # right
        np.stack([cols[::-1], np.full(S, S - 1)], 1),     # bottom
        np.stack([np.zeros(S), rows[::-1]], 1),           # left
    ])
    x = (per[:, 0] - c) / fx
    y = (c - per[:, 1]) / fy
    d = np.stack([x, y, np.ones_like(x)], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    dw = d @ G._tile_R(yaw, pitch)                          # tile -> world
    lon = np.degrees(np.arctan2(dw[:, 0], dw[:, 2]))
    lat = np.degrees(np.arcsin(np.clip(dw[:, 1], -1, 1)))
    return lon, lat

DEFAULT_ERP = "/data/1_personal/4_SWWOO/refer360/data/quic360_format/images/10185122034_f463786774_f.jpg"


def parse_args(argv):
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


def viz(erp_img, erp_small, mode, hfov, vfov, ov, pmax, out):
    pack = a2p.build_anyres_from_erp(erp_img, tile_render_size=256, hfov_deg=hfov, vfov_deg=vfov,
                                     overlap=ov, mode=mode, pmax_deg=pmax)
    g, metas, tiles = pack.yaw_geometry, pack.metas, pack.tiles
    ctiles = [Tile(m.yaw_deg, m.pitch_deg, m.hfov_deg, m.vfov_deg, m.is_cap) for m in metas]
    d = world_dirs(); hit = hitmap(ctiles, d); dist = distortion_map(ctiles, d)
    dm = np.where(np.isinf(dist), np.nan, dist)
    pitches = sorted({round(m.pitch_deg, 3) for m in metas}, reverse=True)   # north -> south
    rows = {p: sorted([i for i, m in enumerate(metas) if abs(m.pitch_deg - p) < 1e-6],
                      key=lambda i: metas[i].yaw_deg) for p in pitches}
    ncol = max(len(v) for v in rows.values())
    ext = [-180, 180, -90, 90]

    glob = pack.global_image.permute(1, 2, 0).cpu().numpy()
    fig = plt.figure(figsize=(max(10, ncol * 1.3), 11 + len(pitches) * 1.5), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.9, 1.5, 1.5, len(pitches) * 0.9])
    top = gs[0].subgridspec(1, 2, width_ratios=[1.0, 1.9])

    axg = fig.add_subplot(top[0]); axg.axis("off")
    axg.imshow(np.clip(glob, 0, 1))
    axg.set_title(f"global_image  ({glob.shape[0]}×{glob.shape[1]} letterbox)", fontsize=9)

    a1 = fig.add_subplot(top[1])
    a1.imshow(erp_small, extent=ext, aspect="auto")
    for m in metas:
        a1.plot(m.yaw_deg, m.pitch_deg, "*" if m.is_cap else "o", ms=14 if m.is_cap else 5,
                color="red" if m.is_cap else "yellow", mec="k", mew=0.6, clip_on=False)
    for p in pitches:
        a1.axhline(p, color="cyan", lw=0.8, alpha=0.6)
    a1.axhline(0, color="yellow", lw=1.6)
    a1.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="yaw", ylabel="pitch",
           title=f"mode='{mode}'  {g['n_tiles']} tiles  caps={g['n_caps']}  ring_counts={g['ring_tile_counts']}")

    # overlay each tile's footprint on BOTH the ERP map and the global_image
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(metas))))
    Werp, Herp = erp_img.size
    gsz = glob.shape[0]
    scale = gsz / max(Werp, Herp)
    nw, nh = round(Werp * scale), round(Herp * scale)
    padx, pady = (gsz - nw) // 2, (gsz - nh) // 2
    for i, m in enumerate(metas):
        lon, lat = tile_footprint_lonlat(m.yaw_deg, m.pitch_deg, m.hfov_deg, m.vfov_deg)
        jump = np.abs(np.diff(lon)) > 180.0                          # split the +/-180 seam
        lon_s = lon.astype(float).copy(); lon_s[1:][jump] = np.nan
        a1.plot(lon_s, lat, lw=1.1, color=colors[i % len(colors)])
        gx = (lon + 180) / 360 * nw + padx
        gy = (90 - lat) / 180 * nh + pady
        gx_s = gx.copy(); gx_s[1:][jump] = np.nan
        axg.plot(gx_s, gy, lw=1.1, color=colors[i % len(colors)])
        axg.plot((m.yaw_deg + 180) / 360 * nw + padx, (90 - m.pitch_deg) / 180 * nh + pady,
                 ".", ms=4, color=colors[i % len(colors)])

    # coverage maps for the actual anyres tile plan (same panels as the coverage viz)
    ah = fig.add_subplot(gs[1])
    imh = ah.imshow(hit, extent=ext, aspect="auto", cmap="viridis", vmin=0)
    ad = fig.add_subplot(gs[2])
    cmap = plt.cm.inferno.copy(); cmap.set_bad("lightgray")
    vmax = float(np.nanpercentile(dm, 99)) if np.isfinite(dm).any() else 8.0
    imd = ad.imshow(dm, extent=ext, aspect="auto", cmap=cmap, vmin=1.0, vmax=vmax)
    for ax in (ah, ad):
        for p in pitches:
            ax.axhline(p, color="cyan", lw=0.8, alpha=0.6)
        ax.axhline(0, color="yellow", lw=1.6)
        ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="yaw", ylabel="pitch")
    ah.set_title(f"hit-count — min={hit.min()} mean={hit.mean():.2f} max={hit.max()}", fontsize=9)
    ad.set_title("best-view distortion — min 1/cos³θ (gray = uncovered)", fontsize=9)
    fig.colorbar(imh, ax=ah, location="right", shrink=0.85, label="# tiles")
    fig.colorbar(imd, ax=ad, location="right", shrink=0.85, label="magnif ×")

    sub = gs[3].subgridspec(len(pitches), ncol)
    for r, p in enumerate(pitches):
        ids = rows[p]; start = (ncol - len(ids)) // 2
        for c in range(ncol):
            ax = fig.add_subplot(sub[r, c]); ax.axis("off")
            j = c - start
            if 0 <= j < len(ids):
                img = tiles[ids[j]].permute(1, 2, 0).cpu().numpy()
                ax.imshow(np.clip(img, 0, 1))
                ax.set_title(f"y{metas[ids[j]].yaw_deg:+.0f} p{p:+.0f}", fontsize=6,
                             color=colors[ids[j] % len(colors)])

    extra = f"  pmax=±{pmax:g}°" if mode == "band" else ""
    fig.suptitle(f"anyres_e2p production output   mode={mode}   hfov={hfov}° vfov={vfov}° overlap={ov}{extra}",
                 fontsize=12)
    fig.savefig(out, dpi=120)
    print("saved", out, "|", {k: g[k] for k in ("mode", "n_tiles", "n_caps", "ring_tile_counts")})


def main():
    erp_path, pos = parse_args(sys.argv[1:])
    hfov = float(pos[0]) if pos else 90.0
    vfov = float(pos[1]) if len(pos) > 1 else 90.0
    ov = float(pos[2]) if len(pos) > 2 else 0.5
    pmax = float(pos[3]) if len(pos) > 3 else 45.0

    erp_img = Image.open(erp_path).convert("RGB") if os.path.exists(erp_path) else Image.new("RGB", (2048, 1024))
    erp_small = np.asarray(erp_img.resize((720, 360), Image.BILINEAR))
    for mode in ("full_sphere", "band"):
        viz(erp_img, erp_small, mode, hfov, vfov, ov, pmax,
            f"docs/figures/visualize_anyres/anyres_{mode}_{hfov:g}_{vfov:g}_{ov:g}.png")


if __name__ == "__main__":
    main()
