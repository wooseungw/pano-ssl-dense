"""Side-by-side comparison: full-sphere (caps) vs equator-centered band, same FOV/overlap.

Renders both hit-count maps (shared scale, ring-line overlay) plus the per-|latitude|
redundancy curve, so the pole-redundancy difference is visible.

Run:  python scripts/compare_coverage.py [hfov] [vfov] [overlap] [pmax] [--erp PATH]
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coverage_common import (parse_args, build_tiles, build_band_tiles,
                             world_dirs, hitmap, load_erp, CW, CH)


def main():
    erp_path, pos = parse_args(sys.argv[1:])
    hfov = float(pos[0]) if pos else 90.0
    vfov = float(pos[1]) if len(pos) > 1 else 90.0
    ov = float(pos[2]) if len(pos) > 2 else 0.5
    pmax = float(pos[3]) if len(pos) > 3 else 45.0

    d = world_dirs()
    latc = (0.5 - (np.arange(CH) + 0.5) / CH) * 180
    w = np.cos(np.radians(latc))
    awm = lambda h: float((h.mean(1) * w).sum() / w.sum())

    full = build_tiles(hfov, vfov, ov, ov)
    band = build_band_tiles(hfov, vfov, ov, ov, pmax)
    hf, hb = hitmap(full, d), hitmap(band, d)
    print(f"FULL: {len(full)} tiles  area-wt redundancy={awm(hf):.2f}")
    print(f"BAND: {len(band)} tiles  area-wt redundancy={awm(hb):.2f}")

    ext = [-180, 180, -90, 90]
    vmax = max(int(hf.max()), int(hb.max()))
    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])

    def ring_lines(ax, tiles):
        for p in sorted({t.pitch for t in tiles}):
            ax.axhline(p, color="cyan", lw=0.8, alpha=0.6)
        ax.axhline(0, color="yellow", lw=1.4)
    eq_tag = lambda ts: "ring on equator" if any(abs(t.pitch) < 1e-6 for t in ts) else "no ring at equator"

    a1 = fig.add_subplot(gs[0, 0])
    a1.imshow(hf, extent=ext, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    ring_lines(a1, full)
    a1.set_title(f"FULL-SPHERE (caps)  {len(full)} tiles  area-wt={awm(hf):.2f}  ({eq_tag(full)})", fontsize=10)
    a1.set(xlabel="yaw", ylabel="pitch")

    a2 = fig.add_subplot(gs[0, 1])
    im = a2.imshow(hb, extent=ext, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    ring_lines(a2, band)
    a2.set_title(f"EQUATOR-CENTERED BAND (pmax=±{pmax:g})  {len(band)} tiles  area-wt={awm(hb):.2f}  ({eq_tag(band)})", fontsize=10)
    a2.set(xlabel="yaw", ylabel="pitch")
    fig.colorbar(im, ax=[a1, a2], location="right", shrink=0.8, label="# tiles (shared scale)")

    a3 = fig.add_subplot(gs[1, :])
    al = np.abs(latc); o = np.argsort(al)
    a3.plot(al[o], hf.mean(1)[o], "-o", ms=3, color="#d62728", label=f"full ({len(full)} tiles)")
    a3.plot(al[o], hb.mean(1)[o], "-s", ms=3, color="#1f77b4", label=f"band ±{pmax:g} ({len(band)} tiles)")
    a3.axvline(pmax, color="gray", ls="--", lw=0.8)
    a3.set(xlabel="|latitude|  (0=equator, 90=pole)", ylabel="mean redundancy (hits)", xlim=(0, 90),
           title="Per-latitude redundancy — full piles tiles toward the poles; band keeps them mid-latitude")
    a3.grid(alpha=0.3); a3.legend()

    fig.suptitle(f"Coverage comparison   hfov={hfov}°  vfov={vfov}°  overlap={ov}", fontsize=13)
    out = f"docs/figures/compare_coverage/compare_{hfov:g}_{vfov:g}_{ov:g}_p{pmax:g}.png"
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    main()
