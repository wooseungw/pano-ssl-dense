"""Gnomonic distortion INSIDE a pinhole tile (radial 1/cos^3 theta), shown on real crops.

Distortion depends only on off-axis angle -> identical for every tile of a given FOV.
Run:  python scripts/pinhole_distortion.py [hfov] [vfov] [overlap] [--erp PATH]
"""

import math
import os
import sys

import py360convert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coverage_common import parse_args, build_tiles, load_erp, tile_distortion


def main():
    erp_path, pos = parse_args(sys.argv[1:])
    hfov = float(pos[0]) if pos else 90.0
    vfov = float(pos[1]) if len(pos) > 1 else 90.0
    ov = float(pos[2]) if len(pos) > 2 else 0.2
    S = 384

    erp = load_erp(erp_path, (2048, 1024))
    tiles = build_tiles(hfov, vfov, ov, ov)
    eqt = min(tiles, key=lambda t: abs(t.pitch))           # equator-most tile
    polt = max(tiles, key=lambda t: abs(t.pitch))          # pole-most tile
    crop = lambda t: py360convert.e2p(erp, (hfov, vfov), t.yaw, t.pitch,
                                      out_hw=(S, S), mode="bilinear").astype("uint8")

    D = tile_distortion(S, hfov, vfov)
    levels = [1.25, 1.5, 2.0, 3.0]
    he = 1 / math.cos(math.radians(hfov / 2)) ** 3
    ve = 1 / math.cos(math.radians(vfov / 2)) ** 3
    thc = math.atan(math.hypot(math.tan(math.radians(hfov / 2)), math.tan(math.radians(vfov / 2))))
    co = 1 / math.cos(thc) ** 3
    print(f"hfov={hfov} vfov={vfov}: center=1.00x  h-edge={he:.2f}x  v-edge={ve:.2f}x  corner={co:.2f}x")

    fig, axs = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    panels = [(crop(eqt), f"crop  y{eqt.yaw:.0f} p{eqt.pitch:.0f}"),
              (crop(eqt), "same crop + iso-distortion lines"),
              (crop(polt), f"crop  y{polt.yaw:.0f} p{polt.pitch:.0f} + lines")]
    for ax, (img, ttl) in zip(axs[:3], panels):
        ax.imshow(img); ax.set_title(ttl, fontsize=10); ax.axis("off")
    for ax in axs[1:3]:
        cs = ax.contour(D, levels=levels, colors="cyan", linewidths=1.3)
        ax.clabel(cs, fmt="%.2fx", fontsize=8)
    im = axs[3].imshow(D, cmap="inferno", vmin=1.0)
    cs = axs[3].contour(D, levels=levels, colors="cyan", linewidths=1.0)
    axs[3].clabel(cs, fmt="%.2fx", fontsize=8)
    axs[3].set_title("distortion map  1/cos³θ"); axs[3].axis("off")
    fig.colorbar(im, ax=axs[3], fraction=0.046, label="area magnification ×")

    fig.suptitle(f"Pinhole-image distortion   hfov={hfov}°  vfov={vfov}°   "
                 f"|  center 1.0×,  edge {he:.1f}×/{ve:.1f}×,  corner {co:.1f}×", fontsize=12)
    out = f"docs/figures/pinhole_distortion/pinhole_distortion_{hfov:g}_{vfov:g}.png"
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    main()
