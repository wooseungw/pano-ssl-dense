"""Equator-centered mid-latitude band (no caps). Same layout as visualize_coverage.

Run:  python scripts/band_coverage.py [hfov] [vfov] [overlap] [pmax] [--erp PATH]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coverage_common import parse_args, build_band_tiles, hitmap, plot_coverage, CH


def main():
    erp_path, pos = parse_args(sys.argv[1:])
    hfov = float(pos[0]) if pos else 90.0
    vfov = float(pos[1]) if len(pos) > 1 else 90.0
    ov = float(pos[2]) if len(pos) > 2 else 0.2
    pmax = float(pos[3]) if len(pos) > 3 else 45.0

    tiles = build_band_tiles(hfov, vfov, ov, ov, pmax)
    hit = hitmap(tiles)
    latc = (0.5 - (np.arange(CH) + 0.5) / CH) * 180
    band = np.abs(latc) <= pmax
    cov = min(90.0, pmax + vfov / 2)
    print(f"hfov={hfov} vfov={vfov} overlap={ov} pmax=±{pmax} -> {len(tiles)} tiles  "
          f"band|lat|<={pmax}: min={hit[band].min()} mean={hit[band].mean():.2f}  (covers |lat|<~{cov:.0f}°)")
    plot_coverage(erp_path, tiles, hit,
                  f"Mid-latitude band   hfov={hfov}°  vfov={vfov}°  overlap={ov}  pmax=±{pmax}°"
                  f"  (poles {'excluded' if cov < 90 else 'reached'})",
                  f"docs/figures/band_coverage/band_coverage_{hfov:g}_{vfov:g}_{ov:g}_p{pmax:g}.png")


if __name__ == "__main__":
    main()
