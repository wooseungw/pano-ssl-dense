"""Full-sphere multi-pinhole coverage of an ERP: tiling + hit-count + distortion + crops.

Run:  python scripts/visualize_coverage.py [hfov] [vfov] [overlap] [--erp PATH]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coverage_common import parse_args, build_tiles, hitmap, plot_coverage


def main():
    erp_path, pos = parse_args(sys.argv[1:])
    hfov = float(pos[0]) if pos else 90.0
    vfov = float(pos[1]) if len(pos) > 1 else 90.0
    ov = float(pos[2]) if len(pos) > 2 else 0.2

    tiles = build_tiles(hfov, vfov, ov, ov)
    hit = hitmap(tiles)
    print(f"hfov={hfov} vfov={vfov} overlap={ov} -> {len(tiles)} tiles  "
          f"min={hit.min()} mean={hit.mean():.2f} max={hit.max()} "
          f"{'FULL' if hit.min() >= 1 else 'GAP'}")
    plot_coverage(erp_path, tiles, hit,
                  f"Full-sphere coverage   hfov={hfov}°  vfov={vfov}°  overlap={ov}",
                  "docs/figures/visualize_coverage/coverage_viz.png")


if __name__ == "__main__":
    main()
