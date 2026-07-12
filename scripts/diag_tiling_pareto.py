"""Tiling Pareto diagnostic — coverage / overlap-multiplicity / distortion vs tile count.

The minimal-tiling question, formalized: choose tile set {(yaw_i, pitch_i, hfov_i)} minimizing N
(compute = N ViT forwards) subject to
  (i)   full-sphere coverage (every direction inside >=1 gnomonic square),
  (ii)  distortion budget: worst-case best-view off-axis cos >= c_min
        (Gate B measured the content-controlled cost of obliquity as MARGINAL, so c_min can be low),
  (iii) overlap-graph connectivity + multiplicity >= m for the SSL/fusion signal
        (consistency losses, Gate-A context, and the +0.088 blend surplus all LIVE in overlaps).

This script evaluates candidate closed-form families on 20k Fibonacci-sphere directions:
  coverage, mean multiplicity, frac covered >=2 (SSL-usable), worst best-view off-axis cos,
  and relative compute (N * tile_pixels ratio vs the 24-tile champion).

Run: conda run -n pano python scripts/diag_tiling_pareto.py   (CPU, seconds)
"""
from __future__ import annotations

import numpy as np


def fib_sphere(n=20000):
    i = np.arange(n) + 0.5
    lat = np.arcsin(1 - 2 * i / n)
    lon = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(lat) * np.cos(lon), np.sin(lat), np.cos(lat) * np.sin(lon)], -1)


def tile_axes(yaw, pitch):
    y, p = np.deg2rad(yaw), np.deg2rad(pitch)
    fwd = np.array([np.cos(p) * np.cos(y), np.sin(p), np.cos(p) * np.sin(y)])
    up0 = np.array([0.0, 1.0, 0.0])
    right = np.cross(up0, fwd)
    nr = np.linalg.norm(right)
    if nr < 1e-6:                                   # pole tile: pick any horizontal right
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / nr
    up = np.cross(fwd, right)
    return fwd, right, up


def inside(dirs, spec):
    yaw, pitch, hfov = spec
    fwd, right, up = tile_axes(yaw, pitch)
    z = dirs @ fwd
    m = z > 1e-6
    t = np.tan(np.deg2rad(hfov) / 2.0)
    x = (dirs @ right) / np.maximum(z, 1e-9)
    y = (dirs @ up) / np.maximum(z, 1e-9)
    return m & (np.abs(x) <= t) & (np.abs(y) <= t), z


def ring(pitches, k, hfov, poles=None):
    specs = [(360.0 * j / k - 180.0, p, hfov) for p in pitches for j in range(k)]
    if poles:
        specs += [(0.0, 90.0, poles), (0.0, -90.0, poles)]
    return specs


CONFIGS = {
    "champion 3x8 @65":         ring((-45.0, 0.0, 45.0), 8, 65.0),
    "3x6 @75":                  ring((-45.0, 0.0, 45.0), 6, 75.0),
    "2x6 @80 (+poles@80)":      ring((-32.0, 32.0), 6, 80.0, poles=80.0),
    "2x5 @90 (+poles@90)":      ring((-35.0, 35.0), 5, 90.0, poles=90.0),
    "cubemap 6 @90":            [(-180.0, 0.0, 90.0), (-90.0, 0.0, 90.0), (0.0, 0.0, 90.0),
                                 (90.0, 0.0, 90.0), (0.0, 90.0, 90.0), (0.0, -90.0, 90.0)],
    "cubemap-rot 6 @100":       [(-180.0, 0.0, 100.0), (-90.0, 0.0, 100.0), (0.0, 0.0, 100.0),
                                 (90.0, 0.0, 100.0), (45.0, 90.0, 100.0), (45.0, -90.0, 100.0)],
    "1x4 @100 (+poles@100)":    ring((0.0,), 4, 100.0, poles=100.0),
    "octa 8 @85":               [(-180.0 + 90.0 * j, s * 35.26, 85.0)
                                 for s in (-1, 1) for j in range(4)],
}

dirs = fib_sphere()
base_cost = 24 * 1.0
print(f"{'config':26s}{'N':>4}{'cover%':>8}{'>=2%':>7}{'meanMult':>9}{'worst-cos':>10}{'rel-cost':>9}")
for name, specs in CONFIGS.items():
    cnt = np.zeros(len(dirs))
    best = np.zeros(len(dirs))
    for spec in specs:
        m, z = inside(dirs, spec)
        cnt += m
        best = np.where(m, np.maximum(best, z), best)
    cov = float((cnt >= 1).mean())
    cov2 = float((cnt >= 2).mean())
    mult = float(cnt[cnt >= 1].mean())
    wc = float(best[cnt >= 1].min())
    cost = len(specs) / base_cost
    print(f"{name:26s}{len(specs):>4}{cov * 100:>8.2f}{cov2 * 100:>7.1f}{mult:>9.2f}{wc:>10.3f}"
          f"{cost:>9.2f}")
print("\nreading: worst-cos = worst direction's BEST-view off-axis cos (distortion budget); "
      "Gate B says the representation cost of low cos is marginal, so configs with worst-cos "
      ">= ~0.5 and >=2% high enough for the SSL overlap signal are candidates for the Pareto set.")
