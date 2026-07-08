"""Geometric-correctness tests for geometry.tile_position_labels (PWW position labels).

The labels are the free supervision for the position pretext; if they are wrong the whole
signal is wrong. Assert exact geometric properties against known tile geometry.
"""
import numpy as np
import pytest

import geometry as G

ERP_H, ERP_W, OUT, PATCH = 1024, 2048, 512, 16
GN = OUT // PATCH  # 32


def _labels(yaw=0.0, pitch=0.0, hfov=65.0):
    return G.tile_position_labels(ERP_H, ERP_W, yaw, pitch, hfov, OUT, PATCH)


def test_shapes_and_grid():
    lat, dlon, clat, fov = _labels()
    assert lat.shape == (GN, GN) and dlon.shape == (GN, GN)
    assert fov == 65.0
    assert lat.dtype == np.float32 and dlon.dtype == np.float32


def test_centre_latitude_equals_pitch():
    for pitch in (-45.0, 0.0, 30.0, 45.0):
        _, _, clat, _ = _labels(pitch=pitch)
        assert abs(clat - pitch) < 2.5, f"centre_lat {clat} != pitch {pitch}"


def test_latitude_monotonic_down_rows():
    # top rows look UP (higher latitude) than bottom rows, at the centre column.
    lat, _, _, _ = _labels(pitch=0.0)
    col = GN // 2
    assert lat[0, col] > lat[GN // 2, col] > lat[GN - 1, col]
    # and the centre cell is ~horizon
    assert abs(lat[GN // 2, col]) < 3.0


def test_relative_longitude_centre_and_sign():
    _, dlon, _, _ = _labels(pitch=0.0)
    row = GN // 2
    assert abs(dlon[row, GN // 2]) < 3.0                    # ~0 at tile centre
    assert dlon[row, 0] * dlon[row, GN - 1] < 0             # opposite signs across the tile
    assert abs(dlon[row, 0]) > 5.0 and abs(dlon[row, -1]) > 5.0


def test_vertical_span_matches_fov():
    # square pinhole: vertical FOV == hfov; latitude span across the centre column ~ hfov.
    for hfov in (45.0, 65.0, 85.0):
        lat, _, _, _ = _labels(pitch=0.0, hfov=hfov)
        span = lat[0, GN // 2] - lat[GN - 1, GN // 2]
        assert 0.8 * hfov < span < 1.2 * hfov, f"span {span} vs hfov {hfov}"


def test_seam_wrap_is_bounded():
    # a tile straddling the +/-180 longitude seam must NOT produce +/-360 dlon jumps.
    for yaw in (170.0, -175.0, 179.0):
        _, dlon, _, hfov = _labels(yaw=yaw, hfov=65.0)
        assert np.abs(dlon).max() < hfov, f"dlon overflow at yaw={yaw}: max={np.abs(dlon).max()}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
