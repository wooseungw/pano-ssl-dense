"""Hard differential gate for the explicit coordinate and nearest-warp helpers."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("DEV", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import py360convert
import pytest

import geometry
import scripts.bench_common as bench_common
import scripts.probe_normal as probe_normal
import scripts.seg_s2d3d_bench as seg_bench


# Captured directly from the four read-only clones with their import defaults:
# HFOV=65.0, TILE=512, TILE_OUT=256, STITCH_HW=(128,256). The warp input is
# RandomState(0).rand(512,1024,3).astype(float32), sampled to a 32x32 grid.
FROZEN = {
    "bench_coord": {
        (0, -90): "06ff61f4d65c762f5e37d8cd831cacd1d28bbdcf8a0009c0527746392a2ba44f",
        (0, 0): "f6fabd8fc7b5cd23a04d68bcdc2f878201c7a067e347af9b2258e985a27817e9",
        (0, 45): "04a0565fb21f430dd1306a2dd226c708cceab3145dd6db2fca16e7a98f76fa12",
        (0, 90): "71712dd035dfb451b110ae1cc98dbb03eb5ca8ee6e56b7417f6e20ffbfb08e35",
        (180, -90): "a636590dbd17bd06dc18d38ae1b3f95f2775d5e4e6d7e463d1501040ed52bdf8",
        (180, 0): "df39944f4f48a40d2c535eb88324d69aa5577c6d844dc3d660a3217609fcb85e",
        (180, 45): "ff4cb44e73bc38d073bbf5e608f0c8016a7af13ca3424a8568d691a454b6bbaf",
        (180, 90): "3389ceb9f254cd02ffa458dfaf378daf179d6b88546fa7768114a15d2c46ea29",
        (359, -90): "e0646d3b841e0910723fd58bfbb7f566aa060299581015ee7252dc19e98ddadf",
        (359, 0): "37ff31f920e289d371318f219aa1955e11b89104cf8b1b437c100fa731673b8b",
        (359, 45): "d50f408f3baf8aea9568ea16d4de4cfddde6159b633b2f985c454608298eb456",
        (359, 90): "19939f77e5cb5a958bb9b6c6659a66750181ff9e87b5244ae771488212801ba1",
    },
    "seg_coord": {
        (0, -90): "06ff61f4d65c762f5e37d8cd831cacd1d28bbdcf8a0009c0527746392a2ba44f",
        (0, 0): "f6fabd8fc7b5cd23a04d68bcdc2f878201c7a067e347af9b2258e985a27817e9",
        (0, 45): "04a0565fb21f430dd1306a2dd226c708cceab3145dd6db2fca16e7a98f76fa12",
        (0, 90): "71712dd035dfb451b110ae1cc98dbb03eb5ca8ee6e56b7417f6e20ffbfb08e35",
        (180, -90): "a636590dbd17bd06dc18d38ae1b3f95f2775d5e4e6d7e463d1501040ed52bdf8",
        (180, 0): "df39944f4f48a40d2c535eb88324d69aa5577c6d844dc3d660a3217609fcb85e",
        (180, 45): "ff4cb44e73bc38d073bbf5e608f0c8016a7af13ca3424a8568d691a454b6bbaf",
        (180, 90): "3389ceb9f254cd02ffa458dfaf378daf179d6b88546fa7768114a15d2c46ea29",
        (359, -90): "e0646d3b841e0910723fd58bfbb7f566aa060299581015ee7252dc19e98ddadf",
        (359, 0): "37ff31f920e289d371318f219aa1955e11b89104cf8b1b437c100fa731673b8b",
        (359, 45): "d50f408f3baf8aea9568ea16d4de4cfddde6159b633b2f985c454608298eb456",
        (359, 90): "19939f77e5cb5a958bb9b6c6659a66750181ff9e87b5244ae771488212801ba1",
    },
    "bench_warp": {
        (0, -90): "236ee5b7ed4011f8947022da3913667a76ee67fb9d8c75ae20845cb06c44a05f",
        (0, 0): "24ecfb41efd6817d2aae76a8f143b50e5a6fe5978df275c24764badfbef2de1b",
        (0, 45): "16538929a674b0c926ea2c785ab703a00962612adc90f8274606096457a2e97b",
        (0, 90): "d5000652ab3d97ab3931a3d23cc1788283432724e97bd1db01abf06b8bdd2188",
        (180, -90): "f9b12c0cc6c724de1a6ec99d07d3a0a7249659af8d052822619633ecea6f2dfe",
        (180, 0): "fa81b7292af2c524c2a22246555627334041665252143796ec03302d384e64c2",
        (180, 45): "89c1c4388d18480a7736cfb4c8513661ed864ba6fd762389b65c321844c225c6",
        (180, 90): "02aa400f27944aad30383f659060ab37d4bd7398898a2e814ded6d74292801de",
        (359, -90): "03fea7a4260254e64b3c360871f7ca4f22b9e138b6cf0358fd48aba6d3d98f0a",
        (359, 0): "1e716ddebc82345844ce5d710c84b14f0d6a83477661fd724627ded0e53f7173",
        (359, 45): "f1f9a0855754c93db44a078eda1fdea8a97365c1e970628fb30c70fb49b9f535",
        (359, 90): "7a87e710d4b3edffcc353a51b412e86ab3131b5b885d93789a902170759dcafd",
    },
    "probe_warp": {
        (0, -90): "236ee5b7ed4011f8947022da3913667a76ee67fb9d8c75ae20845cb06c44a05f",
        (0, 0): "24ecfb41efd6817d2aae76a8f143b50e5a6fe5978df275c24764badfbef2de1b",
        (0, 45): "16538929a674b0c926ea2c785ab703a00962612adc90f8274606096457a2e97b",
        (0, 90): "d5000652ab3d97ab3931a3d23cc1788283432724e97bd1db01abf06b8bdd2188",
        (180, -90): "f9b12c0cc6c724de1a6ec99d07d3a0a7249659af8d052822619633ecea6f2dfe",
        (180, 0): "fa81b7292af2c524c2a22246555627334041665252143796ec03302d384e64c2",
        (180, 45): "89c1c4388d18480a7736cfb4c8513661ed864ba6fd762389b65c321844c225c6",
        (180, 90): "02aa400f27944aad30383f659060ab37d4bd7398898a2e814ded6d74292801de",
        (359, -90): "03fea7a4260254e64b3c360871f7ca4f22b9e138b6cf0358fd48aba6d3d98f0a",
        (359, 0): "1e716ddebc82345844ce5d710c84b14f0d6a83477661fd724627ded0e53f7173",
        (359, 45): "f1f9a0855754c93db44a078eda1fdea8a97365c1e970628fb30c70fb49b9f535",
        (359, 90): "7a87e710d4b3edffcc353a51b412e86ab3131b5b885d93789a902170759dcafd",
    },
}

YAW_PITCH = tuple((yaw, pitch) for yaw in (0, 180, 359) for pitch in (-90, 0, 45, 90))


@pytest.fixture(scope="module")
def synthetic_erp():
    return np.random.RandomState(0).rand(512, 1024, 3).astype(np.float32)


def _sha256(arr):
    return hashlib.sha256(np.asarray(arr).tobytes()).hexdigest()


def _coord(yaw, pitch):
    return geometry.coord_cell_map(512, 1024, yaw, pitch, 65.0, 256, 128, 256)


def _warp(arr, yaw, pitch, gh=32, gw=32):
    return geometry.warp_nearest_centers(arr, yaw, pitch, 65.0, 512, gh, gw)


def test_clone_import_defaults_are_the_frozen_configuration():
    assert (bench_common.HFOV, bench_common.TILE, bench_common.TILE_OUT) == (65.0, 512, 256)
    assert (bench_common.SH, bench_common.SW) == (128, 256)
    assert probe_normal.TILE == 512
    assert (seg_bench.HFOV, seg_bench.TILE, seg_bench.TILE_OUT) == (65.0, 512, 256)
    assert (seg_bench.SH, seg_bench.SW) == (128, 256)


@pytest.mark.parametrize("yaw", (0, 180, 359))
def test_seam_frozen_and_360_periodicity(synthetic_erp, yaw):
    coord = _coord(yaw, 0)
    warp = _warp(synthetic_erp, yaw, 0)
    assert _sha256(coord) == FROZEN["bench_coord"][(yaw, 0)]
    assert _sha256(coord) == FROZEN["seg_coord"][(yaw, 0)]
    assert _sha256(warp) == FROZEN["bench_warp"][(yaw, 0)]
    assert _sha256(warp) == FROZEN["probe_warp"][(yaw, 0)]
    if yaw == 0:
        np.testing.assert_array_equal(_coord(360, 0), coord)
        np.testing.assert_array_equal(_warp(synthetic_erp, 360, 0), warp)
        assert _sha256(_coord(360, 0)) == FROZEN["bench_coord"][(0, 0)]
        assert _sha256(_warp(synthetic_erp, 360, 0)) == FROZEN["bench_warp"][(0, 0)]


@pytest.mark.parametrize("yaw,pitch", YAW_PITCH)
def test_pitch_grid_matches_all_four_frozen_clones(synthetic_erp, yaw, pitch):
    coord_hash = _sha256(_coord(yaw, pitch))
    warp_hash = _sha256(_warp(synthetic_erp, yaw, pitch))
    assert coord_hash == FROZEN["bench_coord"][(yaw, pitch)]
    assert coord_hash == FROZEN["seg_coord"][(yaw, pitch)]
    assert warp_hash == FROZEN["bench_warp"][(yaw, pitch)]
    assert warp_hash == FROZEN["probe_warp"][(yaw, pitch)]


@pytest.mark.parametrize("yaw,pitch", ((0, -90), (37, 23), (359, 45), (180, 90)))
def test_arbitrary_resolution_matches_render_coordmap_oracle(yaw, pitch):
    erp_h, erp_w, out_size = 256, 512, 64
    stitch_h, stitch_w = 73, 149
    cmap = geometry.render_coordmap(erp_h, erp_w, yaw, pitch, 71.0, out_size)
    expected_x = np.clip(
        (cmap[..., 0] / erp_w * stitch_w).astype(int), 0, stitch_w - 1
    )
    expected_y = np.clip(
        (cmap[..., 1] / erp_h * stitch_h).astype(int), 0, stitch_h - 1
    )
    expected = (expected_y * stitch_w + expected_x).reshape(-1).astype(np.int64)
    actual = geometry.coord_cell_map(
        erp_h, erp_w, yaw, pitch, 71.0, out_size, stitch_h, stitch_w
    )
    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, expected)


def test_patch_center_formula_and_frozen_hash(synthetic_erp):
    warped = py360convert.e2p(
        synthetic_erp.astype(np.float32), 65.0, 359, 45,
        out_hw=(512, 512), mode="nearest",
    )
    cy = ((np.arange(32) + 0.5) * 512 / 32).astype(int)
    cx = ((np.arange(32) + 0.5) * 512 / 32).astype(int)
    expected = warped[np.ix_(cy, cx)].reshape(32, 32, 3)
    actual = _warp(synthetic_erp, 359, 45)
    np.testing.assert_array_equal(actual, expected)
    assert _sha256(actual) == FROZEN["bench_warp"][(359, 45)]
    assert _sha256(actual) == FROZEN["probe_warp"][(359, 45)]


def test_rectangular_grid_uses_independent_row_and_column_centers(synthetic_erp):
    gh, gw = 96, 192
    warped = py360convert.e2p(
        synthetic_erp.astype(np.float32), 65.0, 180, -45,
        out_hw=(512, 512), mode="nearest",
    )
    cy = ((np.arange(gh) + 0.5) * 512 / gh).astype(int)
    cx = ((np.arange(gw) + 0.5) * 512 / gw).astype(int)
    expected = warped[np.ix_(cy, cx)].reshape(gh, gw, 3)
    actual = _warp(synthetic_erp, 180, -45, gh, gw)
    clone = probe_normal.warp_to_grid(synthetic_erp, 180, -45, 65.0, gh, gw, 3)
    assert actual.shape == (gh, gw, 3)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, clone)
