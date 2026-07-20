import os
import sys

os.environ.setdefault("DEV", "cpu")

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data  # noqa: E402


def test_load_s2d3d_depth(tmp_path):
    rgb_dir = tmp_path / "pano" / "rgb"
    depth_dir = tmp_path / "pano" / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)

    rgb_path = rgb_dir / "x_rgb.png"
    depth_path = depth_dir / "x_depth.png"
    Image.new("RGB", (5, 1)).save(rgb_path)
    raw = np.array([[512, 5120, 5121, 0, 65535]], dtype=np.uint16)
    Image.fromarray(raw).save(depth_path)

    depth_m, valid = data.load_s2d3d_depth(str(rgb_path), raw.shape, cap=10.0)

    np.testing.assert_array_equal(depth_m, raw.astype(np.float32) / 512.0)
    np.testing.assert_array_equal(
        valid, np.array([[1, 1, 0, 0, 0]], dtype=np.float32)
    )
    assert depth_m.dtype == np.float32
    assert valid.dtype == np.float32


def test_load_s2d3d_depth_requires_cap():
    with pytest.raises(TypeError):
        data.load_s2d3d_depth("unused_rgb.png", (1, 1))
