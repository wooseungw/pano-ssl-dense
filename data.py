"""ERP panorama dataset listing for SSL + dense eval.

Stanford2D3D is the primary source: 1155 ERP panos at 4096x2048 with aligned dense GT
(semantic/depth/normal/global_xyz pointmap/pose) -> covers all 4 downstream tasks.
SUN360 is kept only for quick smoke checks. Structured3D is the scale source
(extracted to structured3d/images/Structured3D/scene_*, ~21k scenes, depth/normal/
semantic GT per room). quic360 (refer360-derived) provides image-text region-grounding
pairs for the CLIP-style grounding task.
"""

from __future__ import annotations

import csv
import glob
import os
from typing import List, Optional, Tuple

ROOT = os.environ.get("PANO_DATA_ROOT", "/mnt/data-hdd/datasets/pano-ssl")

# Stanford2D3D pano modality file-suffix per modality (same stem, different suffix/ext).
S2D3D_SUFFIX = {
    "rgb": "rgb.png",
    "semantic": "semantic.png",
    "depth": "depth.png",
    "normal": "normals.png",
    "xyz": "global_xyz.exr",
    "pose": "pose.json",
}


def list_erps(name: str = "stanford2d3d", limit: Optional[int] = None) -> List[str]:
    """Return ERP RGB file paths for the given dataset."""
    if name == "sun360":
        fs = sorted(glob.glob(f"{ROOT}/SUN360/test/RGB/*.jpg"))
    elif name == "stanford2d3d":
        fs = sorted(glob.glob(f"{ROOT}/stanford2d3d/extracted_data/*/*/pano/rgb/*_rgb.png"))
    elif name == "structured3d":
        fs = list_structured3d()  # extracted panos; empty until extraction finishes
    else:
        raise ValueError(f"unknown dataset {name!r}")
    return fs[:limit] if limit else fs


def s2d3d_gt_path(rgb_path: str, modality: str) -> str:
    """Map a Stanford2D3D pano RGB path to its aligned GT path for `modality`."""
    suffix = S2D3D_SUFFIX[modality]
    p = rgb_path.replace("/rgb/", f"/{modality}/").replace("rgb.png", suffix)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


# --- Structured3D (scale source; depth/normal/semantic GT aligned per room) ---

S3D_IMAGES = f"{ROOT}/structured3d/images"
S3D_DIR = f"{S3D_IMAGES}/Structured3D"


def _s3d_corrupt_dirs() -> set:
    """Room/config dirs (rel to images/) holding CRC-corrupt files from a bad
    download, as detected by 7z and listed in structured3d/corrupt_files.txt.
    Listings drop these rooms so callers never read a corrupt RGB or its GT."""
    f = f"{ROOT}/structured3d/corrupt_files.txt"
    if not os.path.exists(f):
        return set()
    with open(f) as fh:
        return {os.path.dirname(ln.strip()) for ln in fh if ln.strip()}


def list_structured3d(
    limit: Optional[int] = None, lighting: str = "rawlight", config: str = "full"
) -> List[str]:
    """Return Structured3D ERP RGB paths (one per room rendering view).

    Needs extraction first (structured3d/images/Structured3D/scene_*); returns []
    until then. `lighting` in {rawlight, coldlight, warmlight}; `config` in
    {full, empty, simple}. Aligned depth/normal/semantic live in the same dir
    (s3d_gt_path). Rooms with known-corrupt files (corrupt_files.txt) are skipped.
    """
    bad = _s3d_corrupt_dirs()
    fs = []
    for p in sorted(
        glob.glob(f"{S3D_DIR}/scene_*/2D_rendering/*/panorama/{config}/rgb_{lighting}.png")
    ):
        if os.path.relpath(os.path.dirname(p), S3D_IMAGES) in bad:
            continue
        fs.append(p)
    return fs[:limit] if limit else fs


def s3d_gt_path(rgb_path: str, modality: str) -> str:
    """Map a Structured3D pano rgb_*light.png to its aligned GT in the same dir.

    `modality` in {depth, normal, semantic, albedo} (-> depth.png etc). The
    semantic.png is a per-pixel class-index map = the region-grounding substrate
    (region = category mask); index->name uses Structured3D's official label list.
    """
    p = os.path.join(os.path.dirname(rgb_path), f"{modality}.png")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


# --- DensePASS (real outdoor panoramic seg; Cityscapes 19-cls trainId) ---

DENSEPASS_DIR = f"{ROOT}/densepass/DensePASS"


def list_densepass(limit: Optional[int] = None) -> List[str]:
    """DensePASS RGB paths (100 real outdoor 360 strips, 400x2048)."""
    fs = sorted(glob.glob(f"{DENSEPASS_DIR}/leftImg8bit/val/*.png"))
    return fs[:limit] if limit else fs


def densepass_gt_path(img_path: str) -> str:
    """leftImg8bit/val/<id>_.png -> gtFine/val/<id>_labelTrainIds.png (uint8 0-18, 255=ignore)."""
    name = os.path.basename(img_path).replace(".png", "labelTrainIds.png")
    p = f"{DENSEPASS_DIR}/gtFine/val/{name}"
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


# --- Region-grounding image-text pairs (CLIP-style grounding task) ---

QUIC360_DIR = f"{ROOT}/refer360/data/quic360_format"


def list_grounding(split: str = "train") -> List[Tuple[str, str, str]]:
    """Return (image_path, query, annotation) region-grounding pairs from quic360.

    quic360 (refer360-derived) is the only grounding data with images on disk:
    3194 ERP panos, 34 region categories (query, e.g. street/people/cars), each
    with a free-text description (annotation). This is image-level category
    grounding -- refer360's coord-localized SDR targets are unusable (source
    images need a separate download). `split` in {train, valid, test}. Rows whose
    image is missing on disk are skipped.
    """
    rows: List[Tuple[str, str, str]] = []
    with open(f"{QUIC360_DIR}/{split}.csv", newline="") as f:
        for r in csv.DictReader(f):
            url = r["url"]
            if os.path.isabs(url) and os.path.exists(url):
                rows.append((url, r["query"], r["annotation"]))
    return rows
