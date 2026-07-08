"""Diagnose position-pretext headroom (diagnose-before-train — project iron law #1).

Q: does FROZEN DINOv3 ALREADY linearly encode a tile's PITCH and FOV?
  - pitch highly decodable  => "predict pitch" is READOUT of existing info (Q1 fail): near-zero
    new gradient into the encoder => that head is predicted-null.
  - FOV large residual      => FOV is the GENUINE new signal the position pretext can inject.

Linear (ridge) probe on FROZEN tile-pooled + patch features -> {pitch, fov}; report MAE vs a
predict-the-mean baseline (decoded_frac = 1 - MAE/baseline: ~1 = already present, ~0 = headroom).

Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/diag_position_headroom.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import train_ssl as TS  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = "cuda"
MODEL = TS.MODEL
N_PANOS = int(os.environ.get("N_PANOS", 40))
PITCHES = tuple(float(x) for x in os.environ.get("PITCHES", "-45,0,45").split(","))
FOVS = tuple(float(x) for x in os.environ.get("FOVS", "45,55,65,75,85").split(","))
N_YAW = int(os.environ.get("N_YAW", 4))


def ridge_probe(X: np.ndarray, y: np.ndarray, lam: float = 10.0, frac: float = 0.7,
                groups: np.ndarray = None):
    """Standardized ridge; MAE on a held-out split vs predict-train-mean baseline.

    groups (per-row pano id): if given, the held-out split is PANO-DISJOINT (train panos and
    test panos share no tiles) — rules out per-pano memorization / leakage.
    """
    n = len(X)
    if groups is not None:
        ug = np.unique(groups)
        rng = np.random.RandomState(0).permutation(len(ug))
        te_g = set(ug[rng[int(frac * len(ug)):]].tolist())
        te = np.array([i for i in range(n) if groups[i] in te_g])
        tr = np.array([i for i in range(n) if groups[i] not in te_g])
    else:
        idx = np.random.RandomState(0).permutation(n)
        ntr = int(frac * n)
        tr, te = idx[:ntr], idx[ntr:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    ymu = y[tr].mean()
    A = Xs[tr].T @ Xs[tr] + lam * np.eye(Xs.shape[1])
    w = np.linalg.solve(A, Xs[tr].T @ (y[tr] - ymu))
    pred = Xs[te] @ w + ymu
    mae = float(np.abs(pred - y[te]).mean())
    base = float(np.abs(y[te] - ymu).mean())
    return mae, base


def main():
    enc = PanoEncoder(model_id=MODEL, lora_rank=0).to(DEVICE).eval()   # FROZEN, no LoRA
    files = [f for f in data.list_erps("stanford2d3d")
             if "5" not in f.split("extracted_data/")[1].split("/")[0]][:N_PANOS]
    src = "stanford2d3d(train)"
    if not files:
        files = data.list_structured3d(limit=N_PANOS)
        src = "structured3d"
    print(f"diag position headroom: src={src} panos={len(files)} pitches={PITCHES} fovs={FOVS} n_yaw={N_YAW}", flush=True)
    if not files:
        print("NO DATA FOUND — check PANO_DATA_ROOT / extraction.", flush=True)
        return

    yaws = np.linspace(-180.0, 180.0, N_YAW, endpoint=False)
    pooled_feats, pitch_lab, fov_lab, pano_lab = [], [], [], []
    pano_split = os.environ.get("PANO_SPLIT", "0") == "1"
    for fi, f in enumerate(files):
        try:
            erp = TS.load_erp(f, "in")
        except Exception:
            continue
        for fov in FOVS:
            specs = [(float(y), float(p)) for p in PITCHES for y in yaws]
            tiles = TS.render_tiles(erp, specs, fov).to(DEVICE)
            with torch.no_grad():
                fmap = enc(normalize_tiles(tiles))                      # (T,D,gh,gw)
                pooled = fmap.mean(dim=(2, 3)).float().cpu().numpy()    # (T,D) tile-pooled
            pooled_feats.append(pooled)
            for (y, p) in specs:
                pitch_lab.append(p); fov_lab.append(fov); pano_lab.append(fi)
        if (fi + 1) % 10 == 0:
            print(f"  {fi+1}/{len(files)} panos", flush=True)

    X = np.concatenate(pooled_feats, 0)
    pitch = np.asarray(pitch_lab, np.float32)
    fov = np.asarray(fov_lab, np.float32)
    groups = np.asarray(pano_lab) if pano_split else None
    print(f"\ntiles={len(X)} D={X.shape[1]} split={'PANO-DISJOINT' if pano_split else 'tile-random'}\n", flush=True)
    print(f"{'target':12s} {'probe MAE':>10} {'baseline':>10} {'decoded_frac':>13}   interpretation")
    for name, y in [("pitch(deg)", pitch), ("fov(deg)", fov)]:
        mae, base = ridge_probe(X, y, groups=groups)
        frac = 1.0 - mae / base if base > 0 else 0.0
        interp = ("ALREADY PRESENT (readout, low headroom)" if frac > 0.6
                  else "HEADROOM (frozen weak here)" if frac < 0.35 else "partial")
        print(f"{name:12s} {mae:10.2f} {base:10.2f} {frac:12.1%}   {interp}", flush=True)
    print("\nDecision: pitch high-frac => predict-pitch is readout (Q1-null). "
          "fov low-frac => FOV is the genuine lever (reweight the pretext toward FOV + cross-view).", flush=True)


if __name__ == "__main__":
    main()
