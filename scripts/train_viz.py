"""Default train-time qualitative viz (opt out with TRAIN_VIZ=0).

After an SSL trainer saves runs/<slug>/, emit the SEG (3 datasets) + DEPTH + NORMAL
stitch-integration figures for the just-trained adapter — the same layout as runs/..._stitch_demo:
input ERP -> single-tile ERP footprints -> overlap count -> STITCHED-as-scored (metric annotated)
-> GT, rendered at the METRIC eval size, 3 spread val samples each. Small CAPPED probe heads
(VIZ_CAP_TR / VIZ_CAP_VA / VIZ_EPOCHS) keep it a picture, not a benchmark. Reuses the bench
machinery so the stitch IS the real metric path:
  seg    -> viz_seg_integration.emit           (Stanford2D3D + DensePASS + Structured3D)
  depth  -> depth_s2d3d_bench.emit_integration (Stanford2D3D)
  normal -> normal_s2d3d_bench.emit_integration (Stanford2D3D)

Wired into train_ssl / train_pano_ibot / train_ssl_f3 / train_ssl_vicreg. Standalone:
  CUDA_VISIBLE_DEVICES=0 python scripts/train_viz.py <adapter_dir|''>
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_common as BC  # noqa: E402
import data  # noqa: E402
import depth_s2d3d_bench as DB  # noqa: E402
import normal_s2d3d_bench as NB  # noqa: E402
import runlog  # noqa: E402
import seg_s2d3d_bench as SB  # noqa: E402  (viz_seg_integration uses this module's ADAPTER/TAG)
import viz_seg_integration as VS  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

CAP_TR = int(os.environ.get("VIZ_CAP_TR", 100))          # capped probe-train panos (a picture, not a bench)
CAP_VA = int(os.environ.get("VIZ_CAP_VA", 40))
VIZ_EPOCHS = int(os.environ.get("VIZ_EPOCHS", 10))


def _build_encoder(adapter: str) -> PanoEncoder:
    """Frozen DINOv3 + the just-trained adapter (or plain frozen when adapter is empty)."""
    enc = (PanoEncoder(model_id=BC.MODEL, adapter_path=adapter) if adapter
           else PanoEncoder(model_id=BC.MODEL, lora_rank=0))
    return enc.to(BC.DEVICE).eval()


def _emit_dense(run: str, enc: PanoEncoder, mod, out_ch: int) -> None:
    """Capped probe head for a dense task (depth/normal) + its integration figures (S2D3D)."""
    plan = BC.build_plan()
    cids = [BC.coord_map(tp.yaw_deg, tp.pitch_deg) for tp in plan]
    tr, va = BC.split_files(data.list_erps("stanford2d3d"), BC.FOLD)
    ctr = mod.build_cache(enc, tr, plan, want_full=False)
    cva = mod.build_cache(enc, va, plan, want_full=True)
    head = BC.DenseHead(enc.dim, out_ch).to(BC.DEVICE)
    mod.train_head(head, ctr)
    head.eval()
    mod.emit_integration(run, head, cva, cids, plan, va)


def emit_train_viz(run: str, adapter: str = "") -> None:
    """Emit seg(3 datasets)+depth+normal integration figures for `adapter` into run/viz/."""
    if os.environ.get("TRAIN_VIZ", "1") == "0":
        print("train-viz skipped (TRAIN_VIZ=0)", flush=True)
        return
    tag = os.path.basename(adapter.rstrip("/")) if adapter else "frozen"
    BC.TR_PANOS, BC.VA_PANOS, BC.EPOCHS = CAP_TR, CAP_VA, VIZ_EPOCHS      # cap the probe train (bench_common)
    BC.ADAPTER, BC.TAG = adapter, tag                                    # honest labels on depth/normal figs
    SB.ADAPTER, SB.TAG = adapter, tag                                    # ... and on the seg figs (VS uses SB)
    VS.CAP_TR, VS.CAP_VA, VS.EPOCHS = CAP_TR, CAP_VA, VIZ_EPOCHS         # cap the seg probe train
    print(f"train-viz: seg(3 ds)+depth+normal integration for '{tag}' -> {run}/viz "
          f"(cap_tr={CAP_TR} cap_va={CAP_VA} ep={VIZ_EPOCHS})", flush=True)
    enc = _build_encoder(adapter)
    for name, fn in (("seg", lambda: VS.emit(run, enc)),
                     ("depth", lambda: _emit_dense(run, enc, DB, 1)),
                     ("normal", lambda: _emit_dense(run, enc, NB, 3))):
        try:
            fn()
        except Exception as e:                                          # a task failing must not crash the trainer
            print(f"train-viz {name} FAILED: {type(e).__name__}: {e}", flush=True)


def main() -> None:
    adapter = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ENC_ADAPTER", "")
    run = runlog.create_run("train_viz", {"adapter": adapter or "frozen",
                                           "cap_tr": CAP_TR, "cap_va": CAP_VA, "epochs": VIZ_EPOCHS,
                                           "tasks": ["seg(3 datasets)", "depth", "normal"]})
    emit_train_viz(run, adapter)
    print(f"saved -> {run} (config + seg/depth/normal integration viz)", flush=True)


if __name__ == "__main__":
    main()
