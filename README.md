# Pano SSL Dense

Self-supervised panorama representation learning for dense downstream tasks.

This repository is intended to explore 360-degree image understanding with SSL-pretrained vision encoders and
AnyRes-E2P-style panorama inputs. It is separate from PanoLLaVA/CORA so that dense segmentation, depth, and pointmap
experiments can evolve without coupling to the VLM fine-tuning code path.

## Research Goal

Learn or adapt a vision encoder for dense panorama understanding:

- semantic or panoptic segmentation
- monocular depth estimation
- DUSt3R-style pointmap prediction
- cross-view feature consistency over equirectangular panoramas

## Input Assumption

The default panorama input should follow an AnyRes-E2P layout:

1. low-resolution global ERP image for scene context
2. overlapping perspective tiles rendered from the ERP
3. yaw, pitch, FOV, and camera-center metadata per tile
4. inverse projection from tile-space outputs back to ERP or spherical coordinates

This matches the practical direction of using strong planar vision encoders while adding geometry-aware panorama
fusion instead of training a panorama-only encoder from scratch.

## First Milestone

Build a minimal frozen-encoder baseline:

1. Render global ERP plus E2P tiles.
2. Extract dense features with a frozen SSL encoder such as DINOv2.
3. Reproject tile features into an ERP feature canvas.
4. Train a small dense decoder for one target task.
5. Add overlap and seam consistency losses.

## Current Structure

```text
pano-ssl-dense/
|-- anyres_e2p.py           # Standalone AnyRes-E2P renderer copied from PanoLLaVA/CORA
|-- configs/                # YAML experiment configs
|-- docs/                   # Design notes and paper summaries
|-- scripts/                # Thin CLI entry points
`-- tests/                  # Unit tests for geometry and tensor contracts
```

## Reference Anchors

- DINOv2: SSL vision features with strong pixel-level transfer.
- MAE and iBOT: masked image/token modeling for scalable ViT pretraining.
- I-JEPA and V-JEPA: latent feature prediction instead of direct pixel reconstruction.
- DenseCL and VICRegL: dense/local SSL objectives.
- DASC-SPT: self-supervised panoramic semantic segmentation.
- Depth Anything V2 and PanDA: foundation depth adaptation to panorama depth.
- DUSt3R: pointmap-based dense 3D reconstruction.
