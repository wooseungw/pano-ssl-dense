# Initial Design Notes

## Core Question

Can planar SSL vision encoders become strong panorama dense predictors if the input pipeline supplies explicit
spherical geometry through global ERP context, E2P tiles, and cross-view consistency?

## Minimal Baseline

- Encoder: frozen DINOv2 or similar SSL ViT.
- Input: global ERP plus overlapping perspective tiles.
- Fusion: tile feature reprojection into ERP space.
- Head: lightweight dense decoder.
- Losses: supervised downstream loss when labels exist, plus overlap and seam consistency.

## Risks

- Tile-level predictions may disagree at seams.
- Horizon-only pitch coverage can miss floor and ceiling regions.
- Depth and pointmap outputs need scale alignment across tiles.
- Frozen planar encoders may underperform near poles or on heavily distorted ERP context.

## Validation

- Unit-test yaw and pitch metadata against rendered tiles.
- Verify overlap masks by projecting adjacent tiles back to ERP coordinates.
- Compare tile-only, global-only, and global-plus-tile fusion.
- Report seam-region metrics separately from full-image metrics.

