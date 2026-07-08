# PANO-iBOT — lightly imprinting panorama structure onto DINOv3 (masked latent prediction)

**Goal (user):** beat frozen DINOv3 (seg fold-1 = **57.7**) by a LIGHT continued-pretrain on 21.8k
panos that adds panorama structure DINOv3's planar prior lacks — via **patch-level, spatial masked
prediction** (not consistency: consistency adds no info; masked prediction does).

Design is grounded in a 5-area literature survey (2022–2026). Key sources per claim below.

## 0. Why the prior attempt (F-3) failed — and what the papers say to do instead
| F-3 Pano-JEPA (eroded) | Modern best practice (survey) |
|---|---|
| masks **whole tiles** (2–6 of 24) | mask **patch tokens**, block-wise (iBOT, I-JEPA, SimMIM) |
| ~8–25% ratio (too easy) | **high, block-wise** ratio; latent-distill tolerates 0.3–0.5 (iBOT/DINOv2), MAE 0.6–0.75 |
| target = bare EMA, **dormant VICReg** | latent target + **centering/Sinkhorn + sharpening (τ_t≈0.04–0.07)** + KoLeo (DINOv2/v3) |
| BYOL-style anti-collapse only → collapse | **DINOv3 Gram anchoring** = principled anti-degradation + prior anchor |
| no use of overlap geometry | **cross-view masked completion** via E2P warp (unique signal) |

## 1. Lucky fit — our E2P tiles ARE the recommended masking unit
The 360-SSL survey's #1 rule: *don't mask a uniform ERP grid* (polar patches are tiny, over-sampled,
trivially in-painted). Mask on **tangent/cubemap patches** (near-equal-area, reuse planar priors —
Eder *Tangent Images* 2020). **Our E2P tiles are exactly tangent patches.** So we mask the 32×32
patch tokens *within tiles*, which is both the iBOT unit and the distortion-correct spherical unit.

## 2. The objective — PANO-iBOT (two masked-latent terms + Gram anchor)

- **Student** S = DINOv3 + LoRA(r=16) with the **last 1–2 blocks unfrozen** (ExPLoRA 2024: the most
  on-point low-data recipe — continue the native SSL objective under PEFT, ~5–10% of weights; full
  fine-tune catastrophically forgets at this scale).
- **Teacher** T = EMA(S), sees the FULL unmasked tiles, **stop-grad**, momentum 0.996→1.0.
- **Frozen anchor** F = original DINOv3 (never moves).

**Term A — within-tile iBOT MIM** (continue DINOv3's native dense objective → imprint pano statistics):
```
mask block-wise ~40–50% of each tile's 32×32 patch tokens (student sees visible only, mask tokens)
L_A = Σ_{masked patch} H( sharpen∘center(T_patch) ,  softmax(S_patch / τ_s) )      # iBOT patch self-distill, LATENT target
```
Latent (feature) target, NOT pixels — feature-target MIM is markedly more sample-efficient and gives
stronger dense-seg features (iBOT, BEiTv2 56.7 ADE20K, low-data survey).

**Term B — cross-view masked completion** (E2P-unique, the differentiator DINOv3 never saw):
```
mask a contiguous block in tile A whose rays are VISIBLE in overlapping tile B (different obliquity)
predictor P (narrow shallow transformer, conditioned on relative obliquity/pose) predicts the masked
A-patch tokens from B's visible tokens at the warp locations (+ A's visible context)
L_B = Σ_{masked-A patch} ( 1 − cos( P(context_B, geo) , sg[ T_A(masked patch) ] ) )
```
**Non-triviality (the JEPA de-overlap rule — this is what F-3/consistency got wrong):** the target is
**A's own view-specific teacher feature** (not B's), and the masked region is hidden in A, so P must
learn the **cross-view distortion transform**, not copy B. High mask ratio keeps it hard. → teaches
oblique→canonical structure (helps exactly the distorted edge regions where seg fails).

**Anti-collapse + anchor (what F-3 lacked):**
```
L_Gram = || Ŝ Ŝᵀ − F̂ F̂ᵀ ||²        # DINOv3 Gram anchoring: dense-feature Gram matrix pinned to the frozen prior
```
DINOv3's own fix for dense-feature degradation over long training — and it doubles as our **erosion
guard** (the anchor-strength thesis, now principled). Plus **KoLeo** on CLS (uniform hypersphere,
kills dimensional collapse) and teacher **centering/Sinkhorn + sharpening**.

**Panorama weighting:** sample masks and weight the loss by spherical area / obliquity (∝ cos lat) so
oblique/equatorial content drives learning and redundant poles don't dominate (distortion-aware loss).

**Total:**  `L = L_A + λ_B·L_B + λ_gram·L_Gram + λ_koleo·L_KoLeo`

## 3. Why this can beat frozen 57.7 (honest)
- Term A continues DINOv3's proven dense objective on pano → better *pano* dense features (imprint).
- Term B injects cross-view distortion/geometry structure the planar prior lacks → improves the
  oblique/edge regions where the frozen features are weakest.
- Gram anchoring + centering/sharpening give the anti-erosion the whole prior program lacked.

## 4. Honest risk & most likely failure mode
- **Light imprint ≠ big jump.** At 21.8k the gain may be frozen+ε, not a decisive win. Beating 57.7 by
  a clear, multi-seed margin is NOT guaranteed.
- **Most likely failure:** Term B degenerates to trivial copy → collapses back to consistency (≈frozen)
  if de-overlap/high-ratio isn't strict. Mitigation: strict de-overlap, A-view target, high ratio, verify
  B-context is genuinely different-obliquity.
- **Second:** anchor tension — L_Gram too weak → erosion; too strong → can't move past frozen. Needs a sweep.

## 5. Minimal validation (same seg harness)
1. Implement **Term A only** (continue-iBOT), light pretrain 21.8k (~few h), eval on seg fold-1 vs frozen 57.7
   + **linear-probe seg early** (survey: catch weak dense structure before full fine-tune).
2. If A ≥ frozen, add **Term B**; re-eval, especially **oblique-region mIoU** (where B should help most).
3. Sweep λ_gram (anchor strength). Multi-seed the winner.

## References (survey, verified-web)
MAE (He 2022) · SimMIM (Xie 2022) · iBOT (Zhou 2022) · BEiTv2 (Peng 2022) · AttMask/HPM/SemMAE (guided
masking) · I-JEPA (Assran 2023) · V-JEPA / V-JEPA2 (2024/25) · DINO (Caron 2021) · DINOv2 (Oquab 2023) ·
**DINOv3 Gram anchoring** (Siméoni 2025) · KoLeo (2019) · **ExPLoRA** (2024, PEFT continued-SSL) ·
Tangent Images (Eder 2020) · distortion-aware 360 SSL · ERP-RoPE / Dense360 (2025).
