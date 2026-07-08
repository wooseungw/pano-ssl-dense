# Pano Where⊕What (PWW) — unified self-supervised spec

**What this is.** A single masked-view E2P pretext that fuses two ideas the user converged on:
**"Where"** (ERP-roll multi-granularity position prediction with randomized FOV) ⊕ **"What"**
(cross-view masked completion, PANO_MIM Term B), sharing one frozen-DINOv3 + LoRA encoder, with an
**optional Kalman/inverse-variance consensus target**. Companion: `docs/SSL_ACCURACY_DESIGN.md`
(the gate + why this is genuinely SSL), `docs/PANO_MIM_DESIGN.md` (Term A/B), `docs/CONSISTENCY_AND_RICHNESS_LIT.md`.

## 한국어 요약

한 forward에서 타일마다 **(What)** 오버랩 타일 B로 A의 마스킹 블록 **특징을 완성**(왜곡 변환 학습) +
**(Where)** 모든 패치의 **구면 위치를 멀티-그래뉼러 예측**(타일+패치, well-posed 위도 + 타일중심 대비
각오프셋 + FOV), **랜덤 roll + 랜덤 FOV** 하에. 두 헤드가 LoRA 인코더를 공유 → 특징이 **콘텐츠-풍부
(완성)이면서 기하-인식(위치)** 이 됨. **철칙:** 절대 경도(longitude)는 gauge라 예측 금지 — 위도는 절대,
경도는 상대만. 옵션으로 **다중타일 칼만 융합 consensus**를 완성 타깃으로(correctness prior + 불확실성).
distill/gram anchor는 지배적으로 유지(침식 방지). 정직한 기대: **depth/normal에서 실이득**, seg는 frozen+ε.

---

## 0. Why unify (What ⊕ Where)

- Both operate on the *same* E2P render geometry and the *same* LoRA encoder → one forward, shared cost.
- **Complementary information:** completion injects *content + cross-view distortion transform*; position
  injects *spherical placement + projection (FOV) geometry* the planar prior lacks. Content-rich **and**
  geometry-aware in one representation.
- **Mutual reinforcement:** the completion predictor is conditioned on relative A↔B geometry; the position
  head is *learning that very geometry*. Co-training pushes the encoder toward a geometry-aware latent that
  serves both readouts.
- roll + FOV randomization augment both terms simultaneously.

## 1. Data / labels (per step) — all free from known render params

Panoramas are gravity-aligned by capture → ERP row = latitude (exact). Per step:
1. Random **ERP roll** δ (lossless horizontal shift; free gauge aug).
2. Sample overlapping E2P tiles on the DOMAINS pitch rings, each tile `t` with center `(yaw φ_t, pitch θ_t)`
   and a **random FOV `f_t`** from a bank/range. `geometry.py` gives the **exact per-pixel ray** → exact
   gravity-frame `(latitude, longitude)` and intra-tile angular offset for every patch. Labels are free.
3. Pick an overlapping pair `(A,B)` where a contiguous block of A's patches is visible in B at different
   obliquity/FOV (for the completion term).

## 2. Head 1 — WHERE (multi-granularity position), well-posed

> **Well-posedness rule (do not violate):** the panorama has no canonical longitude origin (roll is a free
> gauge) → **absolute longitude is ill-posed, never a target.** Predict absolute *latitude*, intra-tile
> *angular offset*, *FOV*, and *relative* azimuth only.

**Patch-level (dense), per patch token i from student features S(i):**
- **Absolute latitude** `β_i ∈ [−90°,90°]` (gravity-referenced). Direct/Huber regression (no wrap). Teaches
  vertical spherical structure + latitude-dependent distortion.
- **Intra-tile angular offset** `(Δaz_i, Δel_i)` of patch i from the tile center, **in angles**. FOV-dependent
  → with random FOV the model must infer the pinhole projection. Huber regression.

**Tile-level, from pooled/CLS token:**
- **Tile-center latitude** `θ_t` (Huber). · **Tile FOV** `f_t` (Huber) — predicting FOV from one tile is a
  genuine geometric-inference signal (perspective shrink rate).

**Cross-tile (the only well-posed longitude signal):**
- **Relative azimuth** `Δφ = φ_A − φ_B` between the overlapping pair (known from render). Folds in the
  "predict-the-warp" equivariant flavor.

```
L_pos = Σ_i cosλ(β_i)·[ λ_lat·Huber(β̂_i,β_i) + λ_off·Huber(off̂_i,off_i) ]   # dense, cos-lat weighted
      + λ_tlat·Huber(θ̂_t,θ_t) + λ_fov·Huber(f̂_t,f_t)                        # tile
      + λ_rel·Huber(Δφ̂,Δφ)                                                    # cross-tile relative
```
Weight dense terms by `cos(lat)` (spherical area) so poles don't dominate.

## 3. Head 2 — WHAT (cross-view masked completion), PANO_MIM Term B

Mask a contiguous block of A's patches (rays visible in B). A shallow predictor `P` (conditioned on relative
A↔B geometry) predicts the masked A-patch **teacher features** from B's visible tokens at warp locations +
A's visible context.

```
L_comp = Σ_{masked A patch} [ 1 − cos( P(context_B, geo_AB), sg[ T_A(masked patch) ] ) ]
```
**De-overlap rule (what F-3 got wrong):** target = **A's own view-specific teacher feature**, region hidden
in A, high mask ratio → `P` must learn the **cross-view distortion transform**, not copy B. (Optionally add
within-tile iBOT MIM `L_A` for extra dense imprint — already in `train_pano_ibot.py`.)

## 4. Optional — Kalman/inverse-variance consensus target (correctness prior + uncertainty)

Replace the single teacher target in `L_comp` with a **multi-tile Bayesian fusion** of the masked ray's
feature across all covering tiles, each measurement's covariance `Σ_k ∝ obliquity/FOV` (from Head-1 geometry):
```
μ_fused = (Σ_k Σ_k^{-1})^{-1} Σ_k Σ_k^{-1} f_k ,   with per-ray uncertainty U = (Σ_k Σ_k^{-1})^{-1}
L_comp' = Σ  U^{-1} · [ 1 − cos(P(·), sg[μ_fused]) ]        # inverse-variance-weighted
```
This is the Kalman/measurement-fusion view of E2P overlap: each tile = a noisy measurement of a ray; optimal
fusion = inverse-variance update. It formalizes the ensemble +0.088 (variance reduction) and yields a
**denoised target + calibrated uncertainty** (usable as a downstream confidence map).
**Honest caveat (iron law 2 / M2):** the fused target's accuracy is *variance reduction* → it makes a cleaner
training signal, not a guaranteed single-view lift. Ship as an **ablation** (single-teacher vs Kalman-consensus
target), not as the default. (Directly-Kalman SSL exists in state-space/tracking: unsupervised KalmanNet
2110.09005, self-sup differentiable KF for VIO 2203.07207, Kalman-VAE NeurIPS 2017, KalCo — none is image-rep
SSL; the fit here is fusion/uncertainty, and becomes direct if sequences/inter-pano trajectories are added.)

## 5. Total loss (anchor stays dominant — anti-erosion)

```
L = L_comp(or L_comp')                 # WHAT — cross-view completion (info-adding, correctness prior)
  + λ_pos · L_pos                      # WHERE — position + FOV (geometry-adding)
  [+ λ_A · L_A]                        # optional within-tile iBOT MIM
  + λ_distill · distill(S, sg F)       # DOMINANT anchor (TC3 lesson; iron law 4)
  + λ_gram · gram_anchor(S, F)         # dense erosion guard
  + λ_koleo · koleo(cls)               # anti dimensional-collapse / richness floor
```

## 6. Shortcut guards (from the adversarial stress framing)

The killer risk: model cheats position/FOV from **low-level artifacts** (ERP→pinhole interpolation blur that
grows toward poles, borders, resample signatures) → would NOT transfer to a decoder. Defenses:
- Predict position from **LoRA features**, and complete at **feature level** (not raw pixels).
- **Consistent resampling** across FOV; add mild **blur/scale jitter** to break the artifact↔FOV correlation.
- **No black borders** (tiles fully inside the sphere) or mask border patches.
- Longitude is **relative-only** → no absolute-gauge leak.
- **Ablate the latitude-via-pole-stretch cue:** verify latitude prediction survives when stretch cues are
  normalized (else the signal is a shortcut, not geometry).

## 7. Architecture / params (all new heads + LoRA trained; backbone frozen)

- Position head: 2-layer MLP on patch tokens (dense) + small MLP on pooled token (tile). ~few×100K.
- Completion predictor `P`: shallow (2–4 layer) narrow transformer, conditioned on relative geo. ~1–2M.
- Encoder: frozen DINOv3 + LoRA(q/v) (~0.59M); EMA teacher `T`; frozen anchor `F`.

## 8. Reuse map

- `data.py`: add per-step random ERP roll + per-tile random FOV (bank {e.g. 50,60,70,80} or continuous).
- `geometry.py`: exact ray/coordmap → position labels (lat / offset / FOV / relative-az) + warp for completion.
- `encoder.py`: two new heads.
- `losses.py`: `L_pos` (Huber terms); reuse `distill_loss`/`gram_anchor`/`koleo`; optional Kalman-fusion util.
- Training: **extend `scripts/train_pano_ibot.py`** (already has Term A + gram + koleo + EMA).

## 9. Cheapest diagnostic (frozen, before training) — Q1 pre-filter

On **frozen** DINOv3 features, fit linear probes: (a) patch **latitude**, (b) tile **FOV**.
- Frozen already predicts them well → info present → little headroom (decoder may launder) → position term
  adds little. · Frozen **cannot** → headroom exists → the pretext has something to teach.
- Measures presence-of-information (NOT the M2 trap). Minutes on one GPU (`diag_semantic_headroom.py` pattern).

## 10. First experiment + success/kill

Light unified pretrain (~few h, S3D) → eval under **real UPerNet(seg)/DPT(depth,normal) decoder, multi-seed
≥3**, vs frozen **57.7** (never a linear probe — §3.9 laundering).
- **Greenlight**: clears frozen by a multi-seed margin on any head (expected first on **depth/normal**).
- **Report oblique/edge-region metrics separately** — where position+completion should help most.
- **Kill**: ≤ frozen+noise on all heads multi-seed → injected geometry launders → rethink.

## 11. Honest EV

The best-positioned genuinely-SSL design so far: the only one injecting **both** new content-geometry
(completion) **and** new spherical-projection structure (position), single-view-accessible, at feature level,
with a dominant anti-erosion anchor. **Most-likely outcome:** real multi-seed win on **depth/normal**
(distortion/geometry-heavy), **frozen+ε on seg** (semantics stay teacher-bound). Biggest residual risk: the
FOV/latitude signal being partly a low-level shortcut → guarded (§6) and ablated (§9). The Kalman-consensus
target is an upside ablation, not load-bearing.
