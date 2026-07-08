# Raising Consistency AND Feature Richness Together — literature synthesis

**Purpose:** companion to `EROSION_AND_DISTORTION_LIT.md` and `SSL_SUCCESS_CASES_LIT.md`. This doc
targets ONE specific pain: our cross-view consistency objectives (**M1** code-agreement, **F-3**
masked-view) buy invariance/agreement but *pay for it* by ERODING feature richness — prototype
**purity 0.838 → 0.730**, effective **rank (erank) 39 → 25**, dimensional collapse. Only **TC3**
(dominant fixed-teacher distill anchor + geometric correspondence) raised consistency *without*
eroding purity/erank. Here we map the SSL literature on how to get **both** — consistency and
richness — with a frozen DINOv3 ViT-B/16 + ~0.59M LoRA + E2P-overlap pipeline.

## 요약 (한국어 executive summary)

**두 축을 동시에 올리는 원리적 접근은 세 갈래다.** (1) **명시적 richness 항 추가** — coding-rate /
effective-rank 최대화(MCR2 2006.08558, EMP-SSL 2304.03977, Matrix-SSL 2305.17326), covariance
decorrelation(VICReg 2105.04906, Barlow 2103.03230), uniformity/entropy(2005.10242, KoLeo
1806.03198)를 overlap consistency loss에 **병렬로 얹는다**. (2) **EQUIVARIANCE로 프레임 전환** —
E2P warp에 **불변(invariant)** 대신 **등변(equivariant)**을 강제해, distortion/geometry 정보를
버리지 않고 보존한다(2302.11349 SEN, 2302.10283 SIE, 2111.00899 E-SSL, 2211.01244 EquiMod 등). 이것이 우리 실패 원인에
**가장 직접적으로** 대응한다: M1/F-3은 겹침 영역 특징을 *동일하게* 만들어 변환-민감 차원을
붕괴시켰다. (3) **관계적(relational) 정렬** — 절대 특징이 아니라 patch 간 similarity 구조를
맞춘다(Gram anchoring 2508.10104, PaKA 2509.05606, NeCo 2408.11054). **하나의 교차 통찰:**
consistency loss 자체는 richness를 절대 스스로 회복시키지 못한다 — 반드시 *명시적 richness 항*
(rank/coding-rate max, decorrelation, uniformity, equivariance, 또는 reconstruction)을 별도로
얹거나, 정렬 대상을 "동일성"에서 "예측가능한 등변/관계 구조"로 바꿔야 한다. TC3가 이미 이
원리의 부분 사례(rich frozen anchor로 direction collapse 방지)이며, DINOv3의 Gram anchoring이
그것의 출판된 정식화다. **주의:** 아래 대부분은 우리 axis(erank/purity)를 직접 측정하지 않았다 —
"이식 가능한 가설"이지 "증명된 해결책"이 아니다. 반드시 자체 erank/purity/RankMe로 재측정할 것.

---

## The 2-axis mental model

```
   RICHNESS (rank / decorrelation / uniformity / informativeness / retained nuisance info)
   ▲
   │  [MCR2, EMP-SSL, Matrix-SSL,          [ ← the target quadrant:
   │   VICReg(L), Barlow, W-MSE,             consistency AND richness ]
   │   CorInfoMax, coding-rate max]          TC3 (ours) lives near here.
   │  richness-only regularizers             Equivariant SSL + Gram/CKA
   │  (need a host consistency loss)         relational alignment aim here.
   │
   │─────────────────────────────────────────────────────────►  CONSISTENCY
   │                                                              (alignment / invariance)
   │  [frozen DINOv3 raw:                   [ ← M1 / F-3 landed HERE:
   │   high erank 39, purity 0.838,           high agreement, ERODED
   │   but NO cross-view agreement]           purity 0.730 / erank 25 ]
   ▼
```

- **RICHNESS axis** = effective rank (no dimensional collapse), channel decorrelation, uniformity/
  entropy, semantic purity, and *retention of nuisance/transformation info* (the equivariance lever).
- **CONSISTENCY axis** = cross-view alignment/invariance (our E2P overlap loss, M1 code-CE, F-3).
- **The trap:** a pure-invariance loss with only an anti-collapse *floor* (VICReg variance, BYOL EMA,
  Sinkhorn balancing) prevents *total* collapse but does not *enrich* — it moves you right (more
  consistency) and *down* (less richness). To move **up-and-right** you need an *active* richness
  maximizer (coding-rate/rank), OR a *relational/equivariant* reframing that never demands feature
  identity in the first place.

### 보존형(preserve) vs 개선형(actively improve) — the two levels the user asked for

The user's ask is explicit: *"풍부함을 높인다"* means not only **preventing erosion** (preserve) but the
stronger goal of features becoming **richer than the frozen DINOv3 starting point** (actively improve).
The literature splits cleanly along exactly this floor-vs-maximizer line:

| | **보존형 — floor (prevents collapse only)** | **개선형 — active maximizer (pushes richness UP past frozen)** |
|---|---|---|
| Representatives | VICReg var/cov floor, Barlow/W-MSE decorrelation, BYOL EMA, Sinkhorn balancing, uniformity/KoLeo | **coding-rate / rank max** (MCR2 2006.08558, EMP-SSL 2304.03977, Matrix-SSL 2305.17326, CorInfoMax 2209.07999); **LDReg local-dim max** (2401.10474); **equivariance** (adds transform info — E-SSL 2111.00899, EquiMod 2211.01244, SIE 2302.10283); **reconstruction/MIM** (adds signal) |
| Effect on our erank(39→25) | floors the drop (arrests it) | can lift erank *above* the frozen start |
| Effect on purity(0.838→0.730) | mostly semantics-agnostic → **no purity guarantee** | only **equivariance / reconstruction** touch purity; rank-max alone is purity-neutral |

**Takeaway:** if you want richness that *gets better* (not just non-eroded), look at the **개선형** column —
coding-rate/rank-max, equivariance, MIM. But to improve richness **and** keep purity, an active maximizer must
be **paired with an anchor (TC3 / Gram) or an equivariant reframing** — those are the only levers here that
raise richness without sacrificing semantic purity.

**Decisive filter for "shows both together":** does the paper *measure* a richness metric (erank,
RankMe, coding rate, decorrelation, uniformity, purity) rising *jointly* with a consistency/invariance
metric? Downstream accuracy/mIoU is NOT such evidence (rank can erode while a few useful directions
survive). Almost every paper below fails this filter on *our* axes — so treat them as **mechanisms to
graft and re-measure**, not proofs.

---

## Tiered reading list

Format per entry: **[Tier N] Title** (authors, venue/year, arXiv) — consistency mech · richness mech ·
**richness metric reported** · shows-both? · dense/local? · **graft note** (frozen DINOv3+LoRA+E2P).

Tiering: **Tier 1** = *measured* consistency+richness rising together in a graftable form (none fully
qualify on our axes — see note). **Tier 2** = right mechanism, richness claimed/structural but not
measured on our axes, or wrong regime. **Tier 3** = diagnostic metrics only (no training mechanism).
**Tier 0** (dropped, see count below) = pure-invariance + collapse-floor, no richness term.

> **Tier-1 note:** Of 56 verified papers, **none** cleanly clears Tier 1 *for our regime* — the four
> with `breaks_tradeoff=true` (Matrix-SSL 2305.17326, MMCR 2303.03307, CE-SSL (NeurIPS 2024), and
> structurally VICReg-family) all show the joint rise only on **from-scratch, global** embeddings with
> **non-erank/non-purity** proxies. They are listed as **Tier 2 (strongest)** below, flagged ★.

### A. Coding-rate / rank-maximization (the explicit richness objective) — see also §"Coding-rate" below

- **[Tier 2 ★] Matrix Information Theory for SSL (Matrix-SSL)** (Zhang et al., ICML 2024, arXiv:2305.17326)
  — matrix-KL covariance *alignment* (consistency) + matrix-entropy/log-det (richness) · **richness
  metric: effective rank of feature covariance (shown rising during training)** · shows-both: **yes
  (global, from-scratch)** · dense: no · graft: matrix-KL align + log-det are second-moment loss terms
  on tokens (we already compute this family in `losses.vicreg_vc`); addable onto frozen+LoRA. *Caveat:*
  our own `losses.py:71-74` documents VICReg-family cov terms *eroded* on spatially-autocorrelated pano
  tiles (the cov estimate is gamed) — measure before trusting.
- **[Tier 2] MCR2 — Maximal Coding Rate Reduction** (Yu et al., NeurIPS 2020, arXiv:2006.08558) — weak
  indirect align (self-labeled augs) · **ΔR = R(Z) − R(Z|Π)**: log-det volume *expansion* actively
  maximizes covariance rank · **richness metric: coding rate (log-det volume), per-class singular
  spectra** · shows-both: yes (class-discrimination, not cross-view) · dense: no · graft: borrow the
  **R(Z) total-coding-rate expansion term only** (no per-patch "class" for R(Z|Π)); this is the
  acknowledged ancestor of TC3's anti-collapse term — *the mechanism already worked here*.
- **[Tier 2] EMP-SSL** (Tong, Chen, Ma, LeCun, 2023, arXiv:2304.03977) — cross-patch invariance to
  patch-mean · **Total Coding Rate (TCR) log-det** actively maximizes volume/rank · **richness metric:
  TCR is the objective (no separate erank/RankMe outcome reported)** · shows-both: no · dense: no ·
  graft: TCR is a drop-in log-det term over dense patch embeddings; O(d²), needs enough samples/tuning.
- **[Tier 2] CorInfoMax** (Ozsoy et al., NeurIPS 2022, arXiv:2209.07999) — Euclidean alignment ·
  **log-det of feature covariance** (2nd-order entropy max, anti dimensional-collapse) · **richness
  metric: log-det covariance (eigenspectrum plot, global)** · shows-both: no · dense: no · graft: add
  log-det-cov over dense tokens; protects rank/decorrelation, **not purity** (semantics-agnostic).
- **[Tier 2] FroSSL** (Skean, Dhurandhar et al., ECCV 2024, arXiv:2310.02903) — MSE invariance ·
  **Frobenius-norm of normalized covariance** → eigenspectrum whitening (eigendecomp-free) · **richness
  metric: covariance eigenspectrum flatness** · shows-both: yes (spectrum-flatness, global) · dense: no
  · graft: cheap per-batch cov term on LoRA tokens; risk: whitening DINOv3's *structured* spectrum can
  lower purity while raising rank.

### B. Decorrelation / variance-floor family (VICReg lineage)

- **[Tier 2] VICReg** (Bardes, Ponce, LeCun, ICLR 2022, arXiv:2105.04906) — MSE invariance · variance
  hinge (floor) + covariance decorrelation · **richness metric: off-diag covariance (no erank/purity)**
  · shows-both: no · dense: no · graft: HIGH — cov+var on dense patch tokens alongside overlap loss;
  *floor not maximizer* — blunts erank erosion, does **not** guarantee purity recovery. (Already partly
  in-repo; documented to erode on pano tiles.)
- **[Tier 2] VICRegL** (Bardes, Ponce, LeCun, NeurIPS 2022, arXiv:2210.01571) — n/a (richness half) ·
  **local (per-patch) variance+covariance** · **richness metric: none (per-term seg-mIoU ablation:
  var +0.8, cov +1.3)** · shows-both: no · **dense: YES** · graft: HIGH — the *dense* VICReg exemplar;
  covariance decorrelation on LoRA patch tokens is the most directly relevant piece.
- **[Tier 2] Barlow Twins** (Zbontar et al., ICML 2021, arXiv:2103.03230) — on-diagonal cross-corr → 1
  (invariance) · off-diagonal → 0 (decorrelation) · **richness metric: covariance/decorrelation** ·
  shows-both: no (structural, one loss) · dense: no · graft: off-diagonal term drops onto L2-normalized
  dense tokens; on-diagonal maps to E2P correspondence pairs (needs re-derivation).
- **[Tier 2] W-MSE — Whitening for SSL** (Ermolov et al., ICML 2021, arXiv:2007.06346) — whitened-view
  MSE alignment · **hard batch-whitening → identity covariance** (stronger than soft cov penalty) ·
  **richness metric: none (structural)** · shows-both: no · dense: no · graft: whitening layer in the
  SSL projector; dense tokens give many samples (good stats) but it whitens *projector* space, not the
  backbone features we eval purity on.
- **[Tier 2] VCReg** (Zhu, Ge, LeCun et al., 2023, arXiv:2306.13292) — none of its own (portable add-on)
  · variance floor + covariance decorrelation at intermediate layers · **richness metric: neural-collapse
  CDNV 0.28→0.56, NCC 0.99→0.81, MINE MI 2.8→4.6 bits (no erank/purity)** · shows-both: no (base loss =
  supervised CE) · dense: no · graft: clean — pure feature-statistics terms on dense tokens.
- **[Tier 2] Radial-VCReg** (Kuang et al., NeurIPS 2025 Wksp, arXiv:2602.14272) — VICReg invariance ·
  **radial Gaussianization** (Chi-distribution match on feature norms → suppresses *higher-order*
  dependencies beyond covariance) · **richness metric: none (chi-fit correlates w/ acc)** · shows-both:
  no · dense: no · graft: re-derive radial marginal over patch tokens; weaker leverage through LoRA
  delta on an already-fixed frozen spectrum.

### C. Uniformity / entropy spread

- **[Tier 2] Alignment & Uniformity on the Hypersphere** (Wang & Isola, ICML 2020, arXiv:2005.10242) —
  **L_align** (consistency) + **L_uniform** (RBF-potential max-entropy spread) · **richness metric:
  uniformity** · shows-both: yes (for *its own* uniformity metric) · dense: no · graft: L_align maps to
  overlap loss, L_uniform is a differentiable penalty; **but uniformity spreads same-semantic patches
  apart → plausibly HURTS purity.** The conceptual anchor for "name two axes, jointly optimize."
- **[Tier 2] KoLeo / Spreading vectors** (Sablayrolles et al., ICLR 2019, arXiv:1806.03198) — locality
  triplet · **Kozachenko-Leonenko differential-entropy** spread on the sphere · **richness metric:
  uniformity** · shows-both: no · dense: no · graft: HIGH — KoLeo is the ancestor of DINOv2/v3's own
  anti-collapse term, *architecturally native*; a cheap richness-side regularizer beside the anchor.
  *Uniformity ≠ full rank* (uniform on a low-dim subsphere is possible); does nothing for purity.
- **[Tier 2] MVEB — Multi-View Entropy Bottleneck** (Wen et al., TPAMI 2024, arXiv:2403.19078) —
  agreement max · differential-entropy (vMF-kernel) spread + **compression of non-shared info** ·
  **richness metric: uniformity** · shows-both: no · dense: no · graft: entropy term addable; **its
  "discard superfluous info" compression is antagonistic to our retain-nuisance-info goal** — risky.
- **[Tier 2] Rethinking the Uniformity Metric** (Fang et al., ICLR 2024, arXiv:2403.00642) — base
  alignment (unchanged) · **Wasserstein-to-Gaussian uniformity** (provably senses dimensional collapse
  + redundancy where Wang-Isola cannot) · **richness metric: their Wasserstein uniformity** · shows-both:
  no · dense: no · graft: better-shaped anti-collapse term than raw uniformity; still a floor, no purity.

### D. Equivariant SSL — **see dedicated §"Equivariant SSL" below** (most on-point)

- **[Tier 2 ★] Contrastive-Equivariant SSL (CE-SSL)** (Yerxa/Feather/Simoncelli/Chung, NeurIPS 2024;
  **verified 2026-07-07: 2409.06710 = 'McGrids' iso-surface paper, NOT this; CE-SSL has no confirmed arXiv id —
  cite via NeurIPS 2024 / PMC12058038**) — invariant base loss (Barlow/SimCLR/MMCR) + equivariant term
  on transformation-matched *difference vectors* · **richness metric: Bures (transformation vs content
  factorization) — NOT erank/purity** · shows-both: yes (Bures + IT-neural-predictivity) · dense: no ·
  graft: re-derive as E2P-overlap variant (warp → consistent difference *direction*, not agreement).
  **The closest published "SSL is overly invariant → use equivariance" cure.**
- **[Tier 2] SIE — Split Invariant-Equivariant** (Garrido, Najman, LeCun, ICML 2023, arXiv:2302.10283) —
  invariance on invariant sub-vector · equivariant sub-vector predicted by hypernet from the *known*
  transform · **richness metric: none (equivariance MRR/H@k)** · shows-both: no · dense: no · graft:
  split LoRA output; invariant half ← E2P overlap loss, equivariant half ← predictor conditioned on
  E2P homography/relative pose (natural fit — we already have exact geometric correspondence).
- **[Tier 2] SEN — Steerable Equivariant** (Bhardwaj et al., 2023, arXiv:2302.11349) — learn linear
  operator T_g so f(aug_g x) ≈ T_g f(x) (align *modulo* transform) · retains transform info structurally
  · **richness metric: none** · shows-both: no · dense: no · graft: replace "f(view1)=f(view2)" with
  "f(view2)=T_g f(view1)"; T_g learned per E2P geometric relation.
- **[Tier 2] E-SSL — Equivariant Contrastive Learning** (Dangovski et al., ICLR 2022, arXiv:2111.00899)
  — InfoNCE invariance · **auxiliary head predicts which transform was applied** (retain nuisance info)
  · **richness metric: none** · shows-both: no · dense: no · graft: attach a head predicting the E2P
  warp / relative pose between overlapping tiles; cheap, backbone frozen.
- **[Tier 2] STL — Self-supervised Transformation Learning** (Shin et al., NeurIPS 2024, arXiv:2501.08712)
  — contrastive invariance + image-invariant *learned* transform embedding · features change
  equivariantly · **richness metric: none** · shows-both: no · dense: no · graft: our warp is
  *known-geometric* (not learned) → simpler than STL; predictor maps view-A→view-B under E2P warp.
- **[Tier 2] EquiMod** (Devillers & Lefort, ICLR 2023, arXiv:2211.01244) — base invariance · predictor
  predicts augmentation-induced *displacement* in embedding space · **richness metric: none (qualitative
  retained-color)** · shows-both: no · dense: no · graft: condition predictor on E2P warp so overlap loss
  becomes "predict-the-displacement" not "make-equal." Most mechanistically-aligned candidate.
- **[Tier 2] AugSelf** (Lee et al., NeurIPS 2021, arXiv:2111.09613) — base invariance · MLP predicts the
  *difference* of augmentation params between views · **richness metric: none** · shows-both: no · dense:
  no · graft: head predicts relative E2P transform (rotation/crop/scale offset) between tiles.
- **[Tier 2] Understanding Equivariance in SSL** (Wang et al., NeurIPS 2024, arXiv:2411.06508) — theory:
  equivariance "explaining-away" synergy with class-feature extraction · **richness metric: none** ·
  shows-both: no · dense: no · graft: theoretical justification for a predict-the-E2P-warp branch.

### E. Relational / structure alignment (align similarity structure, not features)

- **[Tier 2] DINOv3 Gram Anchoring** (Siméoni, Vo, Seitzer et al., 2025, arXiv:2508.10104) — Frobenius
  match of patch-wise **Gram matrices** to an earlier EMA "Gram teacher" (structure stays, values free
  to move) · anti-collapse floor preserving spatial diversity · **richness metric: none (dense mIoU/RMSE
  gains: VOC 50.3→55.7, NYU 0.307→0.281)** · shows-both: no · **dense: YES** · graft: HIGH — pure
  auxiliary loss on L2-normed patch Gram; build per-view or cross-view overlap Gram, match to synced
  LoRA checkpoint. **This is essentially the published formalization of TC3** (relational anchor). Our
  `losses.py` already has a `gram_anchor` impl.
- **[Tier 2] PaKA — Patch-Level Kernel Alignment** (Yeo et al., 2025, arXiv:2509.05606) — maximize
  **CKA** between two views' patch kernels (relational, isotropic-scale invariant) · avoids constant-
  feature collapse · **richness metric: none** · shows-both: no · **dense: YES** · graft: HIGH — drop-in
  overlap-loss replacement (L = 1 − CKA on centered patch Grams); non-parametric. *Caveat:* CKA is only
  isotropic-scale invariant → anisotropic per-channel shrinkage still allowed (floor, not maximizer).
- **[Tier 2] NeCo — "Near, far: Patch-ordering enhances vision foundation models' scene understanding"**
  (Pariza et al., ICLR 2025, arXiv:2408.11054) — patch
  nearest-neighbor **ordering** consistency (differentiable sort) vs fixed teacher · soft ranking retains
  neighborhood geometry · **richness metric: none (ADE20k +5.5, VOC +6)** · shows-both: no · **dense:
  YES** · **backbone: frozen+adapter (structurally identical to us!)** · graft: HIGH — soft ordering
  loss on E2P overlap patches vs frozen-teacher reference; TC3-shaped signal, drops onto frozen+LoRA.
- **[Tier 2] CrOC — Cross-View Online Clustering** (Stegmüller et al., CVPR 2023, arXiv:2303.13245) —
  consistency over jointly-discovered co-visible clusters (not all locations) · avoids over-invariance ·
  **richness metric: none** · shows-both: no · **dense: YES** · graft: import the clustering
  correspondence-selection as a smarter overlap mask; its own self-distillation is M1-family (pair with
  TC3 anchor + explicit richness term).

### F. Dense masked-modeling precedents (richness via dense supervision, not a rank term)

- **[Tier 2] iBOT** (Zhou et al., ICLR 2022, arXiv:2111.07832) — DINO CLS self-distill · masked-patch
  self-distill (dense supervision) · **richness metric: none** · shows-both: no · **dense: YES** ·
  graft: **this IS what F-3 instantiates** — and F-3 landed on the eroded side. Not a fix.
- **[Tier 2] SiameseIM** (Tao et al., CVPR 2023, arXiv:2206.01204) — cross-view semantic align · dense
  patch prediction from masked view (spatial sensitivity) · **richness metric: none** · shows-both: no ·
  **dense: YES** · graft: dense-target-of-aligned-view template; BYOL-floor only, needs explicit rank term.
- **[Tier 2] LDReg — Local Dimensionality Regularization** (Huang et al., ICLR 2024, arXiv:2401.10474) —
  none of its own (host loss supplies it) · **actively maximizes local intrinsic dimensionality (LID)**
  addressing *local* underfilling global metrics miss · **richness metric: effective rank + mean LID
  (mLID) — BOTH lifted for all baselines** · shows-both: no (no consistency measured jointly) · **dense:
  YES** · **backbone: continued** · graft: HIGH — gradients into LoRA; kNN LID over patch tokens.
  **The most direct principled counterweight to erank(39→25) in the batch** — but LID↑ ≠ purity↑.

### G. Diagnostics (Tier 3 — metrics, not mechanisms)

- **[Tier 3] RankMe** (Garrido et al., ICML 2023, arXiv:2210.02885) — **smooth effective rank** = entropy
  of L2-normed singular-value distribution; label-free, correlates with linear-probe acc. Graft the
  metric (few lines of SVD) on dense patch features per recipe/LoRA config; unsupervised early-stop /
  collapse detector. **Adopt as our standing richness monitor.**
- **[Tier 3] LiDAR** (Thilak et al., ICML 2024, arXiv:2312.04000) — LDA-rank down-weights nuisance
  directions that plain erank still counts → drops sharply exactly when consistency erodes *class-relevant*
  richness. Adapt: LDA scatter over patch tokens, E2P overlap correspondences as the within-class set.
- **[Tier 3] α-ReQ** (Agrawal et al., NeurIPS 2022, OpenReview ii9X4vtZGTZ) — power-law eigenspectrum
  decay exponent α (≈1 best); compute-cheap collapse monitor; softer graded target than a hard erank floor.
- **[Tier 3] DSE — Dense Structure Estimator** (Dai et al., NeurIPS 2025, arXiv:2510.17299) — names the
  exact phenomenon **SDD (Self-supervised Dense Degradation)**; two-factor indicator (class-relevance +
  effective dimensionality) usable as annotation-free per-checkpoint monitor AND additive regularizer
  (L = L_ssl − β·DSE) on dense patch features. **Ready-made joint diagnostic for M1/F-3 erosion.** No
  cross-view term of its own → richness insurance, not a tradeoff-breaker.

*(Also-diagnostic, cited for completeness: WERank arXiv:2402.09586 — weight-orthogonality floor, but our
backbone is frozen so it can only touch LoRA A/B factors, which are deliberately low-rank → conceptually
strange, unvalidated.)*

**Dropped (Tier 0 — pure invariance + collapse-floor, no richness term, would reproduce M1/F-3 erosion):**
IConE (2603.15263), data2vec (2202.03555) & data2vec 2.0 (2212.07525), SdAE (2208.00449), MaskDistill
(2210.10615), RC-MAE (2210.02077), MSN (2204.07141), Leopart (2204.13101), SCRL (2103.06122), PixPro
(2011.10043), Time-Tuning (2308.11796), CMAE (2207.13532), DCL (2110.06848), Img2Vec (2304.12535),
"What Should Be Equivariant" (CVPR22 Wksp, no arXiv). **~14 dropped or demoted** — each is a distillation/
invariance recipe whose only anti-collapse is a floor (EMA/Sinkhorn/normalization), mechanistically the
same family that eroded our purity/erank; several (iBOT/F-3, data2vec/F-3, MSN/M1, Leopart/M1) are
literally what we already ran on the eroded side.

---

## Equivariant SSL — the most on-point family (dedicated)

**Why this is *the* answer to our specific disease.** M1/F-3 force cross-view features in the E2P
overlap to be **identical** (invariance). But the two overlapping perspective tiles differ by a *known
geometric warp* (distortion, relative pose). Demanding identity therefore **quotients out** the
distortion/viewpoint subspace — precisely the "retention of useful nuisance/transformation info" that
our richness definition prizes. This collapse of transform-sensitive directions is a first-principles
explanation for purity 0.838→0.730 and erank 39→25.

**The equivariant reframing:** instead of `f(view_A) = f(view_B)`, enforce `f(view_B) = T_g · f(view_A)`
(or "predict g from the pair"), where **g is our E2P homography / relative pose** — which we *already
compute exactly*. Consistency now lives in the *predictable structured map*, not in feature identity, so
each view keeps its own rich, high-rank basis. This is the same instinct that made **TC3** work (keep a
rich anchor, don't force raw invariance), and 2411.06508 gives the theory (equivariance "explains away"
nuisance, *helping* class-feature extraction).

**Concrete grafts (all backbone-frozen, LoRA-only), cheapest first:**
1. **Predict-the-warp head** (E-SSL 2111.00899 / AugSelf 2111.09613): auxiliary head reads LoRA patch
   features of two overlapping tiles, predicts the relative E2P transform. Cheapest counter-pressure.
2. **Displacement predictor** (EquiMod 2211.01244 / SEN 2302.11349): predictor maps view-A features to
   view-B *under* g; overlap loss = displacement-prediction error, not agreement.
3. **Invariant/equivariant split** (SIE 2302.10283): split LoRA output — invariant half ← overlap loss,
   equivariant half ← g-conditioned predictor. Protects part of the space from the invariance pull.
4. **Difference-vector equivariance** (CE-SSL, NeurIPS 2024): enforce that the E2P warp maps to a
   *consistent difference direction* across overlap pairs.

**Honest caveat:** *none* of these papers measured erank/purity — the richness benefit is conceptual
(retain transform info) or proxied by downstream accuracy. Every graft above is a **hypothesis to
measure**, not a proven fix. But mechanistically it is the family most likely to break our tradeoff.

---

## Coding-rate / rank-max (the explicit richness objective) + RankMe

**Why this family matters:** it is the only one that *actively maximizes* a rank/volume quantity rather
than merely flooring collapse. Our erank drop (39→25) *is* a degeneration of exactly the log-det volume
these methods maximize. Critically, **MCR2 (2006.08558) is the acknowledged ancestor of TC3's own
anti-collapse coding-rate term** — so this mechanism has *already transferred and worked* in this repo.

- **Total Coding Rate / log-det volume** (MCR2, EMP-SSL 2304.03977, CorInfoMax 2209.07999, Matrix-SSL
  2305.17326): add `R(Z) = ½ log det(I + (d/nε²) ZZᵀ)` over dense patch embeddings alongside the overlap
  loss. Gradients into LoRA only. This directly counteracts dimensional collapse.
- **The purity caveat (repeated because it matters):** coding-rate/decorrelation protect **rank**, not
  **purity**. Purity is cluster-homogeneity/informativeness, a *different* quantity — spreading features
  to fill space can even work *against* tight semantic clusters. So a coding-rate term very plausibly
  restores erank while leaving (or lowering) purity. **This must be co-measured, not assumed.** TC3's
  fixed-teacher *distill* anchor is the only thing we have that empirically preserved *purity* — the
  coding-rate term is a **complement** to that anchor, not a replacement.
- **RankMe (2210.02885) as the standing diagnostic:** adopt it now. `RankMe = exp(entropy of L2-normed
  singular values)` on the dense patch-feature matrix, per recipe/LoRA config, as a **label-free
  early-stop / config picker** that flags dense dimensional collapse *before* downstream eval. Pair with
  **LiDAR (2312.04000)** (nuisance-down-weighted rank) and **purity** to arbitrate whether a recipe
  preserves *useful* richness. This is exactly the instrumentation M1/F-3 lacked.

---

## Mapping to our tradeoff

| Failure signature | Mechanism family that targets it | Would it have prevented M1/F-3 erosion? |
|---|---|---|
| **erank 39→25** (dimensional collapse) | coding-rate/log-det max (MCR2, EMP-SSL, Matrix-SSL, CorInfoMax); LDReg local-dim max; VICReg/Barlow/VCReg decorrelation | **Likely blunts it** (active rank max) or floors it (decorrelation). Coding-rate is the strongest — it *is* the ancestor of TC3's working anti-collapse term. |
| **purity 0.838→0.730** (semantic homogeneity) | **fixed-teacher distill anchor (TC3, DINOv3 Gram anchoring); equivariant preservation** | **Only the anchor/equivariance route addresses purity.** Rank/decorrelation/uniformity terms are semantics-agnostic → may protect rank while leaving purity flat or lower. |
| **agreement satisfiable by collapse** (M1 code-CE) | relational alignment (Gram/CKA/NeCo ordering) instead of hard code identity; SwAV/MSN Sinkhorn/ME-MAX (see erosion doc) | **Relational alignment most directly:** it never demands feature identity, so it can't be satisfied by direction-collapse. |
| **TC3 = only non-eroding recipe** | it is a *rich frozen anchor* + geometric correspondence = a dominant richness-preserving reference + relational structure | Confirms the cross-cutting insight: consistency-without-erosion needs either an explicit richness term OR a rich anchor OR an equivariant/relational reframing — TC3 supplies the anchor version. |

**Single most-plausible prevention of M1/F-3 erosion (ranked):**
1. **Reframe the E2P overlap loss as EQUIVARIANT** (predict-the-warp / displacement, §Equivariant) —
   attacks the *root cause* (invariance discards distortion info). Best expected payoff, needs building.
2. **Relational alignment (Gram anchoring 2508.10104 / CKA-PaKA 2509.05606 / NeCo 2408.11054)** — swap
   hard code-agreement for similarity-structure agreement. Highest graftability (NeCo is *literally*
   frozen+adapter), and Gram anchoring is the published TC3 formalization; already stubbed in `losses.py`.
3. **Add an explicit coding-rate / LDReg richness term + RankMe/purity monitor** — guards erank, but
   pair with the TC3 anchor to protect purity.

---

## What to try (honest) — ranked by graftability-onto-frozen × expected payoff

1. **[Highest graft, high payoff] Swap M1 hard code-agreement → relational overlap loss.** Adopt
   **NeCo-style ordering consistency (2408.11054)** or **CKA/PaKA (2509.05606)** or **Gram anchoring
   (2508.10104)** on E2P overlap patches vs the frozen teacher. Pure auxiliary loss, LoRA-only, dense by
   construction. NeCo's regime (frozen+adapter, ~19h/1-GPU) is *identical to ours*. **Elevate the existing
   `losses.py:gram_anchor` and A/B it.** *Measure erank+purity+overlap-cosine jointly.*
2. **[High graft, high-but-untested payoff] Add an equivariance head to the E2P consistency loss.**
   Predict the relative E2P warp/pose between overlapping tiles (E-SSL 2111.00899 / AugSelf 2111.09613),
   or the g-conditioned displacement (EquiMod 2211.01244). Attacks the root cause. Cheap head, backbone
   frozen. *This is the highest-upside experiment; treat as hypothesis, gate on purity.*
3. **[High graft, partial payoff] Add an explicit coding-rate/rank term alongside the overlap loss.**
   TCR/log-det (2304.03977, 2006.08558) or **LDReg local-dim max (2401.10474)** on LoRA dense tokens.
   Actively maximizes rank → should arrest erank 39→25. **Complement, not replacement, for the TC3
   anchor** (purity needs the anchor). Watch for metric-gaming (injecting high-freq variance).
4. **[High graft, insurance] Instrument now: RankMe (2210.02885) + LiDAR (2312.04000) + DSE (2510.17299)
   + purity.** Zero training change; a few lines of SVD/LDA on dense patch features per checkpoint.
   Label-free collapse detector + config picker M1/F-3 lacked. Do this first — it de-risks everything else.
5. **[Medium graft, likely partial] VICReg-family decorrelation guardrail** (2105.04906 / 2210.01571 /
   2306.13292) on E2P overlap tokens. Cheap. **But our own `losses.py:71-74` documents this eroded on
   spatially-autocorrelated pano tiles** (cov statistic is gamed) — use with a decorrelation-friendly
   pooling and re-measure; do not over-weight against the overlap term.
6. **[Likely MIRAGES — flag]** (a) **Uniformity/entropy spread** (2005.10242, 1806.03198, 2403.19078):
   spreads *same-semantic* patches apart → plausibly *lowers* purity; KoLeo is native but weak (uniform
   on a low-dim subsphere still passes). (b) **MVEB compression (2403.19078)** actively *discards*
   nuisance info — antagonistic to our retain-transform-info goal. (c) **iBOT/MIM (2111.07832)** = what
   F-3 already is, on the eroded side. (d) **WERank (2402.09586)** orthogonalizes weights — but our
   trainable weights are deliberately low-rank LoRA; conceptually unsound here.

---

## Footer — verified arXiv ids

**Coding-rate / rank-max:** 2006.08558 (MCR2) · 2304.03977 (EMP-SSL) · 2209.07999 (CorInfoMax) ·
2305.17326 (Matrix-SSL) · 2310.02903 (FroSSL)
**Decorrelation / variance floor:** 2105.04906 (VICReg) · 2210.01571 (VICRegL) · 2310.00527 (CLoVE) ·
2103.03230 (Barlow Twins) · 2007.06346 (W-MSE) · 2306.13292 (VCReg) · 2602.14272 (Radial-VCReg)
**Uniformity / entropy:** 2005.10242 (Align&Uniform) · 1806.03198 (KoLeo/Spreading) · 2403.19078 (MVEB) ·
2403.00642 (Wasserstein Uniformity)
**Equivariant SSL:** 2302.10283 (SIE) · 2501.08712 (STL) · 2111.00899 (E-SSL) · 2211.01244 (EquiMod) ·
2111.09613 (AugSelf) · 2306.13924 (CARE) · 2302.11349 (SEN) · 2306.06082 (CASSLE) · 2411.06508 (Equiv.
theory) · CE-SSL (Yerxa et al., NeurIPS 2024; **2409.06710 = 'McGrids' iso-surface paper, NOT CE-SSL; no
confirmed CE-SSL arXiv id**) · "What Should
Be Equivariant" (CVPR 2022 L3D-IVU Wksp, no arXiv)
**Relational / dense:** 2508.10104 (DINOv3 Gram) · 2509.05606 (PaKA) · 2408.11054 (NeCo) · 2303.13245
(CrOC) · 2204.13101 (Leopart) · 2303.03307 (MMCR)
**Dense masked-modeling:** 2111.07832 (iBOT) · 2206.01204 (SiameseIM) · 2401.10474 (LDReg) · 2207.13532
(CMAE) · 2210.10615 (MaskDistill) · 2304.12535 (Img2Vec) · 2202.03555 (data2vec) · 2212.07525 (data2vec
2.0) · 2208.00449 (SdAE) · 2210.02077 (RC-MAE)
**Diagnostics (Tier 3):** 2210.02885 (RankMe) · 2312.04000 (LiDAR) · 2510.17299 (DSE) · 2402.09586
(WERank) · α-ReQ (NeurIPS 2022, OpenReview ii9X4vtZGTZ, no confirmed arXiv)
**Other invariance/floor (Tier 0):** 2603.15263 (IConE) · 2204.07141 (MSN) · 2103.06122 (SCRL) ·
2011.10043 (PixPro) · 2308.11796 (Time-Tuning) · 2110.06848 (DCL) · 2510.23484 (T-REGS)

*Verification note: all rows drawn from a 56-paper tiered evidence set (fan-out search → fetch → adversarial
refutation). No paper cleanly clears Tier 1 on OUR axes (erank/purity, frozen+LoRA, dense); the 4 with
structural `breaks_tradeoff=true` (2305.17326, 2303.03307, CE-SSL, VICReg-family) are marked ★ Tier 2 and
show joint rise only on from-scratch/global proxies. Two id cautions (primary-source verified 2026-07-07):
**2409.06710** is *McGrids* (iso-surface extraction), NOT CE-SSL — CE-SSL (Yerxa et al., NeurIPS 2024) has no
confirmed arXiv id; **α-ReQ** has no confirmed standalone arXiv id.*
