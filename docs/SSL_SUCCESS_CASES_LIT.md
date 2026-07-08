# SSL Success Cases — when self-supervised / continued pretraining actually RAISED downstream accuracy

**Purpose:** a companion to `docs/EROSION_AND_DISTORTION_LIT.md` and `docs/CONSISTENCY_AND_RICHNESS_LIT.md`
(the latter covers raising cross-view consistency *and* feature-richness together). This doc is a *reading list
of positive cases* — papers where SSL / continued pretraining measurably improved downstream **accuracy** (not
just consistency, not just efficiency). It is organized so the analytically load-bearing papers — the ones that
speak to *our* exact wall (**does continued/self-supervised pretraining beat a STRONG FROZEN foundation model
in a LOW-DATA, DENSE, panorama regime?**) — are unmissable.

**Provenance.** 67 papers were scouted and adversarially tiered against the decisive filter below. This doc
keeps **all Tier-1 (3), the strongest Tier-2, and the most useful Tier-3**, capped at ~22 entries. Papers
that only beat random-init / ImageNet-supervised / other-SSL baselines are *deliberately discounted* — that
class is well-known and says nothing about our ceiling.

> **The decisive filter (why most "SSL wins" don't count for us).** We do NOT care that SSL beats
> random-init or ImageNet-supervised baselines. We care ONLY: does continued/self-supervised pretraining beat
> a **strong FROZEN foundation model** (DINOv2/v3/CLIP/SAM class) in a **low-data, dense-prediction** regime?
> Our hard-won invariant: frozen DINOv3 + E2P + fusion + decoder is already SOTA-competitive (Stanford2D3D seg
> **57.7 mIoU**) at ~0.59M trainable params, and our cross-view *consistency* SSL has NEVER beaten it on
> accuracy. "consistency != accuracy" is the confirmed wall.

---

## 한국어 요약 (executive summary)

- **질문에 대한 직답:** SSL로 "강한 frozen foundation model"을 dense task에서 **정확도로** 실제 이긴 논문은
  67편 중 **단 3편(Tier-1)** 뿐이다 — **NeCo (2408.11054)**, **DIP (2506.18463)**, **LoRA3D (2412.07746)**.
  나머지 "SSL 승리"의 대부분은 random-init / ImageNet-sup / 다른 SSL을 이긴 것으로, 우리 벽과 무관하다.
- **하나의 관통 패턴 (WHEN SSL beats a strong frozen foundation on dense):** SSL이 frozen foundation을 정확도로
  이기는 경우는 **(a) foundation이 그 도메인/태스크에서 이미 강하지 않아 headroom이 크거나(=domain-gap을 메우는
  경우: 대부분의 의료·원격탐사·내시경 사례), 또는 (b) SSL이 단순 consistency가 아니라 "그 태스크에 맞는 새 구조
  정보(correctness/geometry/retrieval-alignment)"를 주입할 때** 뿐이다. frozen backbone이 **이미 in-domain으로
  강하고** SSL 신호가 **consistency뿐**이면, 이득은 accuracy가 아니라 consistency/efficiency로만 나타난다 —
  이것이 정확히 우리 null 결과다.
- **우리 베팅(PANO-iBOT/MIM)에의 함의:** MIM은 "consistency가 못 주는 정보를 준다"는 가설이라 방향은 맞지만,
  **frozen+low-data+dense에서 frozen foundation을 이긴 MIM 사례는 이 67편 안에 하나도 없다.** cross-view completion
  계열(CroCo→DUSt3R→MASt3R)조차 전부 **from-scratch/full-finetune·대규모 데이터**에서 이긴 것이라 우리 regime을
  증명하지 못한다. Tier-1 3편은 모두 **MIM이 아니다** (neighbor-consistency / in-context retrieval / geometric
  self-calibration). 즉 우리 벽을 넘는 검증된 경로는 아직 **없고**, MIM은 여전히 "미검증 유망 가설"이다.

---

## AXIS — When SSL actually RAISED accuracy (tiered reading list)

Each entry: **SSL method · task (dense?) · baseline it beat (bold — the decisive fact) · backbone use ·
data scale · reported gain · mechanism (the lever) · transfer-to-us**.

> **One id note (RESOLVED, primary-source verified 2026-07-07).** `2406.10973` is **ExPLoRA** (Khanna et al.,
> verified). **MASt3R** is a *different* paper at **`2406.09756`** (Leroy/Cabon/Revaud, ECCV 2024). An earlier
> draft conflated the two — MASt3R ids below are corrected to `2406.09756`.

### Tier 1 — genuinely beat a STRONG FROZEN foundation model on dense accuracy (all 3 kept)

These three are the only papers in the set that clear the decisive filter. **None uses MIM** — read them for
the *existence proof* that a frozen-foundation dense-accuracy ceiling can move, and for the *mechanism* that
made it move (which is the real lesson for us). Ordered by relevance to our regime.

**[Tier 1] DIP: Unsupervised Dense In-Context Post-training of Visual Representations**
(Sirko-Galouchenko et al., ICCV 2025, arXiv:2506.18463)
- SSL method: unsupervised in-context post-training via meta-learning-style pseudo-tasks (nearest-neighbor in-context matching), auto-generated from a diffusion model + the encoder.
- Task: in-context dense seg (6 datasets incl. ADE20K, PascalVOC) + monocular depth (NYUv2). **Dense: yes.**
- **Baseline it beat: DINOv2R (registers), a strong frozen foundation model — plus prior post-training (incl. NeCo, +0.5 avg).**
- Backbone use: **continued-pretrain** (base DINOv2R; also tested CLIP, MAE). *NB:* fine-tunes the **last 3 ViT blocks + a new MLP head** — not a frozen backbone + tiny LoRA.
- Data scale: small/efficient — **~8.5h on one A100** (COCO, 5 epochs). Low-compute continued-pretrain.
- Reported gain (from the paper; abstract-level fetch confirmed the method + "outperforms initial encoder and prior methods" but not these exact figures — treat as extraction-sourced, re-check in the PDF): **ADE20K 40.8→42.6, VOC 79.0→82.1 mIoU; NYUv2 RMSE 0.771→0.756; +5.2 mIoU on PascalVOC at 1/128 labeled data.**
- Mechanism (lever): in-context pseudo-tasks reshape dense features to be retrieval/matching-aligned; transfers across tasks (seg-trained pseudo-tasks help depth).
- **Transfer-to-us:** the **strongest** Tier-1 for us on the axes that match — unsupervised continued post-training that *adds accuracy over a frozen foundation on dense seg AND depth*, with an explicit **low-label** win (mirrors our low-label downstream). But: (1) mechanism is **retrieval-matching pseudo-tasks, not MIM** — an analogy to "MIM adds info", not the same lever; (2) wins measured under **in-context NN-retrieval eval**, not a trained decoder — +1.8–2.1 mIoU feature-reshaping gains can wash out under a high-capacity head (our exact setup); (3) DIP **unfreezes 3 blocks**, so gains may come from that extra capacity, not the SSL objective — a like-for-like frozen+LoRA repro may see less. **Must verify any win under our own trained-decoder protocol.**

**[Tier 1] NeCo (method) — "Near, far: Patch-ordering enhances vision foundation models' scene understanding"**
(Pariza, Salehi, Burghouts, Locatello, Asano, ICLR 2025, arXiv:2408.11054 — *title/authors primary-source verified; earlier drafts used the working title "…Patch Neighbor Consistency"*)
- SSL method: **patch-neighbor-consistency** via differentiable sorting (dense self-distillation on top of DINOv2R; not MIM, not contrastive).
- Task: dense semantic segmentation (linear + non-parametric in-context/kNN) + dense correspondence. **Dense: yes.**
- **Baseline it beat: DINOv2-registers (DINOv2R) — the strong frozen foundation it post-trains from.**
- Backbone use: **continued-pretrain** (student updated via student-teacher, bootstrapped from frozen DINOv2R teacher).
- Data scale: ImageNet-1k, **19 GPU-hours on one GPU** — low-COMPUTE, but **NOT low-DATA** (1.28M diverse images).
- Reported gain: **+5.5% ADE20k, +6% Pascal VOC, +7.2% COCO-Things, +5.7% COCO-Stuff mIoU** over DINOv2R.
- Mechanism (lever): differentiable-sort nearest-neighbor ranking of dense patch features sharpens spatial structure the global objective under-optimizes.
- **Transfer-to-us:** closest positive datapoint, but two caveats bite. (1) The mechanism is a **consistency** objective — the very family that produced our wall — it wins only under a *different* consistency target (dense patch-NN ranking, not cross-view E2P agreement). This suggests **our wall may be specific to the cross-view formulation, not consistency-SSL per se.** (2) Gains are measured with **frozen-feature linear/kNN probes**, not a trained decoder; our 57.7 is already **head-driven**, so probe-level +5.5% can shrink or vanish once a strong fusion+decoder compensates. Worth piloting a NeCo-style dense-neighbor-consistency post-pretrain on frozen DINOv3 pano tiles — but it is not proof.

**[Tier 1] LoRA3D: Low-Rank Self-Calibration of 3D Geometric Foundation Models**
(Lu et al., ICLR 2025, arXiv:2412.07746)
- SSL method: self-supervised **self-calibration** — confidence-reweighted robust multi-view optimization makes pseudo-labels, LoRA distills them back (per-scene test-time). **No MIM.**
- Task: dense 3D reconstruction (pointmaps), multi-view pose, novel-view rendering. **Dense: yes.**
- **Baseline it beat: frozen pretrained DUSt3R/MASt3R — a strong 3D geometric foundation model. Up to 88% improvement.**
- Backbone use: **frozen+adapter** (LoRA ~18MB, frozen DUSt3R/MASt3R).
- Data scale: **per-scene / very low data** — self-calibrates on one scene's own multi-view images in ~5 min/GPU; 160+ scenes (Replica, TUM, Waymo).
- Reported gain: **up to 88%** over frozen DUSt3R on reconstruction, pose, novel-view.
- Mechanism (lever): a robust multi-view geometric optimizer injects a **correctness prior** into pseudo-labels; LoRA distills that consistency-*with-correctness* back into the frozen model.
- **Transfer-to-us:** deceptively on-regime (frozen + low-data + dense + LoRA) but the honest read **confirms our diagnosis**: consistency raises accuracy *only when the pseudo-labels also carry a correctness prior* — which our raw E2P-overlap consistency lacked. There is **no closed-form multi-view optimizer** to manufacture correct pseudo-labels for frozen-DINOv3 semantic seg/depth/normal, so the lever doesn't port. It is per-scene **test-time** overfit, not a reusable checkpoint. **Its real value: the strongest argument FOR the MIM bet** — masking forces reconstruction of real signal, which is a correctness prior, unlike pure agreement.

---

### Tier 2 — right mechanism family, wrong regime (kept the most load-bearing)

These do NOT clear the filter (they win under full-finetune, or on weak/in-domain-poor baselines, or on
geometry not semantics), but they carry the **mechanism** our PANO-iBOT bet rests on. Read them as motivation
and design hints, not as proof.

**The MIM foundation-builders (motivate "MIM adds info", prove nothing about frozen ceiling):**

**[Tier 2] Masked Autoencoders Are Scalable Vision Learners (MAE)** (He et al., CVPR 2022, arXiv:2111.06377)
- MIM (75% pixel masking) · ADE20K seg transfer (dense) · **beat supervised-ImageNet init + MoCov3/DINO/BEiT** · from-scratch then full-finetune · ImageNet-1k ~1.28M · ViT-L ~53.6 mIoU ADE20K · reconstructing masked pixels forces holistic features · **transfer:** the seminal "MIM adds info beyond invariance" motivation, but zero evidence at frozen+low-data.

**[Tier 2] iBOT: Image BERT Pre-Training with Online Tokenizer** (Zhou et al., ICLR 2022, arXiv:2111.07832)
- self-distillation MIM w/ online tokenizer (latent target) + DINO [CLS] · ADE20K seg + COCO (dense) · **beat supervised/DeiT, BEiT, MAE, DINO** · full-finetune · ImageNet-1k/22k · ViT-B ~50.0 mIoU ADE20K · latent-target MIM learns local "visual grammar" · **transfer:** *the exact within-tile term (Term A) lineage*; proves latent-MIM yields real dense accuracy — but under full-finetune, not frozen.

**[Tier 2] SimMIM: A Simple Framework for Masked Image Modeling** (Xie et al., CVPR 2022, arXiv:2111.09886)
- raw-pixel-regression MIM · classification-primary (dense transfer only) · **beat prior SSL + supervised, all full-finetuned** · full-finetune · ImageNet-1k · +0.6% top-1 · simple pixel MIM scales rep quality · **transfer:** mechanism prior only; the design says raw-pixel is enough, but our design (`PANO_MIM_DESIGN`) prefers latent targets — see BEiTv2/MaskFeat/data2vec below.

**[Tier 2] Masked Feature Prediction (MaskFeat)** (Wei et al., CVPR 2022, arXiv:2112.09133)
- MIM w/ **HOG feature target** · classification+video (not dense) · **beat MAE/BEiT/SimMIM/MoCov3/DINO + scratch** · full-finetune · ImageNet-1k / K400 · ViT-B 84.0% · a *feature-space* target is more semantic+sample-efficient than pixels · **transfer:** directly supports PANO-iBOT's **latent/Gram feature-target** choice over pixels; regime irrelevant to our ceiling.

**[Tier 2] data2vec: A General Framework for SSL** (Baevski et al., ICML 2022, arXiv:2202.03555)
- masked latent regression to **EMA-teacher contextualized targets** · classification (not dense) · **beat BEiT/MAE/MaskFeat + pixel/HOG ablations** · from-scratch+full-finetune · ImageNet-1k · 84.2% top-1 · contextualized latent targets > local pixel/HOG · **transfer:** underwrites Term A's **latent EMA-teacher target** design; not evidence for our regime.

**The cross-view-completion lineage — our published "Term B" — gets its own section below. Its members
(CroCo 2210.10716, CroCo v2 2211.10408, DUSt3R 2312.14132, MASt3R 2406.09756, plus MuM 2511.17309,
Muskie 2511.18115, ZeroCo 2412.09072) are all Tier 2 and are treated together.**

**Continued-SSL-under-PEFT precedents (right SHELL, mixed evidence):**

**[Tier 2] ExPLoRA: Parameter-Efficient Extended Pre-Training under Domain Shifts** (Khanna et al., 2024, arXiv:2406.10973 — *verified: this id is ExPLoRA's own; MASt3R is the separate 2406.09756*)
- continued DINOv2/MAE SSL under PEFT (unfreeze 1–2 blocks + LoRA, ~6% weights) · classification + ONE dense task (SpaceNet-v1 seg) · **beat frozen DINOv2+LoRA** · continued-pretrain under PEFT · ~360k satellite images · **+7.53 linear-probe top-1** on fMoW, **but SpaceNet-v1 dense seg = 76.69 = 76.69, an exact TIE** · continued SSL injects domain-shift info frozen features lack · **transfer:** *the most on-point PEFT recipe* (it is `PANO_MIM_DESIGN`'s cited basis for "continue native SSL under PEFT"), BUT its **one dense datapoint is a null** — the classification win vanishes on dense, reproducing our wall. Right-mechanism encouragement, adverse dense evidence.

**[Tier 2] DIET-CP: Lightweight and Data-Efficient Self-Supervised Continued Pretraining** (Rodas et al., 2025, arXiv:2509.06990)
- DIET instance-discrimination continued pretraining · **classification only (not dense)** · **beat frozen DINOv3** · **full fine-tune** of DINOv3 (not frozen+LoRA) · **~1000 images/domain (genuinely low-data)** · ~2–5% acc (figure-level, soft) · steers frozen DINOv3 to a new domain without forgetting · **transfer:** existence proof that a **low-data continued SSL can move a DINOv3 accuracy ceiling** — but classification-only (no dense evidence) and it argues **UNFREEZE**, which cuts against our frozen+LoRA constraint (hints the constraint may be *why* our ceiling holds).

**[Tier 2] MedDINOv3: How to adapt vision foundation models for medical segmentation?** (Li et al., 2025, arXiv:2509.02379)
- multi-stage DINOv3 continued pretrain (self-distill + Gram anchoring) · CT/MRI organ+tumor seg (dense) · **beat nnU-Net + DINO U-Net (no isolated frozen-probe row)** · **full-finetune** at 896² · **~3.87M CT slices (large)** · +2.6–5.49% DSC over nnU-Net; **continued-pretrain ablation +1.07% DSC over un-adapted start** · in-domain continued self-distill closes CT gap · **transfer:** the +1.07% ablation is the right *direction* (continued in-domain SSL > FM start on dense) but both arms are full-finetuned, and — critically — **their Gram anchoring gave ~0% here** (the exact piece PANO-iBOT wants to borrow was empirically inert).

**Test-time / weakly-supervised "beats frozen depth FM" (accuracy-real, but not label-free corpus SSL):**

**[Tier 2] Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting** (Bhattarai & Rhodin, CVPR 2026, arXiv:2512.17908)
- test-time SSL via re-lighting + Score-Distillation from a 2D diffusion prior · mono depth (dense) · **beat frozen Depth Anything V2 (all 9 metrics); SOTA on DA3** · frozen backbone, optimize embeddings+decoder only · **per-image, no pretrain corpus** · ~8.5% SIlog/RMSElog↓ (KITTI), ~8.4% AbsRel↓ (ETH3D) · injects NEW info via an **external diffusion prior** · **transfer:** clean existence proof that a "frozen depth FM is a ceiling" wall CAN break — but the new info comes from a **second foundation model** at test time, not from a corpus MIM objective. Portable lesson: **anti-collapse recipe** (update only embeddings+decoder, backbone frozen). Hints our MIM may need an *external strong prior/target* to add info.

**[Tier 2] WeSTAR: Weakly-Supervised Adaptation with Regularization** (Huang et al., AAAI 2026, arXiv:2511.14238)
- dense self-training + **weak ordinal supervision** + LoRA reg · mono depth OOD (dense) · **beat frozen Depth Anything** · frozen+adapter · scale unstated · SOTA OOD (no numeric deltas in abstract) · win driven by **injected weak labels** + self-training · **transfer:** the win needs **labels**, so it does not demonstrate label-free SSL beating frozen. **Directly borrowable: the LoRA-regularization-against-drift trick** to prevent catastrophic forgetting during continued pretrain.

**[Tier 2] EndoDAC: Adapting Foundation Model for Self-Supervised Endoscopic Depth** (Cui et al., MICCAI 2024, arXiv:2405.08672)
- photometric monocular-video SSL + Dynamic Vector-LoRA · endoscopic mono depth (dense) · **beat fully-finetuned Depth Anything (0.051 vs 0.058 AbsRel), zero-shot DA (0.084), AF-SfMLearner (0.059)** · frozen+adapter · ~thousands SCARED frames (low-data) · real accuracy · **but the frozen FM is WEAK in-domain** (0.084 = ~40% worse; huge surgical domain gap) · **transfer:** *the cautionary mirror of our situation*. The win exists because the frozen model has **big domain-shift headroom**; our frozen DINOv3+E2P is **already in-domain-strong** with little headroom. Photometric video SSL is a **consistency** lever — same family as our null. **Predicts NO gain in our regime.**

**[Tier 2] Rapid Adaptation of Earth-Observation Foundation Models for Segmentation** (Marasinghe et al., 2024, arXiv:2409.09907)
- **supervised** LoRA (r=256), NOT self-supervised · flood seg (dense) · **beat frozen EO FM + full-finetune (+6.66 F1, +0.11 IoU)** · frozen+adapter · small flood set (low-data) · supervised task capacity via LoRA · **transfer:** clean *methodological template* for a frozen-FM-vs-adapter dense comparison, but it re-demonstrates what **we already own** (our supervised LoRA ties SGFormer SOTA 0.104). Says nothing about SSL.

**[Tier 2] Diminishing Returns in Self-Supervised Learning** (2025, arXiv:2512.03862)
- MAE-style MIM · semantic seg (dense) · **beat fine-tune-from-random-init (5M ViT), no frozen FM anywhere** · from-scratch + full-finetune · low-data sweep · MIM > FT-only, diminishing with more labels · **transfer:** a **WARNING** — MIM's benefit shrinks as the starting representation gets stronger; a frozen DINOv3 is a very strong prior, so continued MIM has the **least predicted headroom** exactly where we need it. Useful design point: a misaligned intermediate (classification) objective collapses spatial structure — **keep PANO-iBOT dense-aligned** (Term A/B already are).

**[Tier 2] Learning What to Predict: Downstream-Guided Task Design for Continued Pretraining** (Ke & Fanti, 2026, arXiv:2601.22108)
- DINO continued SSL + downstream-guided task designer (learned views/masks) · ADE20K seg + NYUv2 depth (dense) · **beat vanilla continued-SSL on DINOv3 — NOT the off-the-shelf frozen DINOv3 (no step-0 probe row)** · continued-pretrain · ImageNet-1k (not low-data) · +1.14 mIoU / −0.023 RMSE (tiny) · steer SSL masks by gradient-alignment to a **small labeled** seg/depth pool · **transfer:** the lever needs **dense labels at pretrain time**, so it is label-guided-SSL, not our label-free bet. Could inspire a variant (steer PANO-iBOT masks with a tiny labeled pool) but that is a different, label-dependent bet.

**[Tier 2] Clustering-Guided Domain-Specific MAE for Arctic Remote Sensing** (Perera et al., 2026, arXiv:2605.30467)
- domain-adapted MAE (ViT-L) · Arctic VHR seg/detection (dense) · **beat ImageNet-init (+5–8 F1) and Prithvi-EO-2.0 general FM (≥15 F1) — but Prithvi is fine-tuned, not frozen DINO-class** · full-finetune · ~3M Arctic chips (large) · domain-gap closure · **transfer:** genuine FM-vs-FM dense win, but wrong axis (full-finetune, high-data, non-frozen non-DINO baseline). Domain gap they exploit is far larger than ours.

**[Tier 2] Subimage Overlap Prediction (task-aligned pretraining for RS seg)** (Sharma & Marin, CV4EO@WACV 2026, arXiv:2601.01781)
- task-aligned pretext: predict a subimage's overlap/location in its parent · RS seg (dense) · **beat FULLY-FINETUNED DINOv2 (0.6355 vs 0.6265 IoU)** · continued-pretrain from DINOv2 then full-finetune · **~10.7k images (genuinely low-data pretrain)** · overlap/localization pretext aligns to dense seg · **transfer:** *conceptually rhymes with our E2P overlap SSL*, and low-data — but the +0.009 IoU win is over a **fully-finetuned** DINOv2 (within noise), and the paper's real claim is **convergence speed / low-label robustness** (efficiency, not a frozen-ceiling break). Design inspiration only.

### Tier 3 — kept for context (most were dropped)

Of ~35 Tier-3 papers scouted, **most are dropped** as pure "beats from-scratch / other-SSL / weak baseline"
or photometric-consistency 360-depth (2110.10415, 2204.01027, 2203.09855, 2406.12849, 1811.05304, 2209.02952,
1909.08112, 2109.10563, 2104.14540, 2407.14126, 2203.14005, 2503.07125, 2503.09493, 2403.13430, 2512.23903,
2503.15917, 2310.19522, 2202.03026, 2111.12710, 2208.06049, 2205.03892, 2502.08769, 2311.09104, 2503.07561,
2508.20909, 2312.02366, 2604.10609 and others). The few kept below are the ones that *change how you read the
list*:

**[Tier 3] DINOCell: Self-Supervised Pretraining of Cell Segmentation Models** (2026, arXiv:2604.10609)
- DINOv2 continued/domain-adapted SSL · cell instance seg (dense) · **headline beats SAM (+10.42%) and random-init (+36–285%) — but the clean same-backbone test (continued-DINO vs plain DINOv2) is near-identical (+6.48% on ONE OOD set, minimal else)** · continued-pretrain + full-finetune · ~130k images · **transfer:** *keep because it is same-family (DINO) counter-evidence* — continued DINO SSL adds **almost nothing** over the foundation on dense. Mildly confirms our wall.

**[Tier 3] CAPI: Cluster and Predict Latent Patches for Improved MIM** (Darcet et al., TMLR 2025, arXiv:2502.08769)
- pure-MIM predicting Sinkhorn-clustered latent patches (EMA self-distill) · ADE20K/VOC/Cityscapes seg (dense, frozen-probe) · **beat all MIM baselines (MAE/iBOT/I-JEPA/data2vec) but lands 1.1–1.8 mIoU BELOW DINOv2+reg** · from-scratch · IN-22k/LVD-142M · **transfer:** *mild COUNTER-evidence* — a SOTA from-scratch MIM still **fails to beat the frozen foundation** on dense linear seg. Borrowable: the **tokenizer-free Sinkhorn-clustered latent target** (cleaner iBOT-style Term A target).

**[Tier 3] DINO-MVR: Multi-View Readout of Frozen DINOv3 for Medical Seg** (Jiang et al., 2026, arXiv:2605.07221)
- no new SSL — **multi-view entropy-weighted readout of FROZEN DINOv3** · medical seg (dense) · beats naive single-view frozen probe; recovers 98.4% of 40-patient perf w/ 5 patients · frozen+adapter · low-data · **transfer:** *champions the frozen FM* — confirms "accuracy comes from decoding frozen features, not continued SSL". Its **multi-view + entropy-weighted fusion** is structurally analogous to our E2P fusion+decoder and **could improve the frozen baseline's readout itself**.

**[Tier 3] BEiT v2: MIM with VQ-KD Semantic Tokenizer** (Peng et al., 2022, arXiv:2208.06366)
- MIM w/ CLIP-distilled VQ-KD discrete semantic target · ADE20K seg (dense) · beat BEiT/MAE + supervised · from-scratch+full-finetune · ImageNet-1k · **ViT-L 56.7 mIoU ADE20K** · semantic > pixel target · **transfer:** supports "semantic MIM target > pixel target" for Term A; CLIP/DINO used AS teacher, never as a frozen ceiling to beat.

**[Tier 3] Spherical View Synthesis for Self-Supervised 360 Depth** (Zioulis et al., 3DV 2019, arXiv:1909.08112)
- photometric spherical view-synthesis · 360 mono depth (dense) · **NO win — authors conclude supervised beats this SSL** · from-scratch · tens-of-k rendered panos · **transfer:** kept as an honest **negative** 360-dense-depth datapoint; consistency-style SSL failing to beat supervision echoes our wall.

---

## The CroCo → DUSt3R → MASt3R lineage — the published form of our "Term B"

`PANO_MIM_DESIGN`'s **Term B (cross-view masked completion via E2P warp)** is a panorama instantiation of
**cross-view completion (CroCo)**. This is the single most mechanistically-adjacent body of work, so it decides
whether the published record *validates* or *challenges* our frozen thesis. The verdict is unambiguous on one
axis:

> **Every member of this lineage pretrains the cross-view objective FROM SCRATCH (or full-finetunes), on
> LARGE multi-view corpora, and evaluates by FULL fine-tuning of the encoder. NONE of them continues from a
> frozen strong foundation and beats it under a frozen+PEFT constraint.** So the lineage powerfully validates
> the *mechanism* ("cross-view completion adds geometric/correspondence info single-view SSL lacks") but does
> **NOT** validate our *regime* (light continued-pretrain of a frozen DINOv3 on ~21.8k panos).

| Paper | id | pretrain origin | data scale | beat what | axis for us |
|---|---|---|---|---|---|
| **CroCo** (Weinzaepfel et al., NeurIPS 2022) | 2210.10716 | **from scratch** | ~1.8M synthetic pairs | MAE 79.6 / MultiMAE 83.0 → **85.6 δ1 NYUv2** (SSL-vs-SSL) | mechanism ✓, frozen ✗ |
| **CroCo v2** (Weinzaepfel et al., ICCV 2023) | 2211.10408 | **from scratch** | millions real+synth pairs | task-specific stereo/flow SOTA + single-view MAE + CroCo v1 | mechanism ✓, frozen ✗ |
| **DUSt3R** (Wang et al., CVPR 2024) | 2312.14132 | **continues from CroCo, then full-finetune** | ~8.5M pairs, ~8 datasets | COLMAP SfM/MVS + supervised depth/pose nets (NOT a frozen FM) | mechanism ✓, frozen ✗ |
| **MASt3R** (Leroy et al., ECCV 2024) | 2406.09756 | continues DUSt3R + matching head, full-network | millions, 14+ datasets | matching/localization SOTA (+30% VCRE AUC) (NOT a frozen FM) | mechanism ✓, frozen ✗ |
| **ZeroCo** (An et al., CVPR 2025 Highlight) | 2412.09072 | **reuses off-the-shelf CroCo, zero-shot** | (CroCo's 1.8M) | **frozen DINOv2** on zero-shot correspondence (9.41 vs 28.08 AEPE) + DUSt3R/MASt3R | **beats frozen FM, but on GEOMETRY not semantics** |
| **MuM** (2025) | 2511.17309 | **from scratch** | ~20M frames | **frozen DINOv3 AND frozen CroCo v2** on multi-view geometry | beats frozen FM, but geometry; DINOv3 still wins single-view **semantics** |
| **Muskie** (Li et al., 2025) | 2511.18115 | **from scratch** | large multi-view corpus | **frozen DINOv2/v3** on NN-correspondence (47.3 vs 8.5 AUC) | *consistency metric* — near-strawman for DINO; not a semantic-accuracy win |

**How to read this for us.** Three of these (ZeroCo, MuM, Muskie) *do* beat a frozen DINO on a metric — but
**always on geometry/correspondence, the task DINO was never built for**, which is close to the null hypothesis.
On **single-view semantics** (our actual wall, 57.7 mIoU seg) the record has DINOv3 **still winning** (MuM's own
concession). And the two that build genuinely strong geometry (DUSt3R/MASt3R) do it by **full-network training on
millions of pairs** — the exact capacity we deny ourselves by freezing.

**Net verdict for Term B.** The lineage says: *cross-view completion reliably adds **geometric/correspondence**
information*. That is real and worth exploiting — **but for our geometry-flavored downstream heads (depth,
normal, pointmap), not for the semantic-seg ceiling.** It supplies **no evidence** that a *light, frozen+LoRA,
low-data* Term-B lifts semantic-seg accuracy. It is the strongest *design* precedent and the weakest *regime*
precedent simultaneously.

---

## Boundary conditions — when continued in-domain SSL STOPS beating a strong foundation model

Synthesizing the Tier-boundary and negative evidence, SSL-beats-frozen-FM lives inside a narrow box defined by
three interacting axes. Cross any boundary and the accuracy win degrades into a consistency/efficiency mirage —
which is exactly the corner our project sits in.

1. **Backbone in-domain strength (the decisive axis).** SSL beats a frozen FM most cleanly when the FM is
   **weak in-domain** — huge headroom to fill. Every medical/endoscopy/remote-sensing "win" (EndoDAC 2405.08672,
   MedDINOv3 2509.02379, DINOCell 2604.10609, Arctic-MAE 2605.30467, DARES) rides a **domain-gap**: zero-shot DA
   is ~40% worse in surgery; DINOv2 is misaligned to CT/microscopy. **Our frozen DINOv3+E2P is already
   in-domain-strong** (57.7 mIoU, near-zero domain gap on indoor/outdoor panos). On this axis we are in the
   *hard* regime — the same axis that flips DINOCell's clean same-backbone test to "+6.48% on one OOD set,
   near-zero elsewhere."

2. **Data / capacity scale.** The MIM & cross-view "wins" (MAE, iBOT, CroCo→DUSt3R→MASt3R, MuM, CAPI) are all
   **large-data + full-finetune / from-scratch**. `2512.03862` (Diminishing Returns) formalizes it: MIM's benefit
   **shrinks as the starting representation quality rises**, and vanishes when you cannot move the backbone.
   `2512.23903` adds: SSL gains scale with **data diversity**, not model/adapter capacity. Our ~21.8k panos +
   0.59M frozen-LoRA is the **low-data, low-capacity** corner where the predicted headroom is smallest.

3. **Task granularity (global vs dense).** The cleanest continued-SSL-beats-frozen results are **classification**
   (DIET-CP 2509.06990 +2–5%; ExPLoRA +7.53 linear-probe). The moment they hit the **one dense task**, the win
   **collapses** — ExPLoRA's SpaceNet seg is an **exact 76.69 tie**. Dense per-pixel structure is where DINOv3 is
   already strong, so global-readout gains do not survive into dense.

**Tie to our own null (honest).** Our result — cross-view *consistency* SSL never beats frozen DINOv3 on dense
accuracy — is **not an anomaly; it is the predicted outcome** of sitting at the intersection of all three hard
boundaries (in-domain-strong backbone × low-data/frozen-capacity × dense task). **LoRA3D (2412.07746) explains
why**: consistency raises accuracy only when the pseudo-labels also carry a **correctness prior**. Our E2P-overlap
agreement was consistency **without** correctness — so it improved consistency and never accuracy, exactly as the
boundary-condition synthesis predicts.

---

## What this means for pano-ssl-dense (honest)

Referencing `PANO_MIM_DESIGN.md`'s PANO-iBOT bet (Term A = within-tile iBOT MIM; Term B = cross-view masked
completion; + Gram anchoring):

- **The MIM bet is still UNVERIFIED, not de-risked.** Across 67 papers there is **zero** case of a *light,
  frozen+low-data, dense* MIM beating a strong frozen foundation on **semantic** accuracy. The MIM builders
  (MAE/iBOT/SimMIM/MaskFeat/data2vec) prove MIM adds info **under full-finetune at scale** — a different regime.
  Proceed, but treat 57.7 as a genuinely hard target and instrument the ablation to isolate *whether the win
  survives the frozen constraint*.

- **Term B (cross-view completion) is your strongest DESIGN precedent but is likely a SEMANTIC mirage.** The
  CroCo lineage beats frozen FMs **only on geometry/correspondence** (ZeroCo, MuM, Muskie), while DINOv3 keeps
  winning single-view **semantics**. So aim Term B at your **depth / normal / pointmap** heads, where the
  evidence is genuinely positive — and do **not** expect it to move the seg-mIoU ceiling by itself.

- **A NeCo/DIP-style dense-consistency post-train is the most promising NON-MIM alternative to pilot.** These are
  the only two Tier-1 papers that beat a frozen DINO on **dense** tasks, and DIP wins on **both seg and depth**
  with an explicit **low-label** gain. Risk: both win under **probe/in-context** eval and DIP **unfreezes 3
  blocks** — you must re-check under your **trained fusion+decoder** protocol before believing the gain. This
  also reframes our wall: it may be specific to *cross-view* consistency, not consistency-SSL per se.

- **Borrow the correctness-prior lesson from LoRA3D (2412.07746), not its mechanism.** Its win came from a
  correctness-carrying multi-view optimizer, which we cannot replicate for semantics. But it is the strongest
  argument **FOR** MIM: masking forces reconstruction of **real signal** (a correctness prior), unlike pure
  agreement. Make PANO-iBOT's targets **latent/feature-space** (MaskFeat 2112.09133, data2vec 2202.03555, BEiTv2
  2208.06366, CAPI 2502.08769) so the reconstructed signal is semantic, not low-level.

- **Steal two guardrails wholesale.** (1) **LoRA-regularization-against-drift** from WeSTAR (2511.14238) and the
  **update-embeddings+decoder-only anti-collapse** recipe from Re-Depth (2512.17908) to prevent catastrophic
  forgetting during continued pretrain. (2) **Keep every objective dense-aligned** (Diminishing-Returns 2512.03862
  shows a misaligned intermediate classification stage erases the SSL gain) — Term A and Term B already are.

- **Likely mirages (do not over-index on these as validation):** any photometric/temporal **consistency** 360-depth
  result (dropped Tier-3 cluster); **supervised** LoRA-beats-frozen (2409.09907 — we already own this via SGFormer-tie
  0.104); **domain-gap** medical/endoscopy wins (2405.08672 etc. — our backbone is already in-domain-strong);
  **efficiency/convergence** framings (2601.01781). These re-confirm the wall; they do not crack it.

---

## Footer — verified arXiv ids (as reproduced from the evidence set)

1811.05304 · 1909.08112 · 2104.14540 · 2109.10563 · 2111.06377 · 2111.07832 · 2111.09886 · 2111.12710 ·
2112.09133 · 2202.03026 · 2202.03555 · 2203.09855 · 2203.14005 · 2204.01027 · 2205.03892 · 2208.06049 ·
2208.06366 · 2209.02952 · 2210.10716 · 2211.10408 · 2301.03580 · 2310.19522 · 2311.09104 · 2312.02366 ·
2312.14132 · 2403.13430 · 2405.08672 · **2406.09756** (MASt3R) · **2406.10973** (ExPLoRA) · 2406.12849 ·
2407.14126 · 2408.11054 · 2408.17433 · 2409.09907 · 2412.07746 ·
2412.09072 · 2412.14103 · 2502.08769 · 2503.07125 · 2503.07561 · 2503.09493 · 2503.15917 · 2506.18463 ·
2508.20909 · 2509.02379 · 2509.06990 · 2509.17816 · 2509.23991 · 2511.14238 · 2511.17309 · 2511.18115 ·
2512.03862 · 2512.17908 · 2512.23903 · 2601.01781 · 2601.22108 · 2602.08505 · 2603.27904 · 2604.10609 ·
2605.07221 · 2605.23472 · 2605.30467 · 2606.27745

**No arXiv id (venue-only):** *A Systematic Study on Pretraining Strategies for Low-Label RS Seg* (Sensors/MDPI,
DOI 10.3390/s26041385) · *DASC-SPT* (WACV 2025, CVF) · *VLMs Do Not Transfer to Medical CXR* (medRxiv
10.64898/2025.12.06.25341759)

---

**Citation verification (2026-07-07, primary-source spot-check).** The 3 Tier-1 papers and a discriminating
sample of 6 post-knowledge-cutoff ids were fetched from arXiv to distinguish isolated errors from systemic
fabrication: **all resolved to real papers with matching claims.** Confirmed clean: DIP 2506.18463 (method +
direction confirmed; the exact ADE20K/VOC/NYU figures are extraction-sourced, not re-checked at abstract level),
NeCo 2408.11054 (numbers confirmed verbatim; official title is *"Near, far: Patch-ordering…"*, NeCo is the
method), LoRA3D 2412.07746 (88% confirmed), MuM 2511.17309, Diminishing-Returns 2512.03862, DINOCell 2604.10609,
Arctic-MAE 2605.30467, ReDepth 2512.17908, Survey 2606.27745. Two ids were corrected (see "One id note" above:
MASt3R → 2406.09756). Well-known ids (CroCo, DUSt3R, MAE, iBOT, etc.) were not re-fetched.
