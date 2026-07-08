# SSL for Accuracy — creative design proposals (gated & red-teamed)

**Purpose.** Three ranked SSL designs that try to do the one thing this project has never done: move
**single-view downstream accuracy** (not consistency) past the frozen DINOv3 + E2P + LoRA champion
(Stanford2D3D seg fold-1 = **57.7 mIoU**). Every design is filtered through the project's iron laws,
red-teamed honestly, and paired with the single cheapest experiment that would greenlight or kill it.

**Companion docs (read alongside):**
- `docs/SSL_SUCCESS_CASES_LIT.md` — the only 3 papers that beat a *strong frozen* foundation on **dense
  accuracy** (NeCo, DIP, LoRA3D), the CroCo→DUSt3R→MASt3R "Term B" lineage, and the boundary-conditions
  synthesis (in-domain-strong backbone × low-data/frozen × dense = the hard corner we sit in).
- `docs/CONSISTENCY_AND_RICHNESS_LIT.md` — why every pure-consistency objective erodes richness
  (purity 0.838→0.730, erank 39→25), and the equivariant / relational / coding-rate cures.
- `docs/SEMANTIC_IDENTITY_SSL.md` — the M1/M2 gating record and the **frozen-probe M2 trap** (§8):
  *no frozen probe can settle M2* because it is bounded by `single_fair`.
- `docs/PANO_MIM_DESIGN.md` — the PANO-iBOT Term A / Term B masked-prediction design (Design #2 below).

**Provenance note (honesty).** The in-workflow "red-team" stage returned only a stage-completion stub, so
the objections in the original draft were reconstructed from durable evidence. A **genuine independent
red-team was then run separately (critic agent, 2026-07-07)** with codebase access; it found a decisive
correction — **the gate lacked a self-supervision precondition (Q0), so a GT-supervised design (MG-SOG)
wrongly ranked #1.** That correction is now integrated: Q0 added to the gate, ranking inverted to
**#1 PANO-iBOT · #2 RICA · MG-SOG excluded** (see the 자가지도 교정 bullet and the banner under "Ranked
designs"). Section bodies below retain the original order with corrected header labels.

---

## 한국어 요약 (executive summary)

- **벽의 정의 (iron laws).** (1) *consistency ≠ accuracy* — 810·21.8k 스케일에서 확인. overlap
  feature를 더 동일하게 만들어도 단일뷰 정확도는 절대 안 움직였다. (2) 앙상블/blend 이득(+0.088 mIoU)은
  **분산 감소**라서 단일뷰로 구조적 복원 불가(M2/D-A). **어떤 frozen 프로브도 M2를 settle 못 한다** —
  frozen feature(single_fair)가 상한이라 인코더의 도달집합을 bound할 수 없다. (3) **강한 decoder는
  feature-level 이득을 세탁한다**(F-2가 linear probe +0.078 → UUPerNet에서 mean에 패배, §3.9). (4)
  anchor를 풀면 semantics가 침식된다(purity·erank↓); TC3(고정 teacher distill 지배 + 기하 대응)만이
  비침식 레시피지만 그조차 정확도엔 consistency-only.
- **3-부 게이트 (Q1/Q2/Q3).** 어떤 SSL 제안도 통과해야 산다: **Q1** 단일뷰로 접근 가능한 *새 정보*를
  주는가(단순 consistency는 탈락) · **Q2** 그 이득이 단일뷰에 존재하는가(앙상블/시차 변량감소면 M2 함정
  → 탈락) · **Q3** 진짜 DPT/UPerNet decoder·multi-seed에서 살아남는가(약한 probe에서만 보이면 세탁 →
  탈락).
- **⚠️ 자가지도 교정 (독립 레드팀 2026-07-07, 이 요약이 아래 초판 순위를 대체).** 초판 게이트에
  **자가지도 전제(Q0)가 빠져** Q1이 시뮬레이터 GT 라벨을 '새 정보'로 허용 → **GT-supervised 설계
  (MG-SOG)가 잘못 #1**이 됐다. 사용자의 '자가지도학습 기반' 요구엔 **Q0 — 새 정보 항이 사람/GT/시뮬레이터
  라벨에서 자유로운가? — 를 Q1 위에 추가**해야 하며, Q0 적용 시 순위가 뒤집힌다.
- **교정된 자가지도 랭킹.** **#1 PANO-iBOT** (masked latent prediction, 라벨 0, 이미
  `scripts/train_pano_ibot.py`에 Term A 구현됨) — 진짜 자가지도이자 사용자 질문의 정답. 단 **seg 57.7
  돌파 근거는 67편 중 0건**이고 신뢰 가능한 이득은 **depth/normal/pointmap** 헤드(Term B). · **#2 RICA**
  (외부 FM oracle **feature** distillation — 라벨 없어 자가지도 맞음; 단 cheap frozen test로 게이팅,
  ~60% Q1에서 사망 예상). · **제외 — MG-SOG:** 정확도 레버가 Structured3D **GT 래스터**(normal.png/semantic.png)라 자가지도 아님
  (무료 보조 supervision) → **Q0 실패만으로 제외 충분**. (정확히: `annotation_3d.json`은 코드에서 미로드지만
  등가 GT인 normal/semantic 래스터는 `data.py:s3d_gt_path`로 **이미 로드 가능**(다운스트림 eval용) →
  '파이프라인에 없어서'가 아니라 **'GT라서'** 제외.) 유일한 자가지도 변형(전역 중력벡터)은 고정 렌더-pitch
  메타데이터라 Q1도 실패. 제약을 '시뮬레이터 supervision 허용'으로 풀면 훌륭한 *supervised 보조과제*지만 SSL은 아님.
- **4번째 자가지도 후보(초판 과소평가):** **NeCo식 dense-neighbor relational post-train** (arXiv:2408.11054) —
  Tier-1 중 우리 frozen+adapter와 **구조가 동일한 유일 방법**. RICA와 함께 파일럿 가치.
- **첫 수(진단 후 학습, 교정).** 자가지도 기준 첫 수 = **PANO-iBOT Term-A-only 최소 LoRA continued-pretrain
  (~수 시간)** 을 **실제 decoder·multi-seed**로 seg fold-1 평가(선형 probe 금지 — §3.9 세탁). 57.7 초과면
  greenlight + Term B(geometry 헤드 조준), frozen+noise 이하면 seg 천장은 hard-target으로 두고 Term B의
  depth/normal 이득으로 피벗. (MG-SOG gravity 진단은 사용자가 supervision 허용으로 제약을 풀 때만.)

---

## The gate (restated) — every design answers Q0 then these THREE before anything else

A proposal is **dead-on-arrival** unless it clears Q0 + all three. They are the operational form of the iron laws.

- **Q0 — Self-supervised? (added by red-team; upstream of everything).** Is the term that injects the
  **new information** (the accuracy lever, the thing Q1 is about) free of human / GT / simulator labels?
  Passing forms: masked reconstruction of the model's own signal, contrastive/consistency, or distillation
  from **another model's features** (no labels). **Failing forms: GT/simulator annotations** (e.g.
  Structured3D `annotation_3d.json` plane normals, `normal.png`/`semantic.png` rasters). A dominant
  self-supervised *anchor* (distill/Gram) does NOT make a design SSL if its **new-info term** is supervised.
  *Note: this is the correction that inverted the initial ranking — see the 자가지도 교정 bullet above.*
- **Q1 — New single-view-accessible information?** Does the objective inject information that a single
  perspective tile's frozen DINOv3 features *do not already contain* and *could* carry? A pure
  consistency/agreement objective adds **zero** new information (iron law 1) → auto-fail. Passing forms:
  an **external** signal (a second FM, a simulator annotation, a physical prior) or **reconstruction of
  real hidden signal** (a correctness prior, per LoRA3D).
- **Q2 — Single-view, not ensemble/parallax?** Is the gain present in *one* view at inference, or is it
  variance-reduction over independent views (structurally unrecoverable — iron law 2, the M2 trap)? Any
  design whose payoff is "distill the multi-tile ensemble's accuracy into single-view features" fails
  unless it adds genuinely new single-view-accessible information. **A frozen probe cannot answer Q2** —
  it is bounded by `single_fair`; Q2 for an ensemble-flavored idea can only be settled by a short real
  training run.
- **Q3 — Survives a real decoder (laundering-proof)?** Does the gain survive a **trained DPT/UPerNet
  decoder, multi-seed** — not just a linear/kNN/in-context probe? Iron law 3 / §3.9: F-2 beat mean by
  +0.078 under a linear probe and then **lost** to uniform masked-mean under a real UPerNet across seg
  *and* depth. **A probe-only win does not count.** Every design's "laundering-proof eval" below is
  explicitly under a real decoder, multi-seed, on the champion harness.

---

## Ranked designs

> **⚠️ Ranking corrected (red-team, Q0).** The section order below is the *initial* (pre-Q0) ranking.
> The **authoritative self-supervised ranking is: #1 PANO-iBOT · #2 RICA · MG-SOG excluded** (its accuracy
> lever is GT rasters = supervised → Q0 fail). Section headers are relabeled accordingly; MG-SOG is kept in
> full only as a *supervised auxiliary-task* alternative for if the user relaxes the SSL constraint.

### [DEMOTED — supervised, NOT self-supervised; kept only if the SSL constraint is relaxed] — MG-SOG: Manhattan/Gravity Structured Ordinal Geometry distillation

> **Red-team disqualification (Q0).** MG-SOG's accuracy lever is a `CE(patch_feat → gravity/Manhattan/edge)`
> loss whose targets are **Structured3D ground-truth** (world-frame normals/junctions). That is free/auxiliary
> **supervision**, not self-supervision — it fails Q0 and does not answer the user's 자가지도 request. Two
> further facts (verified): (1) the exclusion rests on Q0 alone and that is sufficient — **any MG-SOG target
> is GT**. The specific `annotation_3d.json` is never loaded in code (grep-confirmed), but the *equivalent* GT
> rasters (`normal.png`/`semantic.png`) ARE loadable via `data.py:s3d_gt_path` (they already supply the
> downstream eval), so the honest reason is "it's GT," not "it's unavailable" (`geometry.render_coordmap`
> only renders pixel-index coordmaps, so target-rendering would reuse those eval GT rasters). (2) The only
> self-supervisable sub-signal, the **global gravity vector, is fixed by the tile's known render pitch**
> (`train_ssl.py` DOMAINS pitch rings) → predicting it memorizes metadata → fails Q1. The Q1-passing part
> (per-pixel surface frame) is exactly the part that needs GT. **No genuinely-SSL MG-SOG variant clears the
> gate.** The frozen gravity-separability probe (below) remains a *valid diagnostic* — but the design is a
> supervised proposal. Read the mechanism below only if simulator supervision is acceptable.

Inject a **world-frame geometric prior** (gravity direction + Manhattan axes + layout structure) that
DINOv3's planar, gravity-agnostic patch features do not encode, using **free** dense labels from the
Structured3D simulator annotation. This is not a consistency objective — it is a *new external label*.

**Gate answers (up front):**
- **Q1 — YES.** Gravity direction and the 3 orthogonal Manhattan world axes are **not** determinable from
  a single planar DINOv3 tile (DINOv3 has no gravity prior; a rotated tile looks the same to it). The
  label is external and free (Structured3D `annotation_3d.json`: planes with world-frame normal+offset,
  junctions, Manhattan axes). New information, class-(c) external-label per the parallax bifurcation.
- **Q2 — YES, clears M2 cleanly.** One panorama determines its own layout; inference uses one view. The
  signal is *not* ensemble variance-reduction and not parallax — it is a per-pixel world-frame label.
  Because Q2 is about being single-view-native (not an ensemble transfer), it does **not** hit the frozen-
  probe M2 trap; the design never asks a single view to reconstruct a multi-view average.
- **Q3 — RISKY-but-plausible.** The head forces a **gravity-canonical world frame into the feature
  content** *upstream* of the decoder — a feature-content change, not a probe-space reshaping. Plausibly
  survives a decoder because a UPerNet cannot re-derive gravity from a single planar tile on its own.
  This is the load-bearing risk (see objection).

**Concrete loss / arch (reusing existing assets):**
- Render per-ERP-pixel targets from `annotation_3d.json` via the existing coordmap machinery
  (`geometry.py: render_coordmap`, `WarpField`), then sample per E2P tile patch token:
  1. **gravity-frame orientation class** — floor(up) / ceiling(down) / wall(3 quantized Manhattan azimuth
     bins) / clutter (6-way);
  2. **Manhattan-axis assignment** — which of 3 world axes the surface faces;
  3. **layout structural-edge proximity** — soft target from junction projection (ordinal).
- **Head:** a 2-layer classifier on frozen+LoRA patch features (`encoder.py: PanoEncoder` + LoRA q/v;
  reuse the `CodeHead` pattern for the classifier).
- **Anchor stays dominant (anti-erosion, iron law 4):** keep TC3 fixed-teacher distill the dominant term.
  ```
  L = distill(student, sg[frozen_teacher])          # losses.distill_loss — DOMINANT (anchor thesis)
    + λ_geo · CE(patch_feat → gravity/Manhattan/edge targets)   # the new-info term
    + λ_gram · gram_anchor(student, frozen)          # losses.gram_anchor — erosion guard
    + λ_koleo · koleo(cls)                            # losses.koleo — anti dimensional-collapse
  ```
  λ_geo tuned so distill stays dominant (anchor thesis: weakening the anchor eroded purity 0.838→0.730 in
  M1). No new consistency term is added — this is deliberately not another agreement objective.

**Cheapest falsification test (upgraded — frozen, no training, then one real run):**
1. *Frozen Q1 pre-filter (no training, ~minutes):* render gravity-class labels; train a **linear probe**
   `frozen patch feat → gravity-class` (reuse `diag_semantic_headroom.py` harness pattern). If frozen
   DINOv3 **already** predicts gravity-class at high accuracy → the info is already present → **KILL**
   (Q1 fails). If it **cannot** → headroom exists, proceed. *(This is a legitimate frozen test because it
   measures presence-of-information, not an ensemble-recoverability question — it is not the M2 trap.)*
2. *Real settling run (the greenlight test):* light MG-SOG LoRA continued-pretrain (~few h on S3D), then
   eval under the **real decoder** on seg fold-1 (see laundering-proof eval). Frozen probes cannot settle
   whether the injected frame helps accuracy under a decoder — only this run can.

**Laundering-proof eval (iron law 3):** train the full UPerNet (seg) and DPT (depth/normal) decoders on
the champion harness (`seg_s2d3d_bench.py`, `depth/normal` benches), **multi-seed (≥3)**, MG-SOG-LoRA
features vs frozen-DINOv3 features. A win counts **only** if it holds under the real decoder multi-seed —
a linear-probe-only gap is presumed laundered and discarded.

**Honest EV (incl. most-likely-null):** best-positioned of the three because it is the only one adding a
*genuinely external, single-view, non-consistency* signal (the class the boundary-conditions synthesis
says can move the ceiling). **Most-likely null:** the seg/normal decoder can already extract enough
world-frame structure from context that the injected label is **redundant → laundered** (iron law 3), OR
the win is real on synthetic S3D but **does not transfer** to real Stanford2D3D fold-1 (synthetic→real
gap). Expected outcome distribution: ~35% real multi-seed seg gain, ~40% laundered/null, ~25% S3D-only.

**Surviving strongest objection + response:**
- **Objection (SERIOUS, Q3/iron-law-3):** gravity/Manhattan structure is heavily **correlated with
  surface-normal and layout GT**, which a DPT/UPerNet decoder is powerful enough to re-derive from raw
  frozen features — so the feature-level injection is exactly the kind of gain §3.9 shows evaporating
  under a real decoder.
- **Response/mitigation:** the frozen Q1 pre-filter directly measures this — if a *linear* probe on frozen
  features already recovers gravity-class, the decoder certainly can, and we KILL before training. We only
  proceed if frozen features **cannot** linearly express the frame, which is the regime where a decoder is
  least able to launder it. Additionally, evaluate on **DensePASS-style oblique/edge regions** and report
  **oblique-region mIoU** separately — the injected world frame should help most exactly where the planar
  prior is weakest, a region a context-only decoder cannot trivially recover.
- **Second objection (scope):** Manhattan/gravity is **indoor-only** — dead for DensePASS outdoor. Honest
  scope limit: MG-SOG is an **indoor-seg / depth / normal** bet (Structured3D, Stanford2D3D), not a
  universal recipe. This is acceptable because the champion 57.7 is itself an indoor result.

---

### #1 (SSL) — PANO-iBOT: within-tile + cross-view masked latent prediction (Term A + Term B)

> **Corrected rank #1 for the user's self-supervised ask** (was #2). Genuinely self-supervised (masked
> reconstruction, no labels), already implemented in `scripts/train_pano_ibot.py`. Honest caveat unchanged:
> no evidence it beats 57.7 on **seg** (light frozen+LoRA MIM has never beaten a strong frozen FM on
> semantics in the 67-paper set); its credible win is on the **depth/normal/pointmap** heads.

Continue DINOv3's native dense objective (masked-latent iBOT) on panorama tiles, plus a **cross-view
masked completion** term over the E2P overlap. This is the *reconstruction-adds-real-signal* bet
(correctness prior, per LoRA3D), not a consistency bet. Full design in `docs/PANO_MIM_DESIGN.md`.

**Gate answers (up front):**
- **Q1 — YES (by construction, but see the honest split).** Masked prediction forces reconstruction of
  **hidden real signal** — a correctness prior, unlike pure agreement (the LoRA3D lesson). Term A imprints
  pano statistics; Term B injects **cross-view distortion/geometry** the planar prior never saw. New info:
  yes. *Caveat:* the lit record says Term B's new info is **geometric/correspondence**, not semantic (see
  objection).
- **Q2 — YES.** Both terms operate on and produce single-view features; inference is single-view. Term B's
  target is **A's own view-specific teacher feature** at masked locations (the de-overlap rule), so it is
  not an ensemble-average transfer → not the M2 trap.
- **Q3 — PLAUSIBLE but unproven at our regime.** MIM builders (MAE/iBOT/BEiTv2) prove masked-latent
  features survive a real decoder — but **under full-finetune at scale**, never at *frozen+LoRA+21.8k*.
  This is the honest gap.

**Concrete loss / arch (reusing existing assets):**
- Student S = `PanoEncoder` + LoRA(q/v) (optionally last 1–2 blocks unfrozen per ExPLoRA); Teacher =
  EMA(S) via `encoder.ema_update` (momentum 0.996→1.0, stop-grad); Frozen anchor F = original DINOv3.
- **Term A (within-tile iBOT MIM):** block-wise mask ~40–50% of each tile's patch tokens; latent target.
  ```
  L_A = ibot_loss(student_scores_masked, sharpen∘center(teacher_scores))   # losses.ibot_loss + losses.sinkhorn
  ```
- **Term B (cross-view masked completion):** mask a contiguous block in tile A whose rays are visible in
  overlapping tile B (different obliquity, via `geometry.warp_field_from_coordmaps`); a shallow predictor
  P (conditioned on relative obliquity/pose) predicts the masked A-patch **A-view teacher** tokens from
  B's visible tokens + A's visible context.
  ```
  L_B = 1 − cos( P(context_B, geo), sg[ teacher_A(masked_patch) ] )
  ```
  Non-triviality (de-overlap rule — what F-3 got wrong): target is A's own teacher feature, region hidden
  in A, high mask ratio → P must learn the cross-view distortion transform, not copy B.
- **Anti-collapse / anchor:** `losses.gram_anchor(student, frozen)` (DINOv3 Gram anchoring) +
  `losses.koleo` + teacher centering/Sinkhorn+sharpening. Distortion-aware loss weighting ∝ cos(lat).
  ```
  L = L_A + λ_B·L_B + λ_gram·gram_anchor + λ_koleo·koleo
  ```

**Cheapest falsification test (upgraded — real run, not a probe):** implement **Term A only**, light
continued-pretrain 21.8k (~few h), eval on seg fold-1 vs frozen 57.7 under the **real decoder** (not the
linear probe — a linear-probe early-check is a *pre-filter only*, since §3.9 proves it can false-positive).
If A ≥ frozen multi-seed → add Term B and re-eval on the **geometry heads** (depth/normal), especially
oblique-region metrics.

**Laundering-proof eval (iron law 3):** UPerNet (seg) + DPT (depth/normal), multi-seed ≥3. Term-A judged
on seg fold-1 vs 57.7; **Term-B judged on depth/normal/pointmap heads**, where the lit evidence is
positive — *not* asked to move seg by itself.

**Honest EV (incl. most-likely-null):** strongest *design* precedent (CroCo→DUSt3R→MASt3R is the published
Term-B), weakest *regime* precedent (all wins are full-finetune/from-scratch/large-data). **Most-likely
null:** at 21.8k the imprint is **frozen+ε**, not a decisive multi-seed seg win; Term B degenerates to
trivial copy → collapses back to consistency (≈frozen) if de-overlap/high-ratio isn't strict. Expected:
Term A ~ frozen+ε on seg (~30% clear seg win); Term B **positive on depth/normal** (~55%), **null on seg
mIoU** (the semantic-mirage prediction).

**Red-team objection + response (SERIOUS):**
- **Objection:** the entire CroCo→DUSt3R→MASt3R lineage beats a frozen DINO **only on geometry/
  correspondence** (ZeroCo, MuM, Muskie), while DINOv3 keeps winning single-view **semantics** (MuM's own
  concession). So Term B is very likely a **semantic mirage** on the 57.7 seg ceiling.
- **Response/mitigation:** *accept the objection and re-aim.* Term B is scoped to **depth / normal /
  pointmap** heads where the evidence is genuinely positive, and is **not** expected to move seg mIoU.
  Term A (native dense imprint) carries the seg bet, and even it is treated as "hard target, may be
  frozen+ε." This is an honest down-scoping, not a defense of the seg claim.

---

### #2 (SSL) — RICA: Retrieval-In-Context Anchoring (external-oracle variant only; gated on cheap frozen tests)

> **Corrected rank #2 for the user's ask** (was #3). Genuinely self-supervised — the new-info term is
> distillation from a **second model's features** (DIFT / DINOv2-reg oracle), no labels. Do NOT disqualify on
> SSL grounds. Its problem is the gate, not the label: ~60% likely killed at the frozen oracle test (Q1),
> and probe-laundering-prone (Q3). Keep only because falsification is cheap and pre-training.

Reshape DINOv3 patch features via a DIP-style in-context nearest-neighbor pseudo-task, where the retrieval
targets are an **external correctness prior** — dense correspondences from a *second* frozen model (DIFT /
SD-DIFT, or DINOv2-with-registers as an oracle) carrying a different dense inductive bias. LoRA is trained
so DINOv3 query patches land on the oracle's true dense match. **Down-ranked** (not folded) because the
whole design hinges on an unproven premise the frozen kill-test may falsify.

**Gate answers (up front):**
- **Q1 — CONDITIONAL.** New info **only if** the oracle carries dense-matching structure not derivable
  from a single view's frozen DINOv3. A pure NeCo/DIP replica with DINOv3 as its own target adds **nothing**
  (fails Q1). The external oracle is the *only* thing that could pass Q1 — and whether it does is exactly
  what the kill-test measures.
- **Q2 — YES (if Q1 holds).** Single-view feature reshaping; not ensemble/parallax.
- **Q3 — RISKY (this is the down-rank reason).** NeCo/DIP win under **kNN / in-context** probes; our 57.7
  is head-driven. Feature *reshaping* that helps retrieval is precisely the **F-2 §3.9 laundering
  pattern** — likely to tie under a real 2.4M+ decoder even if it wins under kNN.

**Concrete loss / arch (reusing existing assets):** DIP-style in-context NN pseudo-task; support-set match
labels come from the oracle FM's dense correspondence. LoRA-only on `PanoEncoder`; anchor via
`gram_anchor` + dominant `distill_loss` (anchor thesis). The retrieval loss reuses the `sinkhorn` /
`code_swap`-family assignment machinery in `losses.py`.

**Cheapest falsification test (frozen, discriminating):**
1. Does the oracle's dense-NN retrieval on **frozen** DINOv3 features already agree with GT semantic
   identity **better** than DINOv3's own NN? If the oracle adds nothing over frozen DINOv3 on
   GT-consistency → **Q1 fails → dead.**
2. Linear-probe-vs-decoder gap on one split: if reshaped features beat frozen **only** under kNN and tie
   under the real decoder → **F-2 pattern → dead.**

**Laundering-proof eval:** must survive the real UPerNet/DPT decoder multi-seed; a kNN/in-context-only win
is discarded by construction (this is the design's highest risk).

**Honest EV (incl. most-likely-null):** lowest of the three. **Most-likely null:** (a) the oracle adds no
GT-relevant info over frozen DINOv3 (Q1 fail), or (b) the gain is a probe-only reshaping that launders
under a decoder (Q3 fail). Expected: ~60% killed at the frozen oracle test, ~25% laundered, ~15% survives.

**Red-team objection + response (near-FATAL, hence down-ranked):**
- **Objection:** this is DIP/NeCo, whose Tier-1 wins are measured under **in-context/kNN eval and DIP
  unfreezes 3 blocks** — under our frozen+LoRA + real-decoder protocol the gain is the presumed-dead
  laundering class (iron law 3). And the "external oracle adds info" premise is unverified.
- **Response:** *don't build it until the two frozen kill-tests pass.* Both are cheap and settle the design
  before any training. If either fails, RICA is **folded into the graveyard**, not pursued. It is ranked #3
  precisely because its survival probability is low and fully gated on a cheap frozen check.

---

## Killed ideas (graveyard — do NOT re-propose)

Verbatim from the project graveyard, plus this memo's folds:

- **M1** — learned overlap semantic-code agreement → **eroded** (purity 0.838→0.730). Consistency-only.
- **F-2** — learned set-fusion → beat mean +0.078 under a **linear probe**, then **lost** to uniform
  masked-mean under a real UPerNet across seg *and* depth (5-seed). The canonical laundering case (§3.9).
- **F-3** — Pano-JEPA EMA masked-**whole-tile** → **eroded**, VICReg dormant. (Distinct from Design #2:
  #2 masks **patch tokens** block-wise with a latent target + Gram anchor, the modern MIM recipe F-3 lacked.)
- **Deformable cross-attention** (geometry-guided) → **tied** naive.
- **Obliquity / confidence weighting** → **dead**.
- **Plain warp-equivariance consistency** (`losses.warp_equivariance_loss`) → **consistency-only** (no
  accuracy). *Note:* this is the DEAD *invariance* form. It is NOT the same as any equivariant *reframing*
  (predict-the-warp / displacement, `CONSISTENCY_AND_RICHNESS_LIT §Equivariant`), which is a live but
  separate hypothesis — do not conflate the two.
- **M2 / ensemble→single accuracy transfer** — **plausibly-flat, low-EV, back-of-queue** (not cleanly
  killed, not cleanly tested). Iron law 2: the +0.088 blend gain is variance-reduction, structurally
  unrecoverable by a single view. **No frozen probe can settle it** (bounded by `single_fair`); the only
  clean test is a short real ensemble→single LoRA training run, which is low-EV.
- **Fold from this memo — RICA-without-oracle** (a plain NeCo/DIP self-target replica): fails Q1 (no new
  info) → dead. Only the *external-oracle* variant is kept, and only pending its frozen kill-test.
- **Excluded from the SSL ranking — MG-SOG** (Q0 fail): accuracy lever is Structured3D **GT rasters**
  (normal/semantic — supervised); its only self-supervisable sub-signal (global gravity) is fixed
  render-pitch metadata (Q1 fail). (The GT rasters ARE loadable via `s3d_gt_path`, so it's excluded for
  being GT, not for being unavailable — only `annotation_3d.json` specifically is unwired.) A valid
  *supervised auxiliary-task* proposal, not an SSL answer. Re-admit only if the user allows simulator supervision.

---

## Recommended first move (diagnose before train) — corrected for the self-supervised ask

Under the user's **자가지도학습 기반** constraint (Q0), MG-SOG's gravity probe is **not** the first move
(MG-SOG is supervised and its GT isn't in the pipeline). The genuinely-SSL sequence:

**Primary move — PANO-iBOT Term-A settling run (the top SSL design).** Term A is already implemented
(`scripts/train_pano_ibot.py`). Because §3.9 proves a linear-probe win launders under a real decoder — and
because no frozen probe can settle a Q3 decoder-accuracy question — the decisive experiment is a **real
training run**, not a probe: light Term-A-only LoRA continued-pretrain on 21.8k S3D panos (~few h), then
evaluate under the **real UPerNet (seg) / DPT (depth-normal) decoder, multi-seed ≥3**, on Stanford2D3D fold-1.
- **Greenlight** (add Term B, aimed at geometry heads): Term-A features clear frozen **57.7 by a multi-seed
  margin** on seg — or show a clear multi-seed depth/normal gain.
- **Kill / re-scope**: ≤ frozen + noise on seg → treat the seg ceiling as an unproven hard target, keep only
  if Term B moves **depth/normal/pointmap** (where the CroCo-lineage evidence is genuinely positive).

**Cheap parallel probes (both genuinely SSL, both settle before a big run):**
- **NeCo-style dense-neighbor relational post-train** (arXiv:2408.11054) — the one Tier-1 method structurally
  identical to our frozen+adapter regime; a `diff-sort` ordering loss on E2P overlap patches vs the frozen
  teacher. Pilot it against Term A. *(Elevate the existing `losses.py:gram_anchor` at the same time — Gram
  anchoring is the published TC3 formalization; cheap A/B.)*
- **RICA frozen oracle test (#2)** — does a second FM (DIFT / DINOv2-reg) beat frozen DINOv3 on GT-consistency
  dense NN? If not → Q1 fail → fold RICA. Minutes, no training.

**Only if the user relaxes the constraint to allow simulator supervision:** run the MG-SOG frozen
gravity-class linear-separability probe (`diag_semantic_headroom.py` pattern) — a *valid* presence-of-info
diagnostic (not the M2 trap) — then a supervised MG-SOG LoRA run under the real decoder. The target would
reuse the **GT rasters** (`normal.png`/`semantic.png`) that `data.py:s3d_gt_path` already loads for eval —
so it is easily wireable, but it is GT (Q0-failing), which is exactly why it sits outside the SSL ranking.

This ordering respects the project law (diagnose before train), answers the *self-supervised* question the
user actually asked, and never uses a frozen probe to settle an accuracy question it structurally cannot answer.
