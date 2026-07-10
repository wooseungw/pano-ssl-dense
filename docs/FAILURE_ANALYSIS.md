# Why SSL-for-accuracy never cleared the wall — pano-ssl-dense failure analysis

**Scope.** This is the durable, red-teamed post-mortem of *every* attempt in this project to move
**single-view downstream accuracy** above the frozen DINOv3 + E2P + LoRA champion (Stanford2D3D seg
fold-1 = **57.7 mIoU**; scatter-mean field = **0.557 @64×128**, `RESULTS.md` §3.8). It does not
re-litigate the individual experiments — those are logged in `RESULTS.md`, `SEMANTIC_IDENTITY_SSL.md`,
`CAN_SSL_RAISE_ACCURACY.md`, `PANO_ADAPT_RECIPE_GATE.md`, `INVVAR_FUSION_LOG.md`, `MOGE_PARALLAX_LOG.md`.
It compresses them into the **2–4 root causes that generate the entire failure roster**, the necessary
conditions any winner must satisfy, and an honest forward read on the one live bet, **[C]** (inter-pano
parallax).

**Provenance (honesty, house move).** The synthesis below is the fixed analytical premise of this
session; every empirical anchor was re-confirmed against source this session. The in-workflow
adversarial-critic agent stalled mid-stream (API error) and returned empty; a **genuine independent
critic was then re-run separately** (oh-my-claudecode:critic, Opus, codebase access, 2026-07-09) — see
**§7**. Its verdict is **ACCEPT-WITH-RESERVATIONS**: no misfit failure, no missing 4th cause (capacity /
eval-validity / distribution-shift all dismissed), C's escape of A is *earned for the parallax delta*, and
the single most-likely death of [C] is **N2, not N3** — the surplus may be aleatoric per-tile scale wobble
(cause B in a costume) and the scheduled P1 tests N1, not N2. §5's forward read is updated accordingly.

**Naming note (read once).** There are two different things called "C". The third **root cause** is the
**LOCUS** cause; to avoid collision it is written **L — LOCUS** throughout. The one **live bet** is written
**[C]** (the inter-pano parallax / MoGe-2 direction). A/B/L = the three generative causes; [C] = the
candidate escape.

---

## 한국어 요약 (executive summary)

- **한 줄 판정.** frozen DINOv3(in-domain 강함) × 저데이터/frozen × dense 라는 이 코너에서, 자가지도
  표현학습으로 단일뷰 정확도를 frozen 위로 올린 검증 경로는 **없다**. 전 로스터의 모든 실패는 **단 3개의
  구조적 원인**이 생성한다 — 어느 것도 용량·튜닝·스케일 문제가 아니다.
- **3원인 (+ 1 스코프-라이더).**
  **A — 자기참조/frozen-천장 타깃(SOURCE 끝):** 모든 실패 목적함수의 학습 타깃이 frozen feature의
  함수이거나 모델 자기 출력의 함수다. Data-Processing 부등식 `I(enc;Y) ≤ I(target;Y) ≤ I(frozen;Y)` →
  frozen이 모르는 Y-정보를 **줄 수 없다**. 앵커를 쥐면 정확도 **flat**, 풀면 self-consistency만 남아
  정보를 **버려서 일치**(침식). 이것이 consistency-계열 전부(geo/M1/F3/TC3/pos/scale/yaw/RoPE/④/③)를
  낳는다.
  **B — 광학중심 공유/무-baseline(SINK 끝):** frozen보다 진짜로 풍부한 타깃은 **존재**한다(single_fair
  0.367 < blend_fair 0.455, +0.088). 하지만 그 잉여는 **앙상블 분산감소**라 여러 뷰의 결합분포에만 있고
  단일 주변분포엔 정의상 없다 → 단일뷰 추론 채널이 구조적으로 못 운반. 원인은 하나: intra-pano E2P 타일이
  **한 광학중심을 공유** → 겹침점은 두 뷰에서 **같은 광선**으로 보임 → `f_i=f*+ε_i`에서 `f*`가 모든
  교차뷰 목적함수에서 **소거**, imaging-nuisance ε(경사·리샘플·극/경계 늘어남)만 남음. 삼각측량 불가,
  공유 인코더로 오차 **상관** → 불일치는 *어느 뷰가 틀렸나*가 아니라 *쌍(joint)의 오차*를 예측(Spearman
  0.266) → 역분산 재가중이 **자신만만하게-틀리고-일치하는** 점을 과신 → uniform-mean에 패배.
  **L — 정확도-없는 하강경로(LOCUS/최적화 자리):** 타깃이 무엇이든, 모든 목적함수는 "인코더를 정확도
  방향으로 옮긴다"보다 **저항이 낮은 최소값**을 남긴다 → gradient가 단일뷰 feature를 안 바꾸는 파라미터로
  샌다. 실측 3종: (1) 예측기/헤드 흡수(무제약 1.77M 예측기 → gram 0.001 평탄), (2) stop-grad 앵커 고정
  (타깃=frozen/EMA copy → 증류 최소값이 곧 frozen), (3) 설계상 비인코더 자리(F2/idea1은 인코더 동결,
  fusion·σ만 학습). 병목은 **용량이 아니라 목적함수의 해집합**이다.
  **N5 — task-optimality(스코프-라이더, 4번째 독립원인 아님):** A가 왜 치명적이고(양성이 아니라) 왜
  기하만 열려 있는지의 조건. seg 57.7은 DINOv3가 거의 포화(닫힘) → pos가 죽은 이유; depth/normal/pointmap
  만 headroom(§3.9 depth 유일 4/4). 완전한 승리도 **기하이지 seg 천장이 아니다**.
- **로스터 매핑(요지).** geo·M1·F3·TC3·pos·scale·yaw·RoPE·④closure·③ = **A**. F2·M2·idea1-IV = **B**.
  TermB = **A+B+L의 포스터 차일드**(cause1=예측기흡수 L / cause2=frozen앵커 A / cause3=무시차 B, 한 ID에
  세 원인이 깨끗이 분리). ③ = A+B+L 삼중.
- **승자의 필요조건(음의 공간).** N1(¬A): 타깃이 frozen도 자기출력도 아닌 **외생 X**로 `I(X;task|F)>0`
  — 모든 consistency/equivariance/자기코드/EMA/frozen증류/fused-teacher/set-statistic/disagreement/
  position·scale 탈락. N2(¬B): 잉여가 **단일뷰의 학습가능 함수**여야(체계적·식별가능), 앙상블 복제-잉여
  금지 → 정준 해결책은 **서로 다른 광학중심의 삼각측량**(f\* 소거를 데이터 내부에서 깸). N3(¬L): 인코더
  단일뷰 feature가 손실의 **유일 최소자** — 예측기/σ-헤드/fusion-combiner/디코더-헤드 금지, 인코더가
  움직였는지 **계측**(gram/erank/CKA). N4(¬laundering): 이득은 **실제 학습 디코더 multi-seed**(DPT/UPerNet
  vs frozen-DINOv3-DPT)에서 측정 — probe 이득은 §3.9로 세탁 추정. N5(스코프): **ajar 태스크(geometry)**를
  겨냥, seg 천장은 안 움직임.
- **[C] 정직한 판정.** [C] = MoGe-2(외생 기하 FM) + MP3D/S2D3D 다중뷰 → pseudo-label → DINOv3+LoRA
  증류. **N1을 로스터 최초로 clear**(P0: aligned AbsRel 0.069 > 0.05 → near-GT 아님 → A로 붕괴 안 함).
  그러나 **N2 pending**(P1이 결정하되 현 스펙 불충분: E2P-타일 투영 아티팩트를 시차-교정 가능 분획과
  **분리**해야 하고, 시차-교정 타깃을 단일뷰에서 회귀하는 **두 번째 레그**를 추가해야 함), **N3 미대응**
  (증류 = TermB gram 0.001을 낳은 바로 그 아키텍처 — fat 디코더 헤드 하나면 **조용히 L로 재진입**),
  **N4 미퇴역**. **break-glass:** 이득은 단일뷰 MoGe-2 대비 **시차 델타**여야 함 — 단지 MoGe-2의 더 센
  prior면 그건 "더 센 FM 수입"(framing-A, 앵커 교체)이지 SSL 천장돌파가 아님. **판정: 탈출하도록
  설계됐으나 미정착 — ¬A는 벌었고(N1), sink 미검(N2), locus 무방비(N3).**
- **메타 교훈.** in-domain-강 × 저데이터 × dense 코너에서 SSL이 강한 frozen FM을 정확도로 이기려면
  **새 구조정보를 실제로 주입**해야 한다(consistency로는 절대 불가). 우리 코너를 넘은 논문은 67편 중
  **0편**, frozen을 dense로 이긴 3편(NeCo/DIP/LoRA3D)은 **전부 non-MIM**이며 우리 정확한 regime 밖.
  negative는 결함이 아니라 **결과**다.

---

## 1. TL;DR — the root causes that generate every failure

Three generative causes, plus one scope-rider. Every roster ID reduces to these; none is a capacity,
tuning, or scale problem.

1. **A — SELF-REFERENTIAL / FROZEN-CEILING TARGET (the SOURCE end).** Every failed objective aims its
   training target at a function of the frozen features or of the model's own outputs, so by the Data
   Processing Inequality `I(encoder;Y) ≤ I(target;Y) ≤ I(frozen;Y)` it can carry **no** task-information
   the frozen encoder lacks — forcing the same axis read two ways: hold the anchor and accuracy is
   **flat**; relax it and the only gradient left is self-consistency, whose minimum is reached by
   **discarding** information (blur/quantize to agree) → erosion.

2. **B — SHARED-OPTICAL-CENTER / NO-BASELINE (the SINK end).** A genuinely-richer-than-frozen target
   *does* exist (`single_fair 0.367 < blend_fair 0.455`, the +0.088), but it is **aleatoric
   variance-reduction** that lives only in the multi-view joint and is definitionally absent from any
   single marginal — intra-pano E2P tiles share **one optical center**, so `f*` cancels out of every
   cross-view objective and only the imaging-nuisance residual `ε` remains; two coincident rays cannot
   triangulate, and shared-encoder errors are correlated.

3. **L — ACCURACY-FREE DESCENT PATH (the LOCUS / optimization locus).** Independent of *what* the target
   is, every objective leaves a lower-resistance minimum than "move the encoder toward correctness," so
   gradient drains into a side parameter that satisfies the loss **without changing what a single-view
   feature knows** — predictor/head absorption (TermB gram 0.001 flat), stop-grad anchor pinning, or a
   non-encoder locus by design (F2/idea1 freeze the encoder). The bottleneck is the objective's
   **solution set**, not model capacity.

- **N5 — task-optimality (scope-rider, NOT a fourth cause).** Why A is fatal not benign, and why only
  **geometry** is ajar: DINOv3 is near-saturated on the sealed **semantic** task (seg 57.7 — this is
  why `pos` died: its signal was already inside frozen), and unsaturated only on **depth/normal/
  pointmap** (§3.9 depth is the sole 4/4 positive). No ID is killed by N5 without first going through
  A or B; it is the precondition that decides whether A's headroom is nonzero at all. Even a full win is
  geometry, never the seg ceiling.

---

## 2. Root-cause taxonomy — every roster ID → cause

**Legend.** A = self-referential / frozen-ceiling target (SOURCE). B = shared-optical-center / no-baseline
aleatoric surplus (SINK). L = accuracy-free descent path (LOCUS). "task-optimality" is a **scope-rider on
A** (N5), not a fourth generative cause.

| roster ID | root cause(s) | one-line why | verdict label (verbatim) |
|---|---|---|---|
| **geo** | A | warp-consistency target is a function of F; consistency↑, accuracy flat; obliquity-weight concedes the ε-only residual = `f*` cancelled. | flat (consistency-only) |
| **M1** | A | SwAV codes self-generated → shortcut "blur to agree"; purity **0.838→0.730** = A's erosion; target ⊆ I(frozen;Y). | flat + eroded |
| **F3** | A | Pano-JEPA EMA target is a self-referential copy; purity below frozen (0.821–0.830), erank compressed 39→25; anchor-drift erosion. | flat + eroded |
| **TC3** | A | fixed-teacher-code = frozen anchor held exactly → buys **no-erosion (purity 0.862)** at a HARD frozen ceiling; single_fair Δ **+0.003** = seed noise. | flat (champion, consistency-only) |
| **pos** | A (N5) | pretext target (pitch/FOV) already inside F (frozen decodes **85.9% / 70.2%**, pano-disjoint) → pure readout, target ⊂ frozen = literally zero new info. | **NULL-by-diagnostic** (killed pre-training) |
| **scale** | A | 21.8k panos add MORE self-referential targets, not more truth; **lowest** accuracy of all — scale cannot buy what the channel forbids. | flat (lowest of all) |
| **recipe-yaw/seam** | A | pure equivariance/seam = pure consistency = 0 new info; also **vanishes** under circular-RoPE / circular-padding (double-kill). | **FAIL-provably-dead** |
| **recipe-④closure** | A (Q0/Q1) | gravity/layout/illumination closures are either GT-derived (supervised, out of Q0) or constraints on own outputs = consistency; ④a also empirically settled (E1). | **FAIL-provably-dead** (④a/b) · **FAIL-most-likely-null** (④c) |
| **recipe-④c′** (shape-from-shading) | — (escapes A) | reconstruction of hidden single-view signal under a physical prior ≠ consistency, and no GT (avoids MG-SOG Q0 death) — the ONE non-dead SSL item. | **PARK-exploratory** (behind frozen shading→normal probe) |
| **recipe-RoPE** | A / Q0 | PE/architecture swap re-parameterizes existing info, injects none; re-adds pole content distortion E2P removed. | **OUT-OF-SCOPE-not-ssl** (a live non-SSL architecture bet) |
| **recipe-③** | **A + B + L** | tangent-teacher = fused-frozen = variance-reduction (B, the M2 trap: recon_cos 0.923 → decoded 0.339 ≤ single_fair 0.367); teacher frozen (A); distillation absorbs (L); most-likely §3.9 launders. | **FAIL-most-likely-null** |
| **F2** | **B + L** | learned set-fusion reaches a genuinely-richer set-statistic (clears A) but it is ensemble-aleatoric (B); encoder frozen, gradient in the 1.48M combiner (L); **+0.078 probe → −0.065±0.016 UPerNet, 0/5** = laundered. | lost under real decoder (§3.9) |
| **M2** | B | ensemble→single accuracy transfer; the **+0.088** is multiplicity over overlapping looks, structurally not single-view-recoverable; **no frozen probe can settle it** (bounded by single_fair). | graveyard (back-of-queue, low-EV) |
| **idea1-IV** | B | disagreement predicts **JOINT** pair error (Spearman 0.266), not WHICH view; correlated errors (shared encoder + center) → IV over-trusts confidently-wrong-and-agreeing points; loses to uniform-mean (worst on hi-disagree **+0.0029**). | **KILL** (Gate 0.5, train-free loss) |
| **TermB (whole)** | **A + B + L** | THE POSTER CHILD — cause1 predictor absorption (L, gram 0.001), cause2 frozen-anchor ceiling (A), cause3 intra-pano zero-parallax (B). One ID, all three causes, cleanly separated. | **NULL-NEG** (v1/v2) |
| **[C]** (live bet) | escapes A · PENDING on B · UNGUARDED on L | first target that is exogenous + higher-info (¬A cleared) with a real inter-pano baseline (¬B structurally addressed); but ¬B unproven (P1) and ¬L unaddressed (distillation locus). | **plausibly-escapes-A, sink-untested, locus-unguarded** |

---

## 3. The deep mechanism of each root cause, with evidence

### A — the SOURCE: a target inside `span(F)` cannot carry new Y-information

The information bound is exact: for any target `T` computed from the frozen features `F` (or from the
model's own outputs), a student trained toward `T` obeys `I(encoder;Y) ≤ I(T;Y) ≤ I(F;Y)`. Consistency,
cross-view agreement, self-generated codes (M1's SwAV prototypes), EMA/frozen distillation (F3, TC3),
a fused-frozen teacher (③), and position/scale pretexts (`pos`) are **all** re-encodings of `F` — none
can inject information `F` lacks.

This forces **two observable regimes on one axis**:

- **Keep the anchor dominant** → features pinned to frozen → accuracy **FLAT** (bounded by frozen ±
  seed noise). TC3 is the clean case: frozen anchor held exactly → **purity 0.862, no erosion**, but
  `single_fair Δ = +0.003 ± 0.005` (`SEMANTIC_IDENTITY_SSL.md:613`) = seed noise, not accuracy.
- **Relax the anchor** to "allow restructuring" → the only remaining gradient is self-consistency, whose
  global minimum is reached by **discarding** information (blur/quantize to agree) → semantics **ERODE**.

The **erosion ladder** is A's fingerprint — purity falls monotonically as the anchor weakens, because
the only thing to restructure *toward* inside a closed self-referential system is one's own agreement
(`SEMANTIC_IDENTITY_SSL.md:645-646`):

> TC3 **0.862** > geo 0.854 > frozen **0.838** > F3-EMA 0.830 > F3-student 0.821 > VICReg-distill 0.753
> > M1 **0.730** > VICReg-none 0.728; erank **39 → 25** (`CONSISTENCY_AND_RICHNESS_LIT.md:6`).

"Consistency ≠ accuracy" (the project's iron law 1, confirmed ×8) is a statement about *this* end.

**Laundering is A's empirical DETECTOR, not a separate cause.** A trained decoder is the near-supremum
of readouts of `F`, so any gain whose target lies in `span(F)` is re-derived by the decoder from raw
frozen features, and the controlled SSL-vs-frozen delta collapses to noise. The canonical proof: F-2
learned set-fusion beat uniform-mean **+0.078 under a linear probe** and then **LOST −0.065 ± 0.016
under a real UPerNet, 0/5 seeds** (`SEMANTIC_IDENTITY_SSL.md:440`, §9.7) — with 1.48M *extra* params, so
capacity was not the limiter.

### B — the SINK: `f*`-cancellation makes the surplus non-identifiable per view

The surplus is real and measurable — `single_fair 0.367 < blend_fair 0.455`, a **+0.088** multi-view
surplus (`SEMANTIC_IDENTITY_SSL.md:308`) — but it is aleatoric variance-reduction that lives only in the
joint over views. The one deep geometric cause: intra-pano E2P tiles share **one optical center**, so an
overlap point is seen along the **same ray** by both views:

```
f_i(p) = f*(p) + ε_i ,   with f* IDENTICAL across views.
```

`f*` (the answer) algebraically **cancels** out of every cross-view objective, leaving only the
imaging-nuisance residual `ε` (obliquity, resampling, pole/edge stretch). The project's prize asset —
pixel-exact, parallax-free correspondence — *is* that cancellation condition: exactness = `f*` already
contains the answer = the objective can only see the nuisance.

Two consequences, both measured:

- **No triangulation.** Two coincident rays have no baseline → no depth. "Cross-view completion" reduces
  to distortion-normalization = consistency-class (TermB cause 3).
- **Correlated errors.** The shared encoder makes per-view errors correlated, so cross-view disagreement
  predicts the **JOINT** pair error (Spearman **0.266**, partial 0.261 beyond obliquity — Gate 0,
  `INVVAR_FUSION_LOG.md:72-79`), **not which view is worse**. A confidently-wrong-**and-agreeing** point
  has low disagreement, so inverse-variance reweighting **over-trusts** it and loses to uniform-mean —
  worst exactly where it should help (**+0.0029** on hi-disagree cells, Gate 0.5,
  `INVVAR_FUSION_LOG.md:104`). Uniform beats median AND trimmed too.

**No frozen probe can settle B** — its ceiling *is* `single_fair`: recon_cos **0.923** yet decoded
**0.339 ≤ single_fair 0.367** (`SEMANTIC_IDENTITY_SSL.md:274-276`). The surplus is non-identifiable per
view, hence sink-blocked.

### L — the LOCUS: the encoder is never the unique minimizer

Three verified drains, each satisfying the loss without moving single-view representation:

1. **Predictor/head absorption.** An unconstrained **1.77M** CrossViewPredictor sits between the 0.59M-
   LoRA encoder and the loss and learns the B→A transform, so gradient terminates in the predictor —
   measured: **gram 0.001 flat**, encoder structurally == frozen (`CAN_SSL_RAISE_ACCURACY.md:99, 108`).
2. **Stop-grad anchor pinning.** The target is a `.detach()`'d frozen/EMA feature, so distillation's
   minimum *is* frozen. (This is where L touches A — the anchor is A's information ceiling AND L's descent
   sink; they are separable because L persists even with a **rich** non-frozen target: a rich pseudo-label
   distilled through a decoder head still lets the LoRA encoder sit at frozen+ε.)
3. **Non-encoder locus by design.** F2/idea1 freeze the encoder and learn a fusion-combiner / σ only, so
   gradient never touches representation.

The bottleneck is the objective's **solution set**, not capacity: F2 froze the encoder and gave a 1.48M
combiner and still lost; `scale` gave 21.8k panos to the same geometry and did **worst**; the 0.59M-LoRA
and frozen backbone are **red herrings**. This is the forward danger for any **distillation** bet, because
distillation is the exact architecture that produced gram 0.001.

### The worked example — TermB is the poster child (A + B + L, cleanly separated)

One roster ID exhibits all three causes without overlap, which is why the mechanism is legible here
(`CAN_SSL_RAISE_ACCURACY.md` §4, `PWW_EXPERIMENT_LOG.md`):

- **cause 1 = L (predictor absorption).** The 1.77M predictor learns B→A; the LoRA encoder barely moves
  (gram 0.001 flat). The v1→v2 loss audit *confirmed* weighting was not the bottleneck — the encoder
  simply is not the descent locus.
- **cause 2 = A (frozen-anchor ceiling).** The target is a frozen feature → the student is capped at
  frozen's information; no restructuring toward new semantics is possible.
- **cause 3 = B (intra-pano zero-parallax).** E2P tiles share one optical center → no triangulation →
  "completion" reduces to distortion-normalization = consistency-class.

The completed insight: *this objective has no setting that moves the encoder in the **accuracy**
direction.* Add a predictor and it does not move (L); remove it and completion reduces to
`student_B(warp) → frozen_A`, moving the encoder in the **consistency** direction (= geo) — and
consistency ≠ accuracy. The direction of motion itself is not accuracy.

---

## 4. Necessary conditions for any winner (the negative space)

A candidate must clear **all five**; passing any subset is not sufficient.

- **N1 (clears A / SOURCE).** The new-information term's target must be **provably not a function of the
  frozen features alone nor of the model's own outputs** — an exogenous `X` entering `A(F,X)` such that
  `I(X;task|F) > 0`. This rules out ALL consistency, equivariance, self-generated-code, EMA,
  distill-to-frozen, fused-frozen-teacher, set-statistic, disagreement-confidence, and position/scale
  objectives (every one is data-processing-bounded by `F`). Admissible `X`: a **second FM with a different
  inductive bias**, **real triangulation from displaced optical centers**, or **reconstruction of a hidden
  real signal under a correctness prior** (the LoRA3D / DIP / NeCo class). *External-FM alone is
  insufficient* — if the external target is redundant with `F` it re-hits the frozen ceiling (the RICA
  oracle risk).

- **N2 (clears B / SINK).** The surplus must be a **learnable function of a single view** — systematic and
  identifiable, predictable from one view's content — **not** aleatoric replication-surplus that lives only
  in the multi-view joint. This rules out any "distill the ensemble-average / blend / multi-view-mean into a
  single view" design (the M2 trap). The canonical satisfier is **triangulation from different optical
  centers**: a real baseline breaks `f*`-cancellation *from within the data* (a triangulation term appears)
  and breaks the shared-encoder correlated-error curse, converting aleatoric averaging-surplus into a
  systematic single-view-predictable correctness prior. Averaging/multiplicity alone can **never** satisfy
  N2.

- **N3 (clears L / LOCUS).** The encoder's single-view features must be the **unique minimizer** of the
  loss — no side-module (predictor / σ-head / projector / fusion-combiner / decoder head) and no
  information-discarding move may lower the loss to the same floor. Operationally: (a) no unconstrained
  residual predictor or high-capacity head between the LoRA encoder and the target — use a
  **starved/linear/frozen read-out** so the only path down runs THROUGH the encoder; (b) the accuracy
  target must not equal a frozen/EMA copy of the same encoder's output (overlaps N1's anchor requirement);
  (c) **instrument** that the encoder moved in the accuracy direction (gram / erank / CKA-vs-frozen on the
  LoRA path), not merely that a downstream metric rose. This is the gate a distillation setup is most likely
  to fail **silently** (TermB gram 0.001).

- **N4 (survives Q3 / anti-laundering).** The gain must be measured as the controlled delta `A(F,X)` vs `F`
  under the **same real trained multi-seed decoder** (DPT/UPerNet vs frozen-DINOv3-DPT), the only
  comparison that quotients out the decoder's ability to re-derive everything already in `F`. A probe-level
  gap is presumed laundered by construction (§3.9). **Passing N1–N3 does not imply N4:** a delta can be
  real, single-view, and encoder-carried yet still shrink to noise if a strong decoder on raw frozen
  features already re-derives it.

- **N5 (scope-rider, from task-optimality).** Aim at the **ajar** task — depth/normal/pointmap where DINOv3
  is unsaturated (§3.9 depth is the sole 4/4 positive; MoGe headroom exists) — NOT the **sealed** semantic
  task (seg 57.7, DINOv3 near-optimal; `pos` died because its signal was already in frozen). Not a fourth
  independent cause; it is the precondition that decides whether N1's headroom is nonzero at all. Even a
  full win is geometry, never the seg ceiling.

---

## 5. Honest forward read on [C] — does it escape, or re-enter?

**[C] = inter-pano PARALLAX self-calibration (LoRA3D-class):** MoGe-2 (an external metric-geometry FM) +
MP3D/S2D3D multi-view RGB → pseudo-labels → distill into DINOv3+LoRA. Its claim to be different from the
graveyard: **real triangulation information** (new information, not variance-reduction) + an **external FM**
(MoGe-2), staying pano-ssl-dense (the RICA external-oracle design), **not** LoRA-on-MoGe-2. Honest framing
from its own log: **[C] = framing-A ("import MoGe-2") + an unverified parallax bet**; the parallax delta is
[C]'s only novel content over A.

[C] does **not yet** satisfy all necessary conditions. It is uniquely **structurally positioned** to escape
(no prior roster item could), but only **N1 is cleared now**.

- **N1 (SOURCE) — CLEARED, and this is earned, first in the roster.** The target is MoGe-2 (genuinely
  exogenous, different inductive bias, metric structure NOT a function of frozen DINOv3) PLUS an inter-pano
  triangulation constraint from displaced optical centers. **P0** (2026-07-09, `MOGE_PARALLAX_LOG.md:58-66`)
  gives aligned δ<1.25 = **0.93** and aligned AbsRel **0.069** (> 0.05 → real headroom, not near-GT → does
  not collapse to plain-A), far above the linear probe's ~0.13–0.19 log-error. So `I(X;task|F) > 0` is real
  — the first target that clears the source end without being frozen-or-self, the cause that killed the
  entire consistency family.
  **Break-glass caveat (attached, not omitted):** the win must be the **parallax DELTA over single-view
  MoGe-2**, not merely MoGe-2's stronger frozen prior — the latter is "import a stronger FM" (framing-A, a
  swapped anchor), not an SSL ceiling-break.

- **N2 (SINK) — PENDING; the discriminator is P1, and P1 as-specced is not yet sufficient.** Inter-pano =
  different optical centers = a real baseline = `f*` no longer cancels, which structurally addresses the
  exact curse that sank geo / TermB-cause3 / idea1-IV. But whether the parallax-correctable error is
  **systematic w.r.t. single-view content** (→ learnable/distillable, N2 cleared) or **aleatoric per-tile
  jitter** (→ N2-blocked, M2 in geometry clothes) is **unproven**. P0 found ~47% of MoGe's error is per-tile
  scale drift (parallax-favorable in *kind*, LoRA3D's canonical lever) — but P0 itself flags the confound
  that some of that drift is an **E2P-tile-vs-pinhole projection artifact**, not a depth ambiguity parallax
  can fix (`MOGE_PARALLAX_LOG.md:81-82`). **P1 must be GT-referenced AND separate the tiling artifact from
  the parallax-fixable fraction**; if the 47% is mostly tiling, the delta is thin and [C] collapses toward A.
  Additional demand P1 as-written omits: it measures **SOURCE headroom**, not **SINK deliverability** — it
  must add a **second leg regressing the parallax-CORRECTED target from single-view features on held-out
  tiles**.

- **N3 (LOCUS) — UNADDRESSED; this is the blind spot.** [C] is a **distillation** setup (pseudo-label →
  DINOv3+LoRA) — **exactly** the architecture that produced TermB's gram 0.001 flat. P1 tests A-headroom
  and B-fraction; P1 does **not** test L. The moment any capacity sits between the LoRA encoder and the
  pseudo-label target (a depth/pointmap decoder head, a scale-alignment layer, a σ head), gradient can
  drain there and the 0.59M LoRA stays at frozen+ε — reproducing gram 0.001 with a NEW target — and P1
  could "pass" while the win lives in a decoder head, then launder under §3.9 exactly like F2. [C]'s own
  doc flags the Q0-GT-leak and correlated-error traps but does **not** flag the locus trap, and has not
  pre-registered how it forces the encoder (not a head) to be the unique carrier, nor how it verifies the
  encoder moved.
  *(Provenance: the independent critic pass returned empty; this "most-likely-[C]-failure / re-entry" read
  is sourced from N3 and the C-verdict line — [C] is "one architectural decision (a fat decoder head) away
  from silently re-entering the LOCUS cause" — not from an invented critic.)*

- **N4 (LAUNDERING) — UNRETIRED.** Even a real parallax-corrected pseudo-label must beat frozen under a
  trained DPT multi-seed vs frozen-DINOv3-DPT, not a probe; `MOGE_PARALLAX_LOG.md`'s own "honest EV chain
  (sobering)" flags this final attenuation step as untested.

- **N5 (SCOPE) — HELD by construction.** [C] targets geometry (the ajar box) and concedes seg 57.7 does not
  move.

**Verdict on [C].** It is the **only** idea in the roster that CAN satisfy the winner condition (exogenous
`X` with `I(X;task|F) > 0` AND a real baseline that could make the surplus single-view-systematic). It has
**earned the ¬A escape**. But "[C] escapes" is **not yet earned in full** — the honest label is
**plausibly-escapes-A, sink-untested, locus-unguarded**. The decisive path is: **(i)** P1 GT-referenced with
the tiling-artifact separated AND a single-view regression leg (settles N2); **then (ii)** a pre-registered
LOCUS GUARD — distill into the encoder with a starved/linear/frozen read-out and instrument LoRA-moved-off-
frozen via gram/CKA (settles N3); **then (iii)** the Q3 real-decoder delta vs frozen-DINOv3-DPT multi-seed
(settles N4). Until P1 clears with the tiling confound removed, [C] is architected to escape, empirically
unsettled, and **one architectural decision (a fat decoder head) away from silently re-entering the LOCUS
cause.**

---

## 6. Meta-lesson — SSL-for-accuracy in the in-domain-strong × low-data × dense corner

Our corner is the hard box: an **in-domain-strong** backbone × **low-data/frozen** × **dense**. The
literature says so quantitatively — **0 of 67** papers cleared this exact corner, and the only 3 that beat a
strong frozen FM on dense accuracy (NeCo, DIP, LoRA3D) are **none-MIM** and off our exact regime
(`SSL_SUCCESS_CASES_LIT.md`). The project's eight independent negatives say the same thing structurally:
when the backbone is already strong in-domain, a self-supervised signal that is **pure consistency injects
zero new information** (A), a signal that is **multi-view-only variance-reduction is non-identifiable per
view** (B), and even a rich signal drains into a **non-encoder locus** unless the encoder is forced to be
the unique minimizer (L). The single generative lesson: in this corner, SSL can raise **consistency** and
**fusion-ceiling** cheaply (TC3's home ground) but can raise **accuracy** only by importing **real new
structure** — an exogenous FM with a different bias, or genuine triangulation from a real baseline — and
*then* surviving the LOCUS and laundering gates that this project has watched swallow every prior gain. The
honest state of the art here is a **rigorous negative with a single structurally-distinct live bet ([C])**,
not a solved problem — and in this project, a negative is not a defect, it is the result.

---

## 7. Independent adversarial critic (re-run, 2026-07-09) — ACCEPT-WITH-RESERVATIONS

The in-workflow critic stalled; this is the genuine independent pass (oh-my-claudecode:critic, Opus). It
**accepts the 3-cause (A/B/L) + N1–N5 structure** and adds four sharpenings:

- **Misfit failures — NONE.** `[scale]`'s "accuracy LOWEST" is single-seed within self-declared noise, and
  it was **transductively advantaged yet still lost** → *reinforces* A (more label-free data cannot exceed
  the frozen ceiling), not a new cause. `[TC3]`-flat vs `[M1]`-eroded is **one anchor-strength axis** (hold
  → flat, release → erode), not two mechanisms mislabeled as one.
- **Missing 4th root cause — DISMISSED (all three candidates).** (a) **Capacity refuted**: the *same* 0.59M
  adapter ties SGFormer SOTA (0.104) when trained **supervised**, so `gram 0.001` is capacity **unspent**
  (= cause L), not capacity insufficient. (b) **Eval-validity** is real but it **is N4** — it explains the
  false *positives* (F2 §9.6, recipe-③, TC3-blend), not the failures. (c) **Distribution-shift refuted** by
  the transductive `[scale]` result. So A/B/L is complete; capacity and the boundary-triad are *not* 4th
  causes.
- **[C] escapes A — EARNED, but only for the parallax delta.** `I(delta;Y|DINOv3) > 0` because the
  cross-view triangulation delta is **exogenous geometry, not an encoder feature-property** — so the fact
  that MoGe-2's own encoder is a DINOv2 ViT-L does **not** collapse the escape; the DINOv2-backbone concern
  only reaches the **already-conceded "import-MoGe" half** (the monocular prior, which *may* be redundant
  with DINOv3 — that half is framing-A, a swapped anchor). The SSL-ceiling-break lives entirely in the
  parallax delta.
- **Most-likely death of [C] = N2, NOT N3.** The decisive risk is that the ~47% per-tile scale drift is
  **aleatoric per-tile scale wobble a single view cannot recover** — i.e. **cause B wearing a geometry
  costume** (M2 again). And the scheduled **P1 tests N1 (source headroom), not N2 (single-view
  learnability)** → **false-greenlight risk**. (N3/LOCUS remains a real *second* kill-path, per §5, but N2
  is the *first* and most probable.)

**Binding correction (actionable).** Before any [C] build: **bind P1 to an explicit N2 discriminator** — on
held-out tiles, regress the *parallax-corrected* target from *single-view* features and measure whether one
view predicts the correction (systematic → N2 clears; unpredictable → N2 fails, [C] = B-in-costume, stop).
Keep the N3 LOCUS-guard (starved read-out + gram/CKA instrumentation) pre-registered as the second gate.
The one reservation blocking a clean ACCEPT is exactly this: **[C]'s decisive condition N2 is untested by
its currently-scheduled experiment — cheaply fixable pre-build.**

---

## 8. The two-axes probe — DEPTH is AXIS-2, empirically (2026-07-09)

The information-theoretic refinement of the wall: raising accuracy needs either (axis 1) info F genuinely
LACKS, or (axis 2) info F HAS that a weak decoder misses. Only axis-1 survives a strong decoder (§3.9); axis-2
"gains" launder. The project measured **depth/normal with a LINEAR probe** (0.13–0.19 log-err), so it never
separated the two — the "geometry headroom" could be an encoder gap (axis-1, SSL-addressable) OR a decoder gap
(axis-2, not).

`scripts/diag_axis12.py` settles it for DEPTH (frozen DINOv3, S2D3D area_1, pano-disjoint, GT depth.png/512,
32×32 grid):

| depth head on **frozen DINOv3** | AbsRel ↓ | δ<1.25 ↑ |
|---|---|---|
| linear (the project's probe) | 0.165 | 0.782 |
| MLP | 0.119 | 0.881 |
| **conv (mini-decoder)** | **0.111** | 0.883 |
| MoGe-2 (zero-shot reference) | 0.133 | — |

**Verdict: AXIS-2 for depth.** A strong conv decoder on **frozen** DINOv3 reaches **0.111** — −33% over the
linear probe and *below* zero-shot MoGe (0.133). So the linear-probe depth weakness was **extractability
(decoder), not missing information (encoder)**: **F already contains the geometry to 0.111.** (Confound: the
conv head is in-domain supervised while MoGe is zero-shot — so this is NOT "we beat MoGe"; it is proof that F
*contains* depth info a strong decoder extracts. The axis-2 conclusion does not depend on the MoGe comparison.)

**Unified consequence (closes the per-tile-accuracy encoder question):**
- **seg = axis-1-SATURATED** — F is in-domain-strong, already strong-decoded to 57.7; no new info to add (A).
- **depth/normal = axis-2** — F HAS the info; a decoder is the lever, not the encoder.
- ⇒ **encoder-side SSL has structurally little room to raise per-tile ACCURACY on either task family.** Its
  real, confirmed value is cross-tile **coherence/robustness** (TC3), not per-tile accuracy.
- ⇒ **This weakens [C]/(b)/(c) as ENCODER-injection for depth:** distilling a physical prior or parallax
  geometry INTO the encoder duplicates what a decoder already extracts from F (and would launder, §3.9). The
  honest depth lever is a **stronger decoder** (frozen F + conv already ≈/> MoGe), which is a decoder project,
  not encoder-SSL. The only surviving encoder-SSL accuracy path would require a task where F is *empirically
  axis-1* (a strong decoder on F still fails) AND an exogenous N1 source supplies it — neither seg nor depth
  qualifies. `diag_axis12.py` is the gate to test any future candidate task.