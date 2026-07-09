# Gate Verdict — Panorama-adaptation recipe (ERP spherical-RoPE + cross-projection distillation) vs the SSL-for-accuracy negative

**Status:** Gate ruling. Champion unchanged: frozen DINOv3 ViT-B/16 + E2P tangent tiles + LoRA(0.59M) + fusion + decoder → Stanford2D3D seg fold-1 = **57.7 mIoU** (single-tile), overlap-blend = **0.611**, single-decode scatter-mean field @64×128 = **0.557** (`docs/RESULTS.md` §3.8).

**Provenance:** Recipe is a literature-grounded proposal of unknown origin (likely another model). All 8 cited papers are post-2026-01 cutoff → independently verified on arxiv/proceedings (results below). This ruling reuses the honesty-first, red-teamed house style of `docs/SSL_ACCURACY_DESIGN.md` / `docs/CAN_SSL_RAISE_ACCURACY.md`. Every empirical anchor cited here was re-confirmed against source this session.

---

## 한국어 요약 (executive summary)

- **핵심 판정: 이 레시피는 세션의 SSL-정확도 부정을 탈출하지 못한다.** 레시피가 내세운 "정확도 레버" = **손실 ③(i) 교차-투영 증류**(교사 = tangent-tile 앙상블 frozen DINOv3, cos-lat 가중 → 학생 = ERP 단일패스 증류)는 **새 실험 없이 증명 가능한 두 개의 죽은 길**로 재진입한다: (1) **frozen-anchor 정보 천장** (교사가 frozen이므로 학생은 frozen의 정보 상한에 갇힘 → Q1 실패 = Term B 구조원인 #2, `CAN_SSL` §3-4), (2) **§3.9 세탁** (F-2 학습융합이 linear probe에서 +0.078 이겼다가 실제 UPerNet 5-seed에서 **−0.065±0.016, 0/5** 패배, `SEMANTIC_IDENTITY_SSL.md:440`).
- **③(i)는 프로젝트 그레이브야드 항목 그 자체다.** `SSL_ACCURACY_DESIGN.md:333` "M2 / ensemble→single accuracy transfer — plausibly-flat, low-EV, back-of-queue"를 **ERP를 학생 입력단위로 리스킨**한 것. "입력단위 문제지 구조적 불가능이 아니다"라는 탈출 논리는 §8이 반박: single→blend feature-regression **recon_cos=0.923**임에도 decode는 **0.339 ≤ single_fair=0.367** (blend_fair=0.455). ERP 순전파는 각 표면점을 **한 번** 본다(=coverage/global-context) — +0.088 blend 이득을 만든 **multiplicity(중복 관측 평균)**를 재현하지 못한다.
- **정직한 라벨: FAIL-most-likely-null (증명된-죽음 아님).** 게이트 자신의 Q2 규칙상 **어떤 frozen 프로브도 M2를 settle 못 한다**(single_fair 상한). "ERP-native로 **학습된** 학생 + global context"이라는 특정 주름은 실제로 미검증이다 — 하지만 그것을 정하는 것이 바로 저-EV 고비용 빌드다. "구조적으로 불가능"은 과장이 될 것이다.
- **회전-등변/이음새 손실은 진짜로 죽었다 (FAIL-provably-dead).** yaw-equivariance f(roll(x))=roll(f(x))는 순수 consistency = iron law 1 (새 정보 0), 게다가 레시피 스스로 circular RoPE 하에서 손실이 **소멸**한다고 인정 → 이중 사망. seam 손실도 iron law 1 + circular padding이 이미 해결(잉여) → 이중 사망.
- **④ closure 계열:** gravity-axis(중력축) = MG-SOG 렌더-pitch 메타데이터 Q1 락 + E1 이미 실측 사망. closed-layout = GT branch Q0 실패 / regularizer branch Q1 실패, 실내 전용. single-illumination 평면적 읽기 = consistency-class Q1 실패. **유일한 생존 후보 = ④c′ shape-from-shading**(단일광 물리 prior 하 shading→normal 복원) = 재구성(consistency 아님)이자 GT 없음(MG-SOG Q0 죽음 회피) → **PARK**(frozen shading→normal 프로브로 게이팅 후에만).
- **인용 판정:** 8편 전부 실존(조작 0). 그러나 **결정적 필터**(강한 frozen을 저-데이터에서 dense 정확도로 SSL 목적함수로 이긴다)를 **통과한 논문 0편**. 헤드라인 레버 ③의 DiT360 인용은 **오용**(생성-품질 정규화기이지 융합-수확 증류가 아님).
- **유일한 미검증 생존자 = ERP-native spherical-RoPE 토큰화** — 그러나 이것은 **아키텍처/토큰화 변경(Q0 범위 밖)**이지 **SSL 목적함수가 아니다.** SSL-정확도 주장을 구제하지 못한다. FishRoPE(2604.10391, 실존)가 이 정확한 스택(frozen DINOv2+LoRA+투영 RoPE)에서 실제 정확도 이득(+3.7 mIoU vs 2D-RoPE)을 보고하나, **저-데이터 없음 + position-geometry 레버(Q0-제외)**로 필터 실패. 별개의(왜곡/침식) 트랙 실험으로만 가치.
- **열린 레버 지도 대비:** 레시피는 **두 열린 레버(inter-pano 시차 완성 · 역분산/Kalman 융합) 어느 쪽도 건드리지 않고** seg/fusion/consistency를 재답습한다. cos-lat **면적** 가중은 불확실성(inverse-variance) 가중이 아니다.
- **공학 비용:** "PE 모듈 + 교사 루프"가 **아니다.** 융합 tangent-teacher는 **존재하지 않고**(encoder.teacher()는 frozen-self), ERP-native 학습은 새 파이프라인, multi-seed 실디코더 de-risk는 필수.

---

## 1. TL;DR verdict — does the recipe escape the negative?

**No — not on the SSL-for-accuracy axis.** The recipe's *headline accuracy lever* (loss ③(i), cross-projection distillation) **re-enters two already-proven dead ends that require no new experiment to establish:**

1. **Q1 — frozen-anchor information ceiling.** The teacher is frozen DINOv3 features fused over tiles. Distillation caps the student at the teacher's information content; there is no external signal (no 2nd FM, no simulator, no physical prior) and no reconstruction of real hidden signal. This is **Term B structural cause #2 verbatim** (`docs/CAN_SSL_RAISE_ACCURACY.md` §3-4): a frozen distillation target caps the encoder at frozen's information upper bound. "Multi-view *fused* frozen" is still bounded by frozen's ceiling — fusing frozen features injects no information a single-view frozen forward could not in principle carry. **Auto-fail: zero new single-view-accessible information.**
2. **Q3 — §3.9 fusion/consistency laundering.** This is a fusion-class gain, and strong decoders launder exactly this class. Canonical proof (`docs/SEMANTIC_IDENTITY_SSL.md:440`, §9.7 de-risk): learned set-fusion F-2 (+1.48M params) beat uniform-mean **+0.078 under a linear probe**, then **LOST −0.065±0.016 under a real UPerNet, 0/5 seeds** (p=0.062); depth also worse. A distilled ERP-student probe-gain, if any appeared, would be expected to launder identically.

Both are provable **now**, from durable evidence — that is what "re-entry into two already-proven dead ends, provable without new experiments" means.

**But the honest verdict label for ③(i) is `FAIL-most-likely-null`, NOT `provably-dead`.** By the gate's own Q2 rule (`SSL_ACCURACY_DESIGN.md:82-83`) **a frozen probe cannot settle M2** — it is bounded by `single_fair`. ③(i) IS the project graveyard entry "M2 / ensemble→single accuracy transfer" (`SSL_ACCURACY_DESIGN.md:333`) re-skinned with ERP as the student input-unit. The "input-unit not structural" escape is strongly distrusted by §8's single→blend regression (recon_cos=0.923 yet decoded **0.339 ≤ single_fair=0.367**; an ERP forward adds *coverage*, not the *multiplicity* over overlapping looks that produced the +0.088 blend gain) — but the specific *ERP-native TRAINED student + global context* wrinkle is genuinely untested. Calling it "structurally impossible" would **overclaim**. The honest read: **no citation support + distrusted by three iron laws + not yet empirically settled ⇒ most-likely-null, low-EV to settle.**

**The recipe's ONLY genuinely-untested survivor is ERP-native spherical-RoPE tokenization — and it is an ARCHITECTURE / tokenization change, NOT an SSL objective.** Per Q0, a position-encoding swap re-parameterizes existing information (LoRA heals statistics, injects nothing) — it is Q0-*out-of-scope-as-SSL* (the exact MG-SOG exclusion logic, and the reason FishRoPE's PE-geometry lever is Q0-excluded). It therefore **cannot rescue the SSL-for-accuracy claim.** It is a legitimate but *different* project — a distortion/erosion-reduction (TC3-adjacent) architecture bet with modest feasibility support, evaluated and scored **as architecture, never as SSL.**

**The recipe's core framing error:** it fuses a small coherent PE swap and a low-EV M2 re-tread into one "accuracy lever." They are two different things; only the SSL half (③i) touches the SSL-accuracy question, where it re-enters the honest negative.

---

## 2. Citation-verification verdict (the discriminating result)

**All 8 cited papers are VERIFIED REAL — zero fabrications, zero not-found, zero misattributed IDs.** The discriminating findings:

- **Two independent discriminators against the recipe:**
  - **(a) The headline accuracy lever has ZERO citation support.** Loss ③(i) "cross-projection distillation" (tangent-ensemble teacher → ERP single-pass student) cites **DiT360's cube loss** — but DiT360 (2510.11712, verified) is a panorama **generation** model whose cube/yaw losses are distortion-aware **generation-quality regularizers**. There is **no ensemble-to-single-view fusion-accuracy distillation anywhere** in DiT360. **This citation is a MISUSE**; the fusion-harvest mechanism is uncited. (This is the load-bearing citation finding.)
  - **(b) The one on-recipe paper fails the decisive filter.** FishRoPE (2604.10391, verified, claim MATCHES exactly) runs the recipe's architecture half — frozen DINOv2-B + LoRA(r=16, q/v) + projective spherical RoPE — and reports (per the paper) a real dense-accuracy gain **+3.7 mIoU / +1.7 mAP over 2D-RoPE**, edging a concurrent DINOv2-L FM. But it fails the decisive filter on **two axes**: no low-data regime (standard full-dataset splits), and the lever is **position-encoding geometry = a Q0-excluded architecture change**, not an SSL objective.

- **No cited paper clears the decisive filter** (*beat a STRONG FROZEN foundation on DENSE ACCURACY in LOW-DATA via an SSL objective*):

| Paper (arXiv) | Real? | Claim matches | Filter class → verdict |
|---|---|---|---|
| PanoVGGT (2603.17571) | ✅ real | matches | geometry/reconstruction feasibility (VGGT-style 3D + dataset) → **fails** |
| FishRoPE (2604.10391) | ✅ real | matches | full-data + position-geometry (Q0-excluded) → **fails filter, supports ARCH only** |
| SphereViT (component of DA² 2509.26618) | ✅ real | matches (loose) | distortion-aware PE inside a from-scratch depth model → **fails** |
| DiT360 (2510.11712) | ✅ real | **partial (MISUSE)** | generation-quality regularizer, no fusion-harvest → **does not support ③** |
| SHERPA (2606.12213) | ✅ real | matches | yaw-consistency SSL for **generation** = iron law 1 class → **fails** |
| PanoFormer (2203.09283) | ✅ real | matches | from-scratch supervised depth arch → **fails** |
| Bending Reality / Trans4PASS (2203.01452) | ✅ real | matches | supervised UDA + from-scratch DPE → **fails** |
| Pano-Scene-Analysis survey (2606.27745) | ✅ real | matches | survey, no experiments → **N/A** |

**Net:** the SSL accuracy lever (③) is unsupported by any citation and mechanically distrusted by three iron laws. The only thing that *moves* is the ARCHITECTURE experiment (RoPE swap / ERP-native), which gets modest feasibility support — as a **distortion/geometry lever, not an SSL accuracy lever.**

---

## 3. Per-item gate table

The recipe reuses the numbering ①②③④ for **both** tokenization and the loss stack. Rows are labeled descriptively to disambiguate; ③ is split into SSL-distillation (i) vs RoPE-encoding (ii), and ④ into a/b/c-plain/c′-steelman.

| Item | SSL objective? | Q0 | Q1 | Q2 | Q3 | Verdict | Iron-law / structural cause hit |
|---|---|---|---|---|---|---|---|
| **Loss ③(i) — Cross-projection DISTILLATION** (tangent-ensemble frozen teacher → ERP single-pass student; the headline "accuracy lever") | **YES** | **PASS** (distill from another model's features, no labels — an explicit Q0-passing form) | **FAIL** — teacher frozen ⇒ caps student at frozen's info; no external signal, no hidden-signal reconstruction | **FAIL-flavored but NOT frozen-probe-settleable** — equivocates multiplicity (Q2/M2) as new info (Q1); ERP forward sees each point ONCE (coverage ≠ multiplicity) | **MOST-LIKELY LAUNDERS** — fusion-class; §3.9 F-2 −0.065±0.016, 0/5 seeds | **FAIL-most-likely-null** | Q1 **frozen-anchor ceiling** (Term B cause #2); Q2 **M2 trap / iron law 2** (§8 recon 0.923→0.339≤0.367); Q3 **§3.9 laundering**; = graveyard "M2 ensemble→single" re-skinned |
| **Tokenization-① / ERP-native spherical-RoPE ENCODING** (swap 2D axial RoPE → circular-yaw / lat-corrected-pitch; ERP full-sphere input; LoRA heals shift) | **NO** | **OUT-OF-SCOPE-as-SSL** — architecture/PE swap re-parameterizes existing info (MG-SOG exclusion logic); LoRA injects nothing | N/A as SSL lever (no new-info term). *Compounding:* RoPE fixes POSITION not CONTENT — raw-ERP→planar 16×16 patch-embed reintroduces the pole/latitude content-stretch E2P tiles were built to remove (erp_direct already < E2P in-repo) | N/A | N/A as SSL. *(FishRoPE = real full-data ARCH gain, but Q0-excluded, no low-data)* | **OUT-OF-SCOPE-not-ssl** (a live NON-SSL architecture bet, ≠ FAIL) | Q0 architecture exclusion; NOT an SSL claim — belongs in the distortion/erosion track |
| **Tokenization-② — HEALPix equal-area cell resampling** (`healpix_seg.py` "heads this way") | NO | OUT-OF-SCOPE-as-SSL | N/A | N/A | N/A | **OUT-OF-SCOPE-not-ssl; not-yet-built** | Architecture; `healpix_seg.py` is a mosaic linear-probe **eval** script, not a per-cell gnomonic tokenizer — "heads this way" overstates it |
| **Tokenization-③ — Deformable patch embed** (PanoFormer/Trans4PASS DPE; recipe says NOT recommended) | NO | OUT-OF-SCOPE-as-SSL | N/A | N/A | N/A | **OUT-OF-SCOPE; recipe self-excludes** | From-scratch surgery, bad fit with frozen transplant (recipe agrees) |
| **Loss ① — Yaw-equivariance** f(roll(x))=roll(f(x)) | **YES** | PASS (ERP roll = exact label-free symmetry) | **FAIL** — pure consistency/invariance = zero new info | N/A (dead at Q1) | N/A | **FAIL-provably-dead** | **iron law 1** (consistency≠accuracy, ×8); = graveyard `warp_equivariance_loss` dead-invariance form; **DOUBLE-KILL:** recipe concedes loss **VANISHES** under circular RoPE |
| **Loss ② — Seam / boundary-continuity** | **YES** | PASS (wrap-continuity = label-free) | **FAIL** — cross-position agreement = zero new info | N/A | N/A | **FAIL-provably-dead** | **iron law 1**; **NULL-BY-REDUNDANCY:** circular padding already enforces continuity at token level → no residual gradient |
| **Loss ④a — Gravity-axis consistency** | YES | PASS (gravity = capture metadata, not GT) | **FAIL** — consistency variant = 0 info; predict variant = fixed render-pitch metadata (MG-SOG Q1 lock); **E1 already killed it** (frozen encodes pitch/FOV) | N/A | N/A | **FAIL-provably-dead** | iron law 1 + MG-SOG metadata Q1 lock + empirically settled (E1) |
| **Loss ④b — Closed-layout** | YES | branch(i) GT-target = **FAIL** (MG-SOG Q0); branch(ii) regularizer = PASS | branch(i) barred at Q0; branch(ii) hypothesis-space constraint on own outputs = 0 external info → **FAIL** | N/A | N/A | **FAIL-provably-dead** | fork, both tines closed: Q0-GT / iron-law-1-regularizer; indoor-only scope |
| **Loss ④c — Single-illumination (PLAIN reading)** | YES | PASS | **FAIL** — one-light consistency on own predictions = 0 external info | N/A | N/A | **FAIL-most-likely-null** | iron law 1; recipe's own weakest-form ("weak precedent, not formalized") |
| **Loss ④c′ — Shape-from-shading (STEELMAN)** — recover normals from shading under single-light physical prior | YES | **PASS cleanly** (no GT, no simulator — beats MG-SOG's one disqualifier) | **PASS (candidate)** — reconstruction of real hidden single-view signal under a physical/correctness prior (Q1-passing class) | Plausibly PASS (single-view-native) — but **frozen probe cannot settle** | **UNRESOLVED** — faces §3.9 laundering (a DPT normal-decoder may re-derive shading→normals) | **PARK-exploratory** | The ONE item that does NOT re-enter any lock — escapes iron law 1 (reconstruction≠consistency) AND MG-SOG Q0 death; gate behind frozen shading→normal probe |

---

## 4. Honest reconciliation — measured against the map of what's LEFT

The project's map of genuinely-open levers (`docs/CAN_SSL_RAISE_ACCURACY.md` §6-7) has exactly two entries:

1. **Inter-pano PARALLAX completion** — the real triangulation information intra-pano E2P tiles structurally lack (they share one optical center → zero parallax). Needs MP3D (not on disk) + a mini-DUSt3R-scale build.
2. **Inverse-variance / Kalman fusion** for depth/pointmap — the principled harvest of the fusion ceiling (uncertainty weighting), unexplored.

**The recipe engages NEITHER open lever and re-treads closed ground:**

- **③(i) is intra-pano ZERO-PARALLAX** — it does not touch inter-pano parallax completion (open lever #1). Its teacher is a fused set of tiles sharing one optical center; "cross-view completion" here reduces to distortion-normalization = consistency-class (Term B structural cause #3).
- **cos-latitude AREA weighting is deterministic geometry, NOT inverse-variance / Kalman fusion** (open lever #2, which is uncertainty weighting). The recipe does not engage the Kalman lever even though it operates in the fusion family.
- The fusion ceiling ③ tries to harvest is **already harvested cheaply**: uniform-mean scatter → single-decode = **0.557 @64×128** (`RESULTS.md` §3.8, near per-tile pooled ~0.58). Uniform-mean is already the field-fusion champion; the "extra info" the fused teacher carries IS the already-harvested M2/blend quantity (raised to blend_fair 0.473 by the TC3 adapter, `SEMANTIC_IDENTITY_SSL.md:587`), not a new lever.
- Loss ①/② and ④a/b/c are **seg/fusion/consistency re-treads** — pure agreement objectives (iron law 1) or GT/metadata (Q0/Q1 locks).

**Net:** on the SSL-for-accuracy axis the recipe is a re-tread. The only member that lands *off* the closed map is ④c′ shape-from-shading (a new candidate, not a mapped lever), and it is parked, not greenlit.

---

## 5. What survives as worth-one-experiment — cleanly separated by track

### SSL-for-accuracy track → **HONEST-NEGATIVE #2**

Nothing on the recipe's SSL axis survives as worth an experiment *over the re-tread it duplicates*. ③(i) is the graveyard's "M2 / ensemble→single" entry re-skinned; the losses are provably dead; ④a/b/c die at Q0/Q1. The one SSL item that does not die is **④c′ shape-from-shading**, and it **PARKS behind two gates, not greenlit** (and note: it is a *steelman reformulation* of the recipe's weakest, least-formalized item — not what the recipe actually proposed):

- **Gate 1 (frozen, minutes, kill-cheap):** linear-probe frozen DINOv3 patch feat → surface normal / shading, E1-style (`diag_position_headroom.py` pattern). If frozen **already** recovers it → Q1 null → **dead** (same fate as the pitch/FOV pretext). Only if frozen *cannot* → real headroom.
- **Gate 2 (only if Gate 1 shows headroom):** laundering-proof multi-seed real-DPT normal eval (Q3). A probe-only gap is discarded.
- **Honest EV:** exploratory. It is the self-supervised twin of MG-SOG's "genuinely external, single-view, non-consistency" class, minus the GT problem — the one honest SSL bet the recipe surfaces, but it is a reformulation and gated, not a recommendation to build now.

**So the SSL-for-accuracy verdict is: the recipe does not move it. Record as honest-negative #2.**

### NON-SSL architecture track (clearly labelled — this is NOT an SSL experiment)

If the user wants to pursue **tokenization** (a legitimate but different project — distortion/erosion, not SSL-accuracy), the single cheapest kill-gate:

- **Near-free PRE-CHECK first (a few GPU-min):** push the **existing** `erp_direct` path (planar RoPE, no swap) through the **same** UPerNet used in §3.9. If that planar-ERP field sits far below 0.557, it bounds the hole position-only spherical RoPE must climb before adding *any* value over tiles — a chance to **pre-kill before building the swap.**
- **TREATMENT (architecture only — no fused teacher, no ② rebuild):** frozen DINOv3 + spherical-RoPE swap (replace `get_patches_center_coordinates` + yaw-angle construction with circular-periodic-yaw / latitude-corrected-pitch) + LoRA heal **with `k_proj` added to `LORA_TARGETS`** (RoPE rotates BOTH q and k, but baseline `qv`-LoRA leaves k_proj un-adapted on the frozen path a PE-basis change most disturbs — code-audit flag). ERP-native forward → scatter ERP patch features into ONE 64×128 field → decode ONCE.
- **CORRECT BASELINE (do not collapse it):** the **uniform scatter-mean field = 0.557 @64×128** (`RESULTS.md` §3.8) — NOT the 57.7 single-tile (too easy) and NOT the 0.611 N-decode overlap-blend (different decode budget). Same UPerNet, same seeds, same 180-pano subsets; reuse `scripts/verify_upernet.py` + `multitask_eval.py` verbatim. Linear/kNN probe **forbidden** (§3.9 laundering). ≥3–5 seeds.
- **Baseline-mismatch caution (do not read FishRoPE as validation):** FishRoPE's +3.7 is spherical-RoPE > **planar**-RoPE on the *same distorted ERP input*. It never beat an undistorted-tile-scatter baseline. This project's baseline is E2P tiles, which **already** solve the distortion problem in pixel space — tiling and spherical-RoPE are two solutions to the SAME problem, and tiling's win is already priced into 0.557. The treatment must beat 0.557, not the planar-ERP strawman FishRoPE beat.
- **GREENLIGHT / KILL:** GREENLIGHT = treatment beats the paired scatter-field arm by > std, clearing the §3.9 de-risk bar (frozen ~0.543±0.028 at this scale). KILL = ≤ 0.557 within noise → spherical-PE ERP-single-pass cannot even match scatter-mean → the entire fused-teacher + ERP-native-training + loss-③ build is dead.
- **Honest EV:** P(beat 0.557 under a real multi-seed UPerNet) ≈ **0.15–0.35**, modal outcome **null/tie** (~50–55% wash near 0.557, ~20–25% regression from unhealed k_proj + pole content-stretch, ~15–20% real win). **This can move the ACCURACY verdict as an architecture/distortion lever; it CANNOT move the SSL-negative** (injects no new single-view information).

---

## 6. Engineering-cost note (from the code audit)

**It is NOT "just a PE module + a teacher fusion loop."** Honest decomposition:

1. **Spherical PE swap — genuinely small and coherent.** The code-audit **overturned the briefing's assumption**: DINOv3 ViT-B/16 as loaded (`facebook/dinov3-vitb16-pretrain-lvd1689m`) uses **2D axial RoPE** (`rope_theta=100.0`, `DINOv3ViTRopePositionEmbedding`, `apply_rotary` on q&k patch tokens), NOT learned absolute position embeddings. *(Independently re-confirmed this session: `transformers 5.12.1` exposes `DINOv3ViTRopePositionEmbedding` and the `rope_theta` / `apply_rotary` / `get_patches_center` internals — the swap target genuinely exists in the loaded model.)* So "swap RoPE" is a targeted module swap (replace `get_patches_center_coordinates` + yaw-angle build), the least-invasive item. **Two caveats:** (a) circular-yaw changes RoPE's **functional form** (periodic vs linear), so "LoRA heals the statistics shift" is asserted, unproven; (b) baseline `LORA_TARGETS='qv'` adapts only q_proj/v_proj while RoPE rotates BOTH q and k — **k_proj sits un-adapted on the frozen path the basis change most disturbs** (would want k_proj in targets).
2. **ERP-native training pipeline — new.** Today training is strictly E2P tiles (`train_ssl.py`: `load_erp → render_tiles → enc(tiles)`; teacher & student both on tiles). ERP is **eval-only** and runs **planar** RoPE (`probe_seg_dinov3.py:feats_erp`, `tiling_compare.py:erp_direct`) — precisely why `erp_direct` trails E2P in-repo. An ERP-native training loop with the swapped PE live is a substantial build.
3. **Fused tangent-teacher — does NOT exist.** `encoder.teacher()` is the frozen **self** (same backbone, LoRA disabled via `disable_adapter`), a single-tile frozen teacher — NOT an area-weighted tile ENSEMBLE. The fused teacher must be assembled from `fusion.py` (`scatter_mean_field`, `SetFusion`, `pack_sets`) + `geometry.py` warp/coordmaps + cos-lat weighting. Reusable pieces exist, but the integration is new and non-trivial.
4. **Q3 de-risk — mandatory.** A real trained UPerNet/DPT multi-seed run, because §3.9 shows fusion/consistency-class gains launder under strong decoders (F-2 −0.065, 0/5 seeds). A probe win does not count.

**Bottom line:** "PE swap (small) + a NEW ERP-native training pipeline + a NEW fused teacher + a real multi-seed decoder de-risk fighting four documented locks." The recipe's framing hides items 2–4 behind the cheap PE swap.

---

### Sources (re-confirmed this session)
- `docs/SSL_ACCURACY_DESIGN.md` — Q0 gate text (L67-89), M2 graveyard entry (L333-336), MG-SOG Q0/Q1 exclusion (L100-114), warp-equivariance dead-invariance (L329-332).
- `docs/SEMANTIC_IDENTITY_SSL.md` — single_fair=0.367 / blend_fair=0.455 (L272), recon_cos=0.923 → decoded 0.339 (L274-276), F-2 laundering −0.065±0.016, 0/5 seeds (L440, §9.7), TC3 blend_fair=0.473 (L587).
- `docs/RESULTS.md` — scatter-mean field 0.557 @64×128 (L179-184), overlap-blend 0.611 (L65), §3.9 multi-seed de-risk seg +0.004±0.010, 2/4 noise (L210-215).
- `docs/CAN_SSL_RAISE_ACCURACY.md` — frozen-anchor ceiling / Term B causes (§3-4), open levers = inter-pano parallax (L160-165) + inverse-variance/Kalman fusion (L150-154, L176).
- Code audit / citation verification (workflow inputs, this session) — RoPE confirmed present; encoder.teacher()=frozen-self; healpix_seg.py=eval mosaic; all 8 papers real, DiT360 misuse, FishRoPE fails filter.