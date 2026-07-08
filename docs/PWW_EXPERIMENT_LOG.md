# PWW / Term B — 실험 로그 (2026-07-07~08)

이번 세션의 실험 여정: **"자가지도로 정확도를 올릴 수 있나"** 를 문헌 → 설계 → 진단 → 구현 → 학습 →
평가 → 손실 감사까지 끝까지 밀어붙인 기록. 정제 수치는 아래 표, 설계 근거는 companion 문서
(`SSL_ACCURACY_DESIGN.md`, `PANO_WHEREWHAT_SPEC.md`, `CONSISTENCY_AND_RICHNESS_LIT.md`,
`SSL_SUCCESS_CASES_LIT.md`) 참조.

---

## 0. 출발 질문과 게이트

정확도-via-SSL의 벽(consistency≠accuracy)을 넘으려면 신호가 **새 정보를 주입**해야 한다. 모든 아이디어는
4-부 게이트를 통과해야 함(`SSL_ACCURACY_DESIGN.md`): **Q0** 자가지도인가(GT 없음) · **Q1** frozen에 없는
단일뷰 접근 정보인가 · **Q2** 단일뷰-접근(앙상블 분산감소 아님, M2) · **Q3** 실디코더 생존(probe 세탁 아님).

---

## 1. 문헌 조사 (3 딥리서치 워크플로우)

- **`SSL_SUCCESS_CASES_LIT.md`** (67편): frozen foundation을 dense 정확도로 이긴 검증 사례 **3편뿐(DIP/NeCo/LoRA3D),
  전부 MIM 아님**. 우리 regime(frozen+저데이터+dense)엔 0. CroCo→DUSt3R→MASt3R은 mechanism ✓ / regime ✗
  (전부 from-scratch·대규모, geometry에만 이김).
- **`CONSISTENCY_AND_RICHNESS_LIT.md`** (56편): 트레이드오프(일관성↔풍부함)를 우리 축에서 깨끗이 깨는 논문 0;
  경로 3갈래(coding-rate/rank-max, equivariance, relational). 보존형 vs 개선형 구분.
- **`SSL_ACCURACY_DESIGN.md`** (설계 판정단 + 독립 레드팀): Q0 추가로 순위 뒤집힘 →
  **#1 PANO-iBOT(진짜 SSL) · #2 RICA · MG-SOG 제외(GT-supervised)**.

---

## 2. 실험 E1 — 위치 pretext headroom 진단 → **NULL (학습 전 킬)**

`scripts/diag_position_headroom.py`: frozen DINOv3 tile-pooled feature를 linear-probe로 pitch/FOV 예측.
사용자 피드백(소표본 우려)에 따라 pano-disjoint split까지 확인.

| split | pitch decoded | FOV decoded |
|---|---|---|
| tile-random (2400 tiles) | 91.6% (MAE 2.5°) | 78.2% (MAE 2.5°) |
| **pano-disjoint (3600 tiles)** | **85.9% (MAE 4.2°)** | **70.2% (MAE 3.6°)** |

**판정:** frozen이 pitch·FOV를 이미 강하게 인코딩 → 위치 예측은 readout(Q1-null) → **학습해도 frozen±ε** →
학습 전에 킬(iron law #1). ERP-roll 위치 pretext 폐기. (`geometry.tile_position_labels` + `test_position.py`
6/6 통과는 보존 — 라벨은 기하적으로 정확.)

---

## 3. 실험 E2 — Term B (cross-view masked completion) v1

**설계:** 겹침 타일 A의 마스킹 블록을, 타일 B의 warp 위치 evidence + A 문맥으로 예측. 타깃 = **A의 frozen 특징**
(de-overlap 규칙 → B→A 왜곡 변환 학습, 복사 불가; frozen 타깃 → 침식 방지). 예측기 = zero-init residual.

**구현:** `encoder.CrossViewPredictor`, `losses.cross_view_completion_loss`, `scripts/train_pano_termb.py`
(`build_geometry` pairs+warps 재사용). 테스트 `test_termb.py` 5/5.

**학습** (810 pool, 3ep, obliquity 가중, gram-on-masked): comp 0.27→0.076, erank teacher 추적(침식 없음),
0.591M LoRA + 1.77M 예측기.

**평가 결과 → null-negative** (아래 종합표 v1 열).

---

## 4. 실험 E3 — 손실 감사 + Term B v2 (수정본)

**감사 발견:**
1. **obliquity 가중이 completion엔 거꾸로** — 경사 셀(정보 많은 B→A 변환)을 down-weight, 정면 셀(쉬움)을
   강조. warp-consistency에선 옳았으나 completion엔 반대.
2. **gram_anchor를 마스킹 student에** 검 → mask_token/frozen 불일치.

**수정** (env 토글, 기본=수정본): `COMP_WEIGHT=uniform`, `GRAM_ON=full`. 재학습(`runs/ckpt_pano_termb_uniform`).

**핵심 관측:** **gram이 학습 내내 0.001 평탄** → unmasked LoRA 표현이 frozen과 구조적으로 거의 동일 유지 =
**인코더가 거의 안 움직임** (예측기가 completion 흡수 + frozen 앵커).

**결과 → v1과 사실상 동일** (종합표 v2 열). 수정은 방향 효과만(uniform→ARI 소폭↑) 있었고 정확도 무변.

---

## 5. 종합 결과표

### 5.1 정확도 (Stanford2D3D area5, 3-seed, encoder-fixed, frozen 대비 Δ)

| task (head) | frozen | Term B v1 (obliquity) Δ | Term B v2 (uniform) Δ |
|---|---|---|---|
| seg Linear (no-finetune) ↑ | 0.529 | **−0.026** | **−0.023** |
| seg UPerNet ↑ | 0.518 | −0.007 | −0.017 |
| normal Linear ↓ | 57.5° | −0.01 (tie) | −0.10 (tie) |
| normal UPerNet ↓ | 54.6° | +0.66 | +0.95 |
| depth Linear ↓ | 0.196 | ~0.000 (tie) | −0.003 (tie) |
| depth UPerNet ↓ | 0.172 | +0.012 | +0.007 |

**어디서도 frozen 못 이김.** linear-probe geometry는 tie, seg는 −0.02, UPerNet은 mild-worse. v1=v2.
(단일-seed의 "normal +0.58 worse"는 노이즈였음 — multi-seed로 tie 확인.)

### 5.2 일관성 + 다양성 (DensePASS, K=64)

| encoder | ARI (일관성↑) | feat cosine | purity (다양성) | cluster→mIoU | blend_fair |
|---|---|---|---|---|---|
| frozen | 0.236 | 0.678 | 0.838 | 0.269 | 0.455 |
| **TC3** (기존 챔피언) | **0.595** | 0.921 | **0.862** | **0.344** | 0.473 |
| Term B v1 | 0.236 | 0.769 | 0.844 | 0.288 | 0.472 |
| Term B v2 | 0.274 | 0.700 | 0.834 | 0.269 | 0.465 |

**Term B(양 변형) 모든 축에서 TC3에 밀림.** 정확도 null-neg, 일관성 null(ARI≈frozen), 다양성 보존(v1)~미세침식(v2).

---

## 6. 핵심 결론

1. **위치 pretext = null** (frozen이 pitch/FOV 이미 인코딩).
2. **Term B(v1/v2) = 전축 null** — frozen을 정확도로, TC3를 일관성/다양성으로 못 이김.
3. **손실 가중치는 병목이 아니었다** (v1=v2 확증). audit은 옳은 코드 비판이었으나 방향 효과만 있고 크기 무의미.
4. **구조적 천장 3종:** (a) 예측기(1.77M)가 completion 흡수 → 인코더 정체, (b) frozen 타깃 앵커 → frozen 상한,
   (c) **intra-pano 무시차** → completion이 왜곡정규화(=일관성 계열)로 환원.
5. **완결된 통찰:** 이 intra-pano cross-view 목적함수는 인코더를 *정확도 방향*으로 움직이는 설정이 없다 —
   예측기 있으면 안 움직이고(Term B), 없으면 *일관성*으로 움직임(=geo, ARI 0.495). **일관성 ≠ 정확도.**

---

## 7. 산출물

**문서:** `SSL_SUCCESS_CASES_LIT.md`, `CONSISTENCY_AND_RICHNESS_LIT.md`, `SSL_ACCURACY_DESIGN.md`,
`PANO_WHEREWHAT_SPEC.md`, 본 로그.
**코드:** `geometry.tile_position_labels`, `encoder.CrossViewPredictor`, `losses.cross_view_completion_loss`,
`scripts/train_pano_termb.py` (COMP_WEIGHT/GRAM_ON env), `scripts/diag_position_headroom.py` (PANO_SPLIT env),
`scripts/multitask_eval.py` (SEED/SEEDS/ONLY_HEADS env).
**테스트:** `tests/test_position.py` (6), `tests/test_termb.py` (5) — 11/11 통과.
**체크포인트:** `runs/ckpt_pano_termb_uniform` (v2). (v1은 smoke가 실수로 clobber; 수치는 본 로그에 기록.)

---

## 8. 남은 선택지

1. **Inter-pano parallax completion** — intra-pano가 못 주는 진짜 삼각측량 정보. 유일하게 미검증 genuinely-new
   레버. **선행: MP3D/S3D 다중뷰 페어 데이터 가용성 확인** (없으면 이 길도 막힘).
2. **정리 + 방향 전환** — 완결된 negative를 남기고 어댑터가 실제 이기는 축(TC3 일관성, depth/pointmap 융합+Kalman)으로.
