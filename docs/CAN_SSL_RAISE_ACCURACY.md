# 자가지도로 정확도를 올릴 수 있는가 — pano-ssl-dense 종합 판정

**세션: 2026-07-07~08.** 이 문서는 이번 세션 내내 붙든 하나의 질문 — *"frozen DINOv3 + E2P + LoRA
파이프라인에서, 자가지도(SSL) 표현학습으로 다운스트림 정확도를 frozen 위로 올릴 수 있는가?"* — 에 대한
문헌·진단·직접 실험을 하나의 답으로 종합한다. 세부는 companion 문서 참조:
`SSL_SUCCESS_CASES_LIT.md`, `CONSISTENCY_AND_RICHNESS_LIT.md`, `SSL_ACCURACY_DESIGN.md`,
`PANO_WHEREWHAT_SPEC.md`, `PWW_EXPERIMENT_LOG.md`.

---

## 한 줄 답

**이 regime(frozen ViT-B + 소규모 label-free pano + dense)에서 자가지도로 정확도를 frozen 위로 올리는
검증된 경로는 없다.** 정확도는 teacher(frozen)의 정보량에 묶이고, 자가지도 신호는 **일관성·융합**은
움직이지만 **정확도**는 못 움직인다. 이번 세션은 이 명제를 **문헌 → 진단 → 직접 실험** 세 층위로 확인했고,
*왜 그런지*의 구조적 메커니즘까지 규명했다. 정직한 결론은 **rigorous negative** 이며, 이것이 이 프로젝트의
방법론("정직한 negative가 모여 thesis가 된다")대로 하나의 방어 가능한 판정이다.

---

## 0. 판정 기준 (4-부 게이트)

정확도를 올리려면 SSL 신호가 **새 정보를 주입**해야 한다. 모든 후보는 순서대로 통과해야 한다
(`SSL_ACCURACY_DESIGN.md`):

- **Q0 — 자가지도인가?** 새-정보 항이 사람/GT/시뮬레이터 라벨에서 자유로운가.
- **Q1 — 새 단일뷰 정보인가?** frozen DINOv3에 아직 없고 단일뷰로 접근 가능한 정보를 주입하는가.
  (순수 consistency = 정보 0 → 즉시 탈락.)
- **Q2 — 단일뷰-접근인가?** 이득이 단일뷰에 있는가, 아니면 앙상블 분산감소(M2, 구조적 복원불가)인가.
- **Q3 — 실디코더 생존인가?** 진짜 UPerNet/DPT·multi-seed에서 살아남는가, probe-레벨 세탁인가(§3.9/F-2).

---

## 1. 증거 A — 문헌 (deep-research 3 워크플로우, 약 123편)

- **frozen foundation을 dense 정확도로 이긴 검증 사례는 극소:** `SSL_SUCCESS_CASES_LIT.md` 67편 중 **3편**
  (DIP·NeCo·LoRA3D), 그나마 **전부 MIM 아님**이고 probe-eval·부분 unfreeze 등 우리 regime과 어긋남.
  우리 코너(in-domain-강함 × 저데이터/frozen × dense)엔 **0건**.
- **cross-view completion(CroCo→DUSt3R→MASt3R)** 은 mechanism ✓ / regime ✗ — 전부 from-scratch·대규모,
  그리고 frozen DINO를 이긴 것도 **geometry/correspondence에서만**(ZeroCo/MuM/Muskie), 단일뷰 **의미**는
  DINOv3가 여전히 우세.
- **일관성+풍부함 동시 향상**(`CONSISTENCY_AND_RICHNESS_LIT.md` 56편): 우리 축(erank/purity, frozen+LoRA,
  dense)에서 트레이드오프를 깨끗이 깬 논문 0. 원리적 경로(coding-rate/rank-max, equivariance, relational)는
  전이 가설이지 증명된 해결책 아님.

**문헌 판정:** 우리 regime에서 정확도-via-SSL의 사전확률은 낮다.

---

## 2. 증거 B — 진단 (위치 pretext = null, 학습 전 킬)

사용자 아이디어(ERP-roll 멀티-그래뉼러 위치 예측 + FOV 랜덤화)를 **학습 전에** frozen 진단으로 게이팅
(`scripts/diag_position_headroom.py`, iron law #1):

| split | pitch decoded | FOV decoded |
|---|---|---|
| tile-random (2400) | 91.6% (MAE 2.5°) | 78.2% (MAE 2.5°) |
| **pano-disjoint (3600, leakage-proof)** | **85.9% (4.2°)** | **70.2% (3.6°)** |

frozen이 pitch·FOV를 **이미 인코딩** → 위치 예측 = readout(Q1-null) → 학습해도 frozen±ε. **몇 분 진단이
수 시간 헛학습을 막음.** (사용자의 소표본 우려는 정당했고 pano-disjoint로 확인 — 수치가 5~8pt 내렸으나
판정은 불변.)

---

## 3. 증거 C — 직접 실험 (Term B cross-view completion, v1·v2 모두 null)

문헌·진단이 가리킨 유일한 live 레버 = **cross-view masked completion**(CroCo의 파노 버전). 완전 구현·학습·
평가(`PWW_EXPERIMENT_LOG.md` 상세).

### 3.1 정확도 (Stanford2D3D area5, 3-seed, encoder-fixed, frozen 대비 Δ)

| task (head) | frozen | v1 (obliquity) Δ | v2 (uniform, 수정) Δ |
|---|---|---|---|
| seg Linear (no-finetune) ↑ | 0.529 | **−0.026** | **−0.023** |
| seg UPerNet ↑ | 0.518 | −0.007 | −0.017 |
| normal Linear ↓ | 57.5° | tie | tie |
| normal UPerNet ↓ | 54.6° | +0.66 | +0.95 |
| depth Linear ↓ | 0.196 | tie | tie |
| depth UPerNet ↓ | 0.172 | +0.012 | +0.007 |

**어디서도 frozen 못 이김.** linear-probe geometry는 tie, seg는 −0.02, UPerNet은 mild-worse. (단일-seed
"normal worse"는 노이즈 → multi-seed로 tie 확인.)

### 3.2 일관성 + 다양성 (DensePASS, K=64)

| encoder | ARI (일관성↑) | purity (다양성) | cluster→mIoU | blend_fair |
|---|---|---|---|---|
| frozen | 0.236 | 0.838 | 0.269 | 0.455 |
| **TC3** (기존 챔피언) | **0.595** | **0.862** | **0.344** | 0.473 |
| Term B v1 | 0.236 | 0.844 | 0.288 | 0.472 |
| Term B v2 | 0.274 | 0.834 | 0.269 | 0.465 |

**Term B(양 변형) 전축에서 TC3에 밀림.** 정확도 null-neg, 일관성 null(ARI≈frozen), 다양성 보존~미세침식.

### 3.3 손실 감사 (v1→v2)

obliquity 가중이 completion엔 거꾸로(정보 많은 경사 셀을 down-weight)임을 코드 감사로 발견 → uniform +
gram-on-unmasked로 수정(v2). **결과는 v1과 동일** — 학습 중 `gram`이 0.001 평탄(인코더 거의 안 움직임) →
**손실 가중치는 병목이 아니었음이 확증.**

---

## 4. 왜 안 되는가 — 구조적 메커니즘 (핵심 기여)

이번 세션의 진짜 산출물은 *null*이 아니라 **왜 null인지의 규명**이다. 정확도가 안 오르는 원인 3종:

1. **예측기 흡수.** Term B의 예측기(1.77M, 무제약)가 B→A 변환을 학습해 손실을 낮춤 → 정작 인코더(0.59M
   LoRA)는 학습의 주체가 안 됨(`gram` 0.001 평탄이 실측).
2. **frozen 앵커 상한.** 타깃이 frozen이라 침식은 막지만(good) 동시에 **인코더를 frozen 정보 상한에 묶음** →
   재구조화(새 의미) 불가. 프로젝트의 **보존 vs 재구조화** 긴장 그대로.
3. **intra-pano 무시차.** E2P 겹침 타일은 광학중심 공유 → 삼각측량 정보가 없음 → "cross-view completion"이
   실질적으로 **왜곡 정규화**(=일관성 계열)로 환원. CroCo/DUSt3R가 이득을 본 진짜 3D는 **실제 baseline(시차)**
   에서 나오는데 intra-pano엔 없다.

**완결된 통찰:** 이 intra-pano cross-view 목적함수는 **인코더를 *정확도 방향*으로 움직이는 설정이 존재하지
않는다.** 예측기를 두면 안 움직이고(Term B), 빼면 comp이 "student_B(warp)→frozen_A"로 환원돼 인코더가
*일관성 방향*으로 움직인다(=geo, ARI 0.495). 그리고 **일관성 ≠ 정확도**(프로젝트 iron thesis). 즉
움직임의 방향 자체가 정확도가 아니다.

---

## 5. 프로젝트 전체와의 정합 (반복 확인된 불변식)

이번 세션의 null은 이상현상이 아니라 프로젝트가 반복 확인해온 불변식의 재확인이다:

| 시도 | 정확도 | 비고 |
|---|---|---|
| geo warp-consistency | flat | 일관성↑ / 정확도 flat |
| M1 code-agreement | flat | 침식(purity 0.730)까지 |
| F-2 learned fusion | 진 | 실디코더에서 mean에 패배 |
| F-3 Pano-JEPA EMA | flat | 침식 |
| **TC3** | flat | 일관성↑·무침식 (챔피언, 단 정확도는 flat) |
| **21.8k scale** | **최저** | scale이 정확도를 못 삼 |
| **위치 pretext (이번)** | null(진단) | frozen이 이미 인코딩 |
| **Term B v1/v2 (이번)** | null-neg | 구조적 3종 원인 |

→ **"자가지도 어댑터는 cross-view *일관성* 어댑터지 *정확도* 어댑터가 아니다"** 가 이번 세션으로 8번째
확인됨. 정확도는 teacher 정보량에 묶인다.

---

## 6. 그럼 어댑터는 어디서 이기나 (재프레이밍)

정확도는 아니지만, 어댑터가 **실재하고 재현되는** 이득 두 가지:

1. **일관성:** TC3가 cross-view 코드 일치(ARI 0.236→0.595)를 침식 없이(purity 0.838→0.862) 크게 올림.
   head-free correspondence retrieval도 0.21→0.86. 이건 어댑터의 확고한 홈그라운드.
2. **융합 천장:** 모든 어댑터가 blend_fair(멀티뷰 융합 상한)를 frozen 위로 올림(0.455→0.47대). 단
   uniform-mean 융합으로 수확되며(F-2 학습융합은 세탁), **원리적 수확 = 역분산(Kalman) 융합**(depth/pointmap)
   이 미탐색 상방.

**정직한 방향 전환:** 정확도-via-표현학습을 쫓는 대신, (a) 일관성 축의 TC3를 배포 챔피언으로,
(b) 융합 축(Kalman/uncertainty, depth·pointmap)을 다음 정확도-관련 실험으로.

---

## 7. 남은 미검증 레버 (정직한 열림)

**Inter-pano parallax completion** — intra-pano가 구조적으로 못 주는 **진짜 삼각측량 정보**. 유일하게
"이번엔 다를 수 있는" genuinely-new 레버. 단 현실 제약:
- **데이터:** MP3D는 현재 디스크 미탑재(`matterport3d` dir 없음). Stanford2D3D는 pose+global_xyz가 있어
  이론상 겹침-시차 대응 계산이 가능하나 미검증.
- **복잡도:** 시차 대응은 **depth-의존**이라 intra-pano의 무시차 warp(`warp_field_from_coordmaps`) 재사용
  불가 → 사실상 mini-DUSt3R(pointmap 예측) 빌드 필요. 큰 작업.

→ 값어치는 있으나 **큰 빌드 + 데이터 선결**. 착수 전 데이터 확보와 범위 산정 필요.

---

## 8. 판정 요약

- **질문:** 자가지도로 정확도를 올릴 수 있는가? → **이 regime에선 검증된 경로 없음 (rigorous negative).**
- **근거:** 문헌 0건 + 위치 진단 null + Term B v1/v2 null + 구조적 3원인 규명 + 프로젝트 8번째 확인.
- **어댑터의 진짜 가치:** 정확도 아님, **일관성(TC3)·융합 천장**.
- **유일한 미검증 상방:** inter-pano parallax (큰 빌드 + 데이터 선결) / Kalman 융합(depth·pointmap).
- **방법론적 성취:** 몇 분짜리 frozen 진단이 헛학습을 막았고(iron law #1), 손실 감사가 "가중치는 병목
  아님"을 확증했으며, null의 *구조적 원인*까지 규명 — 이 프로젝트에서 negative는 결함이 아니라 결과다.
