# Semantic-Identity SSL (SI-SSL) — 설계 문서

파노라마 분해(E2P) + overlap을 이용한 **의미적 동일성(semantic identity)** 기반 SSL 설계.
현재의 geometric-identity SSL이 도달한 지점("일관성 어댑터, 정확도 아님", `RESULTS.md`)에서
출발해, 다음 축을 정의하고 통합 방식·진단·평가까지 한 문서에 담는다.

> 사용자 정의(핵심): *"latent feature 레벨에서, 동일한 위치에서 주변과 기하적 특징이 다르더라도
> 의미적으로 동일한 부분이라는 것을 학습하도록"* — 이 문장을 그대로 목적함수로 번역하는 것이 이 문서의 과제다.

> **Canonical 정식화 (사용자, 2026-07-02, TC3 검증 후):** *"자가지도는 기하적 변화가 있는
> 타일들에서, 오버랩·특징 추출 지점에서, **전체 파노라마 이미지에** 의미적·기하적으로 일치하는
> 특징을 만들어내게끔 하는 과정이며; 융합은 이런 특징들을 통합해 다운스트림 헤드로 보내기 위함."*
> — 현재 구현은 "전체와의 일치"를 인접-쌍 transitivity로 근사(간접). 명시적 전역 항 후보:
> (a) tile↔융합필드 일치, (b) **masked-view modeling**(타일 숨기고 나머지 융합으로 예측 —
> 융합이 자가지도 목적함수로 승격, F-2 label-free 변형), (c) global ERP context(AnyRes 원설계,
> SSL 미사용). S3D 21.8k 도착으로 (b)가 데이터적으로 가능해짐.

---

## 0. 한 줄 요약 & 위치

- **현재 신호(geometric identity):** `warp_equivariance_loss` — 같은 ray의 두 뷰에서
  raw feature를 cosine으로 맞춘다: `F_A(p) ≈ F_B(Hp)`. 무시차 homography라 대응은 정확하지만,
  손실은 왜곡 잔차를 **양보**한다(`losses.py`: obliquity weighting으로 "irreducible edge-stretch
  residual"을 덜 벌함). 결과 = 일관성↑, 정확도 flat.
- **다음 축(semantic identity):** 같은 위치의 콘텐츠를, 주변 문맥·기하 왜곡이 달라도 **동일한 잠재
  의미 코드**로 매핑하도록 학습. 왜곡 잔차를 *양보*하지 않고 *추상화*한다.
- **핵심 베팅 = M1** (overlap semantic-code agreement). **게이팅 확장 = M2** (앙상블→단일 정확도 전이).
- **정직한 through-line:** 이 프로젝트의 창립 제약은 "scarce data → **보존**(distill anchor)"인데,
  semantic identity가 feature를 **재구조화**하려면 그 anchor를 풀어야 한다. 이 *보존 vs 재구조화*
  긴장이 성패를 가르며, "얼마나 풀 것인가"는 knob이 아니라 **진단 질문**이다(§5).

---

## 1. 문제 재정의: geometric identity → semantic identity

### 1.1 형식화 (사용자 문장의 번역)

3D ray `x`가 tile A에는 픽셀 `p_A`, tile B에는 `p_B = H·p_A`로 나타난다. 두 뷰에서 그 지점의
국소 관측은 다르다:

```
관측_A(x) = ( 기하 φ_A : gnomonic 왜곡·경사·스케일,  문맥 N_A : 주변 이웃 )
관측_B(x) = ( 기하 φ_B,                              문맥 N_B )      with  φ_A≠φ_B, N_A≠N_B
```

- **현재(geometric):** `F_A(p_A) ≈ F_B(p_B)`를 raw feature에서 강제. 그러나 ViT 토큰은 문맥
  `N`으로 attention-contextualized 되고 `φ`에 민감 → 등식은 근사에 그치고, obliquity weight가
  잔차를 *양보*한다.
- **목표(semantic, 사용자 요구):** 잠재 코드 `z = h(F)`를 학습해 `z_A(x) = z_B(x)`가 **정확히**
  성립하도록. 즉 `z`는 `(φ, N)`에 **불변**인 "그 ray의 의미 정체성".

> "what"(의미, 불변) 을 "how imaged"(기하/문맥 nuisance) 로부터 **분리**하는 것. raw warp는
> "how"를 양보하고, semantic-code는 "how"를 소거한다.

### 1.2 왜 이것이 파노라마 특화인가 (generic SSL과의 차이)

1. **False-positive 0.** 오버랩 대응은 기하적으로 정확(무시차 homography, `geometry.py`). SimCLR류
   random-crop positive와 달리 "정말 같은 콘텐츠"임이 보장된다.
2. **불변시키는 nuisance가 정확히 우리가 원하는 것.** 두 뷰의 차이 `(φ_A,N_A)↔(φ_B,N_B)`는
   파노라마가 만드는 왜곡·경사·프레이밍·문맥 차이 그 자체. 여기서 배우는 불변성은 곧 **파노라마
   왜곡 불변성** — color jitter 불변성 같은 generic augmentation invariance가 아니다.
3. **Hard positive 무료.** 같은 객체를 서로 다른 obliquity/왜곡으로 본 쌍은 color-jitter보다 훨씬
   유익한 hard positive.
4. **문맥 불변성.** 같은 ray가 A에선 중앙(넓은 문맥)·B에선 가장자리(잘린 문맥)에 놓인다. "문맥이
   달라도 같은 의미"를 배우는 건 raw-feature matching이 줄 수 없는, 더 의미적인 불변성이다.

---

## 2. 핵심 기법 M1 — Overlap Semantic-Code Agreement

**한 줄:** 오버랩 대응에서 raw-feature cosine을 **공유 프로토타입 코드 일치**로 격상. 두 뷰가 같은
위치를 *같은 클러스터/프로토타입*에 배정하도록 학습 → 왜곡·문맥에 불변인 의미 코드가 생긴다.

### 2.1 구조 (DINO/SwAV/iBOT 계열, dense 버전)

- **Projector + Prototype head** `h`: dense feature `F ∈ R^{D×Gh×Gw}` → code logits
  `Z ∈ R^{K×Gh×Gw}`, `Z = C·g(F)/τ` (`g`=2-layer MLP projector, `C`=K개 학습 프로토타입,
  L2-normalized; `K`≈256~2048, `τ`≈0.1). LoRA(0.59M) 위에 얹는 **작은 신규 head**뿐.
- **손실이 계산되는 곳 vs 불변성이 필요한 곳을 구분하라(핵심).** 손실은 projected code `Z`에서
  계산되지만, downstream dense task와 사용자가 말한 "latent feature 레벨"은 **backbone `F`**다
  (projector head는 downstream에서 버려짐). 따라서 M1의 실제 성패는 *"`Z`에 건 code-agreement
  압력이 거의-frozen backbone `F`(+0.59M LoRA)까지 전파되는가"* 라는 **열린 질문**에 달렸다. 이
  regime에선 표준 SwAV/DINO보다 더 첨예하다 — projector 혼자 손실을 만족시키고 backbone은 거의
  안 움직일 수 있다. backbone은 warp로 저수준 정렬(유지), code head는 `Z`에서 semantic identity를
  걸되, **`F`로의 전파 여부는 진단 D-B로 실측한다(§5).**

### 2.2 손실 — swapped code prediction on the overlap

기존 `warp_field_from_coordmaps`가 주는 A-cell→B-cell 대응(grid, valid, weight)을 **그대로 재사용**해
코드를 정렬한다:

```
# A의 target 코드는 B의 대응 셀에서 grid_sample (기존 warp grid 재사용)
q_B = sinkhorn( Z_B_teacher )          # balanced soft assignment (SwAV), stop-grad target
p_A = softmax( Z_A / τ_s )             # student prediction at A cells
L_code(A→B) = - Σ_valid  w · < q_B(Hp),  log p_A(p) >     # cross-view swapped prediction
L_code = ½ ( L_code(A→B) + L_code(B→A) )                  # 대칭
```

- **Sinkhorn-Knopp** 균등 배정으로 target을 만들어 trivial collapse(모두 한 코드) 방지(SwAV).
- **stop-grad target** — 기존 warp loss와 동일 관례. (대안: EMA mean-teacher head; §3.4에서 선택.)
- **obliquity weight `w` 재사용** — 단, raw warp와 달리 여기선 잔차를 *양보*가 아니라 신뢰가중으로만.

### 2.3 왜 raw-cosine보다 의미적 동일성에 맞나

| | raw warp (현재) | semantic code (M1) |
|---|---|---|
| 정렬 대상 | 문맥·왜곡 민감한 raw feature | `(φ,N)` 소거된 코드 |
| 잔차 처리 | 양보(obliquity down-weight) | 추상화(같은 코드면 0) |
| "같은 위치, 다른 기하" | 근사 등식 | **정확 등식(코드)** |
| 정확도 축 | 정렬만(재구조화 X) | 프로토타입 = 의미 분할을 *생성* |

**M1의 정직한 가치(H1, 높은 확신):** 사용자가 요구한 성질 — "같은 위치를 왜곡·문맥이 달라도
의미적으로 동일하게" — 은 본질적으로 **불변성/일관성** 성질이다. M1은 이걸 raw-feature warp보다
*더 의미적인* 수준(문맥·왜곡 불변 코드)에서 정확히 제공한다. 이것이 M1의 고신뢰 기여다.

**천장 돌파(H2, 열린 베팅 — M2와 같은 바):** "프로토타입이 의미 파티션을 *생성*하니 정확도
천장도 넘는다"는 건 *가설*이지 약속이 아니다. 이 프로젝트의 모든 일관성 objective는 H1-yes /
H2-flat이었고(`RESULTS.md` §3.1), §3.9는 정확도 이득 false-positive까지 잡았다. 게다가 그 파티션
생성이 정확도로 이어지려면 §2.1의 **backbone 전파**가 선행돼야 한다. 따라서 M1의 정확도 주장은
M2와 **동일한 회의·게이팅 기준**으로 다룬다(D-B에서 backbone 전파 실측, §6.1 H2로 반증).

### 2.4 Anti-collapse (필수)

클러스터링 objective는 붕괴 위험이 크다. 3중 방어: **Sinkhorn**(배정 균등) + **prototype entropy**
정규화(코드 사용률 평탄) + 기존 **VICReg floor**(`gamma=0.04`, `losses.py`) + erank 모니터(`train_ssl.py`).

---

## 3. 통합 방식 (integration) — 보존 vs 재구조화

이 절이 through-line이다. 손실 하나 추가가 아니라, **distill anchor의 재조정**이 진짜 통합 문제다.

### 3.1 긴장의 정체

- `distill_loss`의 **token 항**(per-token cosine to teacher)은 feature를 teacher에 **핀**한다.
  이것이 (a) scarce-data anti-forgetting anchor(창립 제약, `RESEARCH_LOG` Phase 0)이자 동시에
  (b) **정확도 천장의 원인**(feature가 teacher 너머로 못 감, `RESULTS.md` §3.1).
- semantic code가 의미 분할을 *생성*하려면 feature가 움직여야 하는데, token-distill이 막는다.
  **anchor를 풀면** 재구조화가 가능하지만 scarce data + 0.59M LoRA에서 **teacher 정확도를 더 빨리
  잃을** 위험. 이것은 clean win이 아니라 trade다.

### 3.2 제안: relational 유지, token anneal

- **relational distill 유지** (`distill_loss`의 Gram 항): 절대 feature는 풀되 **상호관계 구조**는
  보존 → 붕괴/망각 방지하면서 재구조화 허용.
- **token distill anneal**: warm-up 후 λ_token을 0으로 감소(완전 제거 X, 하한 유지 옵션).
- **단, "relational-only가 충분한 anti-forgetting인가"는 확정 knob이 아니라 진단 D-D(§5)로 검증.**

### 3.3 통합 손실

```
L = λ_warp · L_warp                     # 저수준 기하 scaffold (왜곡 양보, 유지)
  + λ_code · L_code                     # M1: 고수준 semantic identity (신규 핵심)
  + λ_rel  · L_relational_distill       # 구조 보존 (유지)
  + λ_tok  · L_token_distill            # anneal → 0 (재구조화 허용)
  + λ_reg  · L_VICReg                   # anti-collapse (유지, 강화)
  [ + λ_ens · L_ensemble   ]            # M2, 게이팅 통과 시에만 (§4)
```

### 3.4 스케줄

- **Stage A (안정화):** `L_warp + full distill` — 현재 레시피. 대응·feature 안정화. code head는
  Sinkhorn만 예열(gradient 약).
- **Stage B (의미화):** `L_code` ramp-in ↑, `λ_tok` anneal ↓. warp는 저수준 scaffold로 잔존.
  VICReg floor 상시.
- **teacher 선택:** target 코드는 (i) stop-grad 동일 encoder(warp와 일관, 간단) 또는 (ii) EMA
  mean-teacher head(DINO식, 더 안정적이나 복잡). **기본 = (i)**, 붕괴 관찰 시 (ii)로 승급.

---

## 4. 확장 M2 — Ensemble→Single 정확도 전이 (게이팅됨)

**아이디어:** 멀티뷰 오버랩 **앙상블**(covering tile들의 feature 평균, `eval_ssl.scatter_pano`)은
단일 타일보다 semantic이 정확하다(측정 **+0.12 mIoU** @DP50, `RESULTS.md` §2). 이 앙상블 코드를
teacher로, 각 단일 타일 코드를 앙상블 코드로 self-distill → 앙상블 정확도를 단일 feature에 주입.

**⚠️ 게이팅 이유(핵심 경고):** `eval_ssl.py`에서 `blend`는 *covering tile feature의 평균*, `single`은
*least-oblique 단일 타일*. 즉 +0.12는 **평균화 이득**이다. 만약 그 이득의 정체가 독립 뷰 노이즈의
**분산 감소**라면, 단일 뷰는 그 평균을 **구조적으로 복원 불가** → M2는 또 정확도 flat(null). 복원
가능한 건 *canonicalization* 성분(encoder가 배울 수 있는 결정론적 왜곡 보정)뿐이다.

**⚠️ 실측(§8)에서 밝혀진 근본 한계:** *어떤 frozen 프로브도 M2를 settle 못 한다.* M2는 인코더를
재학습해 단일 타일 픽셀에서 *다른* feature를 만드는데, frozen 프로브는 frozen feature(=single_fair)를
구조적 상한으로 가져 인코더의 도달가능 집합을 bound할 수 없다. 따라서 M2 판정은 "게이트 통과"가
아니라 **저-EV·후순위**이며, 유일한 clean 테스트는 짧은 실제 M2 학습이다(§8).

---

## 5. 진단 우선 (diagnose before training) — 프로젝트 법칙

`RESEARCH_LOG` 메타교훈 #1: *학습 전에 진단하라.* 아래는 학습 없이 **frozen feature**로 돌리는
head-only/무학습 프로브. **우선순위대로.**

- **D-A — 단일뷰 복원가능성 (M2 게이트, 최우선).**
  frozen feature에서 head_A(single→GT)=single mIoU, head_B(single→**앙상블의 예측**)를 fit.
  *단일 뷰가 앙상블이 말하는 것을 얼마나 복원하나?* 높음 → 정보가 single-view-accessible →
  LoRA가 각인 가능 → M2 유망. 낮음 → irreducibly multi-view → M2 flat 예상 → **M2 보류.**
  산출: `+0.12` 중 **복원가능 분율(%)**. (신규 스크립트 `diag_semantic_headroom.py`.)
  **⚠️ 실측 교훈(§8): 모든 frozen 프로브(label 예측·feature 회귀 모두)는 single_fair에 구조적으로
  갇혀 M2 recoverability를 판별하지 못한다.** frozen이 말할 수 있는 건 "앙상블 이점이 단일 feature에
  *선형적으로* 없다"까지(clean); "재학습 인코더가 만들 수 있나"는 실제 학습만 답한다.

- **D-B — cross-view 코드 일치도 (M1 헤드룸) + backbone 전파 프로브.**
  frozen feature에 Sinkhorn/k-means로 K 프로토타입 → 오버랩 대응에서 cross-view assignment
  일치도(**ARI/NMI**). 낮으면 M1 헤드룸 큼(코드가 뷰마다 어긋남 = 배울 게 있음), 높으면 이미
  일치(M1 이득 작음). `diag_overlap_match.py`의 대응 로직 재사용.
  **학습 후엔 같은 지표를 projected code `Z`가 아니라 backbone `F`에서 pre/post로 측정** — §2.1의
  "code 압력이 backbone 잠재로 전파되는가"를 직접 검증(이득이 `Z`에만 갇혔는지 vs `F`까지 갔는지).

- **D-C — 프로토타입 순도 (상한).** frozen 프로토타입 vs GT 카테고리 정렬(held-out label로
  cluster→class purity). M1이 담을 수 있는 semantic 정확도의 *상한*을 준다.

- **D-D — 보존 vs 재구조화 (통합 검증).** token-distill 제거(relational-only)로 짧게 학습 →
  teacher token cosine·erank·linear-probe 하락 속도 모니터. §3.2의 "relational이 충분한가"를 실측.

**Laundering guard:** 모든 정확도 주장은 pooled mIoU가 아니라 **assignment-based + head-free**
지표로도 확인(`diag_overlap_match.py`, `diag_consistency_metrics.py` 재사용). `RESEARCH_LOG` D1의
경고("linear head가 일관성을 세탁") 준수.

---

## 6. 가설 · 위험 · 평가

### 6.1 가설과 반증 조건

| | 가설 | 반증 조건 |
|---|---|---|
| **H1** (M1, 거의 확실) | cross-view **코드 일치도↑** | collapse(entropy/VICReg로 감시) — 쉬운 승리, 단 이것만으론 "정확도" 미해결 |
| **H2** (진짜 베팅, 열린) | M1이 **단일 타일 semantic 정확도**를 올림 (왜곡·문맥 불변 코드), head-free/code-level로 laundering-proof | 정확도 flat(또 "일관성-only"), **또는 이득이 projected `Z`에만 있고 backbone `F`로 전파 안 됨(D-B)** |
| **H3** (M2, 게이팅) | D-A 통과 시 single mIoU가 blend 천장으로 이동 | D-A 낮음, 또는 학습 후 single flat |
| **H4** (통합) | token anneal이 forgetting 없이 재구조화 허용 | erank 붕괴 또는 (teacher 정확도 손실 > code 이득) |

### 6.2 정직한 null 위험

DINOv3 feature는 이미 linear-probe near-optimal. 0.59M LoRA + scarce pano로 semantic을 재구조화
못 하면 **M1도 flat**일 수 있다(프로젝트가 반복 확인한 "정확도는 teacher에 묶임"). 그 경우에도
D-A/D-B가 *왜 flat인지*를 사전 예측하므로 기여는 남는다(negative가 thesis가 된다, 메타교훈 #5).

### 6.3 평가 (재사용 우선)

- 정확도: `eval_ssl.py` (single vs blend, **%headroom closed** = 1차 지표).
- 일관성/head-free: `diag_overlap_match.py`, `diag_consistency_metrics.py` (Hungarian/CKA).
- 신규: cross-view **코드 ARI/NMI**, code-level semantic match, (M2) ensemble gap.
- **multi-seed de-risk** 필수 — `RESULTS.md` §3.9의 single-split false-positive 교훈.

---

## 7. 다음 단계 (최소 개입, 진단 먼저)

1. **진단 스크립트 `scripts/diag_semantic_headroom.py`** (D-A/D-B/D-C) — 학습 전 headroom 측정.
   기존 `probe_seg_dinov3` / `eval_ssl.scatter_pano` / `diag_overlap_match` 대응 로직 재사용.
   → **여기서 M1/M2 진행 여부가 판가름.**
2. `losses.py`에 `code_agreement_loss` + Sinkhorn; `encoder.py`(또는 별 head 모듈)에 projector+prototype.
3. `train_ssl.py`에 code head + Stage A/B 스케줄 + token-distill anneal.
4. M2는 D-A 통과 후에만 `L_ensemble` 추가.

**재사용 자산:** `geometry.py`(warp 대응), `encoder.py`(LoRA/teacher), `eval_ssl.py`(scatter/ensemble),
`losses.py`(distill/VICReg/erank). 신규는 code head + Sinkhorn + 진단 스크립트뿐 — 나머지는 재조립.

---

## 8. 진단 1차 결과 (DensePASS outdoor, frozen DINOv3, 100 파노)

`scripts/diag_semantic_headroom.py densepass 64` — 학습 없음, 292s, 70/30 split, 10 equator 타일.

**머신 검증 ✅:** single=0.326 blend=0.445 → **+0.119** — `RESULTS.md` §3.1 DP@50를 정확히 재현.
scatter/ensemble 파이프라인 신뢰 확보(이후 수치의 기반).

**D-A (M2 게이트) — frozen 진단으로는 판정 불가 (핵심 방법론 교훈):**
- fair 기저: **single_fair=0.367 < blend_fair=0.455**. frozen blend가 frozen single보다 선형
  디코딩 가능한 클래스 정보를 ~+0.09 더 담음 → 앙상블 이점은 **단일 feature에 선형적으로 부재**(clean).
- feature-regression 게이트(R:single→blend, recon_cos=**0.923**): W_blend(R(single))=0.339 ≤ single_fair.
  R이 blend를 잘 복원(cos 0.92)해도 정확도는 single_fair를 못 넘음 — 왜냐면 `W_blend∘R`은 single의
  affine 분류기라 **구조적으로 single_fair가 상한**. (label-예측 프로브도 같은 이유로 −0.26.)
- **근본 원인:** *모든 frozen 프로브는 frozen feature를 상한으로 갖는다.* M2는 인코더를 재학습해
  단일 타일 픽셀에서 *다른* feature를 만드는 것 → frozen 프로브는 인코더의 도달가능 집합을 bound
  못 함. **⇒ frozen 진단으로 M2는 settle 불가. (프로브를 더 만들어도 같은 천장에 갇힘.)**
- **M2 판정 = plausibly-flat / 저-EV / 후순위** (killed 아님, cleanly-tested 아님). 근거: 단일
  feature가 앙상블 정보를 선형적으로 결여 + blend는 정의상 멀티관측 평균 + 프로젝트 전 accuracy
  시도가 flat(§3.9 false-positive). **유일한 settling 실험 = 짧은 실제 M2 학습**(LoRA, ensemble→single
  soft-code distill, single-tile mIoU vs frozen, multi-seed). priors상 저-EV → 지금 실행 안 함.

**D-B (M1 헤드룸) — 확인:** backbone feature 프로토타입(K=64) cross-view 일치 **ARI=0.236**
(same-prototype 2.1%는 K-brittle이라 ARI가 정직한 값), cosine 0.678. frozen 코드는 뷰 불변과 거리가
멂 → M1이 움직일 **실질 헤드룸 존재**. (단 헤드룸 존재 = "M1이 움직일 게 있다"이지 "정확도 이득"
아님 — H2·backbone 전파는 여전히 열림.)

**D-C (의미 상한) — 확인:** 프로토타입 순도 **0.838**(코드가 의미적으로 유의미 → M1 전제 지지),
cluster→majority mIoU 0.269(hard code라 선형 probe 미만, 예상대로).

**종합 판정:**

| | 판정 | 근거 |
|---|---|---|
| 머신 | ✅ 검증 | +0.119 재현(RESULTS §3.1) |
| **M1** (요청 기법) | **GO** | D-B 헤드룸(ARI 0.236) + D-C 의미성(purity 0.84)이 전제 지지; H1 확신 / H2·backbone전파 열림 |
| **M2** (게이팅 확장) | **후순위** | frozen으로 판정 불가; 단일 feature에 선형 부재 + priors → 저-EV; settling = 실제 학습 |

---

## 9. F-track — 타일 예측 × 통합 예측의 적응적 결합 (사용자 제안, 2026-07-02)

**제안:** *"각 패치(타일)에서의 예측과 통합에서의 예측은 다운스트림 작업 모두에 적용 가능하며,
적절한 특징 결합방식이 요구된다."* — 이는 §8 M2 교훈의 정확한 응답이다: 앙상블 이득은 단일 뷰로
**증류 불가**(분산 감소)이므로, 수확하는 유일한 길은 **추론 시 결합을 잘하는 것**. 실측 근거:
blend_fair 0.455 vs single_fair **0.367** — 결합에만 존재하는 **+0.088**이 이 프로젝트에서 정확도가
실제로 움직이는 것으로 확인된 유일한 레버다.

### 9.1 설계 공간 (2축) — 어디가 미탐색인가

기하 축은 **소진됨**(RESULTS §3.8): naive scatter ≈ obliquity(+0.006) ≈ deformable(tie). E2P 대응이
정확해서 "어디를 볼지"는 배울 게 없다. 미탐색은:

| 축 | 소진/미탐색 | 후보 |
|---|---|---|
| **결합 수준(level)** | 미탐색 | feature-mean(1 decode, 효율) vs **prediction/logit-mean**(N decode) vs **agreement-gated selection** |
| **가중(weight)** | 기하만 소진 | uniform/obliquity(소진) vs **content confidence**(미탐색) vs **semantic code**(M1 학습 후) |

**selection의 근거 — D1:** overlap cell의 24–28%가 *모순* 예측을 담는다. 모순 feature의 평균은
"세탁"(정보 손실 가능)이고, 모순 셀에서는 **평균 대신 선택**(가장 신뢰되는 타일)이 나을 수 있다 —
결합 방식들이 실제로 갈리는 곳은 오직 모순 셀이므로, 평가도 모순 셀에서 분리해서 본다.

### 9.2 M1과의 통합 (semantic identity ↔ fusion 의 양방향 결합)

- **M1 → fusion:** M1의 cross-view **code agreement가 결합 게이트의 semantic 신뢰 신호** —
  코드 일치 셀은 평균(안전), 불일치 셀은 선택/다운웨이트. frozen에선 k-means code가 proxy.
- **fusion → M1 평가축:** M1 학습 후 feature 일관성↑ → 평균의 세탁 손실↓ → **feature-level
  fusion이 prediction-level에 근접**할 것으로 예측. frozen vs M1의 fusion-품질 delta가 M1의
  새로운 laundering-proof 평가가 된다.

### 9.3 다운스트림 일반화 (사용자 지적 그대로)

결합 *신뢰 신호*(confidence/code agreement)는 task-공유이고, 결합 *연산*만 task별이다:

| task | 결합 연산 | 비고 |
|---|---|---|
| seg | logit mean / gated selection | F-1이 측정 |
| depth | scale-aligned mean 또는 median | §3.6 pointmap fusion(−22% point gap)과 직결 |
| normal | 평균 후 renormalize | §3.5 cross-tile 일관성 −15%와 직결 |
| pointmap | along-ray fusion (공유 광학중심) | ghosting = depth 불일치 → 같은 신뢰 신호 |

하나의 결합 모듈(신뢰 신호 + task별 연산)이 모든 다운스트림에 적용된다 — 사용자 직관과 일치.

### 9.4 F-1 진단 (frozen bake-off) — 학습 전 실측

`scripts/diag_fusion_bakeoff.py`: frozen DINOv3 + 공유 linear head, eval_ssl과 동일한 ERP-stitch
기저. variants = single / featU(현 blend) / featW(obliquity) / logitU / logitC(confidence) /
gateC(모순→최고신뢰 선택) / gateO(모순→최소경사 선택). 지표 = mIoU **all / seam(cov≥2) /
disagree cells** (결합이 갈리는 곳만 분리).

**판정 기준:** 어떤 variant가 featU를 모순 셀에서 유의미하게 넘으면 → content 축 생존, M1
code-gate로 정밀화할 가치. 전부 동률이면 → 결합은 가중 방식에 둔감(naive field fusion이 그대로
답), 그것도 §3.8과 일관된 정보다.

### 9.5 F-1 결과 (DensePASS, frozen, 30 val 파노, 112s)

머신 검증 ✅: disagree|seam=**0.323** — `eval_ssl` frozen DP disagree 0.323과 정확히 일치.
(cov는 established 프로토콜대로 *기여 수*(같은 타일의 인접 셀 포함) — seam=1.0은 그 의미.)

| variant | all | disagree cells |
|---|---|---|
| single (최소경사 선택) | 0.326 | 0.204 |
| **featU (uniform 평균, 현 blend)** | **0.445** | **0.376** |
| featW (obliquity 가중) | 0.445 | 0.377 |
| logitU (prediction-level 평균) | 0.445 | 0.376 |
| logitC (confidence 가중) | 0.448 | 0.378 |
| gateC (모순→최고신뢰 선택) | 0.374 | 0.278 |
| gateO (모순→최소경사 선택) | 0.326 | 0.204 |

**세 가지 판정:**

1. **가중 축 전멸 (frozen에서).** featU=featW=logitU≈logitC(+0.003, 노이즈 수준). 기하 가중(§3.8)에
   이어 **content 가중(confidence)도 죽었다.** frozen feature에서 "적절한 결합" = **단순 uniform
   평균이 이미 cheap-variant 천장**이며, +0.119 헤드룸은 전부 평균이 수확한다.
2. **selection은 blending에 완패 — 분산감소 가설의 fusion-레벨 확증.** 모순 셀에서 평균 0.376 ≫
   최고신뢰 선택 0.278 ≫ 최소경사 선택 0.204. 타일들이 *모순일 때조차* 평균이 최선 = 뷰별 오류가
   "한쪽이 맞고 한쪽이 틀림"이 아니라 **노이즈적**이라는 것(D-A의 irreducible variance-reduction과
   같은 이야기). 또한 single과 blend는 agree 셀에서 정의상 동일하므로, **+0.119 헤드룸 전체가
   모순 셀(32%)에 산다** — 그리고 거기서 왕은 평균이다.
3. **level 축은 이 셋업으로 판정 불가 (정직한 한계).** 선형 head에선 `head(mean(f)) ≡
   argmax(mean(logits))`가 **수학적 동치**(0.445=0.445는 코드 정합성 확인). feature-level vs
   prediction-level의 진짜 비교는 **비선형 디코더**(예: UPerNet — 타일별 디코드 후 logit 평균
   vs 융합 필드 1회 디코드)가 필요 — 유일하게 열린 축.

**M1 통합에의 정직한 업데이트:** selection이 평균에 지므로 **code-gated *selection/가중*은 강등**
(§9.2의 (ii)). M1이 fusion을 돕는 경로는 (i) — **feature 일관성↑ → 모순 셀에서 평균의 세탁 손실↓ →
frozen 0.376을 넘는 disagree-cell 성능** — 이며, 이것이 M1의 laundering-proof 신규 평가축이다
(frozen vs M1의 fusion 품질 delta, 특히 모순 셀). code agreement의 남은 용도는 가중이 아니라
**불확실성 신호**(downstream에 uncertainty map으로 전달).

**보류:** indoor(F-1@Stanford2D3D)는 val fold **area_5 미다운로드**로 대기(디스크에 area_1/3만).
canonical 프로토콜을 깨는 area-내부 split은 leakage로 기존 수치와 비교 불가라 하지 않는다.
(추후 도착함 — §9.6의 F-2가 우선.)

### 9.6 F-2 — learned set-fusion at scale (사용자 반론 수용, 2026-07-02)

§9.5의 "가중 축 전멸"은 **스칼라 가중 + 작은 규모(60–250 파노)**에서의 반증이었다. 사용자 반론:
(i) S3D 21.8k 라벨 파노 도착으로 규모 전제가 무너짐(§3.8 스스로 "훨씬 큰 규모는 열려 있다"고
남김), (ii) 시험된 함수 클래스가 좁음 — set-함수 융합(평균 너머의 집합 통계: 뷰 간 분산=불확실성,
채널-선택 결합, 재샘플링 blur 복원)은 미반증, (iii) 타일링 구성이 증강 축(ERP roll은 무손실).

**설계** (`fusion.py` + `scripts/train_fusion_f2.py`): 인코더(TC3) **동결**(타일 특징 보존 —
사용자 요구), 학습은 결합에만. `SetFusion`: **fused = uniform_mean + g(set)**, g의 출력층
zero-init → **초기 상태 = 검증된 평균 베이스라인** (학습은 그 위에만 쌓음, 초기 퇴행 불가).
기여별 기하 토큰(obliquity, 중심거리, pitch). 학습: S3D 라벨 파노(scene-disjoint split),
증강 = 랜덤 ERP roll(매 스텝) + hfov 뱅크{63,65,67}. **Paired 프로토콜**: FUSION=mean(평균+head)
vs FUSION=attn(SetFusion+head), 같은 seed/데이터. 판정: mIoU all + **고분산 셀**(뷰 간 특징 분산
상위 30% — 결합이 실제로 갈리는 곳) 분리 보고.

**사전 확률(정직)**: 두 번의 소규모 부정 결과가 있으므로 입증 부담은 "함수 클래스+규모가 결과를
바꾼다" 쪽. 상태: 구현·테스트 완료, GPU 대기.

**결과 (2026-07-03, S3D 3000 학습 / 300 검증 scene-disjoint, paired):**

| val (S3D, 41-cls) | mean (0.03M) | **attn (1.48M)** | Δ |
|---|---|---|---|
| mIoU 전체 | 0.335 | **0.413** | **+0.078** |
| 고분산 셀 (30%) | 0.286 | **0.362** | **+0.076** |
| 저분산 셀 | 0.361 | **0.439** | **+0.078** |

정성 (fix/break, 6 파노): attn이 mean의 오류 **312셀 수정**, 142셀 훼손 — net **+170**
(`docs/figures/viz_fusion_f2/`, run 폴더 viz/에 GT-나란히 샘플).

**판정: 사용자 반론이 옳았다 — 규모(90×) + set-함수 클래스가 F-1의 부정 결과를 뒤집었다.**
학습 융합이 uniform 평균을 +0.078 mIoU로 결정적으로 이김. 이득이 고/저분산 셀에 **균일** —
모순 해소만이 아니라 기하 토큰 조건부의 일반적 feature 정제로 보임. 유보: single-split
single-seed (단 Δ가 측정된 seed-노이즈 ±0.01의 ~8배라 생존 가능성 높음), S3D 실내 한정
(도메인 전이 미검), param 비대칭(1.48M vs 0.03M)은 의도된 비교축. multi-seed de-risk 권장.

### 9.7 F-2 de-risk — +0.078은 선형-head 아티팩트, 실디코더에서 융합이 진다 (2026-07-03, §3.9 세 번째 적중)

§9.6의 +0.078은 **선형 프로브 head** + single-split에서 측정됐다. 실제 dense head(**UPerNet**,
`scripts/fusion_downstream.py`)로 교체하고 **5-seed paired**(covered-cell만 채점, 인코더 TC3 동결,
S3D 400 학습/120 검증 scene-disjoint, 12ep)로 재측정:

| 다운스트림 (UPerNet, 5-seed) | mean (0.03M) | attn (+1.48M) | Δ(attn−mean) | seed 우세 |
|---|---|---|---|---|
| **SEG** mIoU ↑ | **0.398 ± 0.009** | 0.332 ± 0.013 | **−0.065 ± 0.016** | 0/5 (p=0.062) |
| **DEPTH** \|Δlog\| ↓ | **0.131** | 0.141 | **−0.0099 ± 0.0036** | 0/5 (p=0.062) |
| DEPTH δ<1.25 ↑ | **0.857** | 0.831 | −0.026 | — |

**판정: F-2 학습 융합은 실제 디코더에서 두 태스크 모두 mean에게 일관되게 진다.** 결정적으로 attn은
**1.48M 파라미터를 더 쓰고도** 졌으므로 용량 부족이 아니라 학습 융합이 순(net) 해롭다. §9.6의 +0.078은
선형-head가 mean의 저용량을 벌한 **프로브 아티팩트**였다 — 강한 디코더가 집합-통계 이득을 스스로
흡수한다(§3.9 UPerNet laundering 패턴). **§3.9 규율의 세 번째 처형**(§10.6 blend_fair, §9.6 F-2 정확도
힌트에 이어). 다운스트림 융합의 결론 = **uniform masked-mean**. 아티팩트:
`runs/0703_1547_fusion_downstream_seg/`, `runs/0703_1745_fusion_downstream_depth/`
(config + weights + GT-나란히 viz).

---

## 11. F-3 스케치 — Pano-JEPA (masked-view modeling, EMA teacher)

사용자 방향 제시 두 개의 수렴점: "통합 과정이 학습에 없다" + "DINOv3는 파노라마 SOTA가 아닌데
영구 동결이 천장이 된다". §0의 canonical 정식화 옵션 (b)를 목적함수로 승격:

```
타일 1~k개 마스킹 → 보이는 타일 인코딩 → [학습 융합 모듈] → 융합 필드
  → 마스킹된 타일의 warp 위치에서 그 표현을 예측
타깃: EMA teacher (DINOv3+TC3로 init, 천천히 진화 — 영구 동결 아님)
```

한 손실이 포섭: 기하 일치(warp 대응 위 예측) + 의미 일치(표현 공간) + **통합의 학습**(융합이
손실 안) + **전체 파노라마와의 일치**(문맥=나머지 전체) + **DINOv3 천장 탈출 가능**(EMA 진화).
I-JEPA와의 차이: 위치 예측을 배울 필요 없음 — 기하 대응이 정확히 주어짐(파노라마 고유 이점).
anchor 스펙트럼: 영구동결(TC3, 검증) → **EMA(미시험, 데이터 도착으로 가능)** → 무anchor(M1, 반증).
안전장치: laundering-proof 평가 스위트가 EMA 드리프트의 침식 전환을 즉시 탐지. F-2 다음 순서.

### 11.1 F-3 결과 (2026-07-03) — EMA 중간지대는 작은 규모에서 동결(TC3)에 진다

학습(2ep, 406 steps, **고정 pool 810** — TC3 비교용): pred loss 0.63→0.10(목적함수 정상),
drift(vs frozen DINOv3) 0.20→~0.26(EMA 진화 허용), **erank 39.9→~25 압축(침식 경고)**.

| DP@50 헤드룸 | frozen | TC3 | F3-student | F3-EMA |
|---|---|---|---|---|
| 코드 ARI (일관성) | 0.236 | 0.595 | **0.663** | 0.656 |
| **순도 (D-C, 의미)** | 0.838 | **0.862** | 0.821 | 0.830 |
| cluster→majority mIoU | 0.269 | **0.344** | 0.243 | 0.254 |
| single (eval_ssl) | 0.326 | 0.337 | 0.301 | 0.325 |

**판정:** 순도 순위 TC3 0.862 > geo 0.854 > frozen 0.838 > **F3-EMA 0.830 > F3-student 0.821** >
M1 0.730. F-3은 ARI를 높였지만(0.66, TC3 위) 그 일치를 **의미로 지불** — 순도가 frozen 아래로
내려가 사전 등록 기준(≥0.862) 실패. **EMA(0.830)가 student(0.821)보다 덜 침식**된 건 JEPA
"느린 평균이 안정적" 이론의 성립 증거지만, TC3를 넘진 못함. **TC3가 챔피언 유지.**

**정직한 한계(설계 흠):** F-3은 TC3 비교를 위해 **810 파노**에서 학습 — 그런데 EMA 제안의 논거는
"S3D 21.8k 규모가 EMA를 가능케 한다"였다. 즉 F-3이 답한 것은 *"같은 작은 규모에서 freeze vs EMA →
freeze 승"*이고, *"21.8k에서 EMA가 천장을 넘는가"*는 **미검증**. pool 고정이 두 질문을 섞었다.
스케일 가설의 공정한 검증은 F-3을 S3D 21.8k(F-2가 쓴 pool)에서 재학습해야 한다.

**anchor 스펙트럼 최종(현 데이터 기준):** 무anchor(M1, 순도 0.730 = 최악 침식) < EMA(F-3, 0.821–0.830
= 완화된 침식) < **영구동결(TC3, 0.862 = 무침식+개선, 챔피언)**. 스케일↑ 시 EMA 순위가 오를 여지는 열림.

### 11.2 F-3 침식의 근본 원인 — VICReg가 켜져 있었으나 무력 (설계 흠 #2)

로그: **학습 내내 `var=0.000 cov=0.000`** — 붕괴 방지 항이 침식 중에 gradient 0. 분해:
- **분산항 휴면:** `relu(gamma−std)`, gamma=0.04 < DINOv3 자연 std ~0.07 → 파국 붕괴(std<0.04)만
  발화, F-3의 완만한 수축(std는 0.04 위 유지)은 못 봄. gamma가 *바닥*으로 설정됨(TC3/geo에선 distill이
  주 방어라 이게 정상이었음).
- **공분산항 중화:** raw backbone에서 off-diag ~0.005 → `off²합/(d·(d−1))` ~**1e-5**, 가중 1.0 →
  pred(~0.1–0.6) 옆에서 gradient 무시. losses.py 주석의 경고("projector head에 걸어라") 위반.
- **정체 = rank 수축:** erank 39→25 = 채널 상관↑ → 유효차원↓ (rand cos 0.435→0.711). 채널별 std는
  이걸 못 보고, 이걸 막을 **공분산항이 하필 중화**돼 있었음. VICReg 구조는 옳으나 F-3 regime 미조정.
- **근본:** F-3이 distill anchor(TC3/geo의 실제 침식 방어)를 제거하고 방어를 *수동 바닥용* VICReg에
  떠넘김 → 방어 공백. 사용자 지적("VICReg가 이걸 막으라고 있는 것 아니냐")이 정확.

**⇒ 재시험 처방(F-3′):** 스케일만 올리지 말고 VICReg 능동화 — (1) gamma↑(~0.1, 수축에 능동 저항),
(2) 공분산을 정규화 projector에 걸고 canonical sum/D 가중(rank push), (3) 선택적으로 relational
distill 약하게 복원(anchor 공백 메움). 그 후 S3D 21.8k에서 학습해야 스케일 가설의 공정한 검증.

---

## 10. M1 구현·학습·판정 (2026-07-02) — 그리고 v2 (TC3)

### 10.1 구현과 학습

`losses.py`(sinkhorn + code_swap_loss), `encoder.py`(CodeHead K=512), `scripts/train_ssl_m1.py`
(staged: Stage A 기존 레시피 → Stage B code ramp-in + token-distill 1→0.1 anneal, 양방향 warp).
adversarial 리뷰(5렌즈×반박검증)에서 확정 결함 1건(Sinkhorn-균형 때문에 눈먼 붕괴 모니터) 수정.
학습(3ep, 2355 steps, pool 785) 건강: warp 0.054, code CE 6.2→2.84, perp 480+/512, 무NaN.
**단 student erank가 teacher의 ~0.65×로 압축** — 이것이 §10.2의 침식으로 현금화됐다.

**공정 비교를 위해 geometric 베이스라인을 같은 pool·3ep로 재학습**(원 `ckpt_ssl_lora`는 구
데이터 루트와 함께 소실; pool도 달랐음 — RESULTS.md 절대치와 직접 비교 금지).

### 10.2 3-way 평가 (frozen / geo / M1, DensePASS@50)

| 축 | 지표 | frozen | geo | M1 |
|---|---|---|---|---|
| 일관성 | overlap cosine | 0.678 | 0.895 | **0.956** |
| | ret@1 (head-free) | 0.208 | 0.854 | **0.967** |
| | 코드 ARI (D-B, backbone F) | 0.236 | 0.495 | **0.677** |
| 의미 보존 | 프로토타입 순도 (D-C) | 0.838 | **0.854** | 0.730 |
| | cluster→majority mIoU | 0.269 | **0.328** | 0.197 |
| | single / blend mIoU (probe) | 0.326/0.445 | **0.333/0.449** | 0.276/0.318 |
| | sem@1 (head-free NN 의미) | 0.903 | 0.959 | 0.958 |

### 10.3 판정

1. **H2의 전파 조건은 성립** — code 압력은 backbone F에 도달했다(ARI 0.677 최고). §2.1의 열린
   질문의 답은 "전파된다".
2. **H4 반증 발화** — 그 불변성은 의미를 버려서 샀다: 순도 0.854→0.730, probe −0.05~−0.13.
   Sinkhorn은 *코드 사용률* 붕괴만 막고 *의미 희석*은 못 막는다(코드는 다양하되 의미적으로
   임의화). 침식은 정확히 **선형 판독 가능성**에 국한(sem@1 무침식 — NN-매칭 의미는 보존).
   메커니즘: 자가 생성 코드 타깃 = "뭉개면 일치 공짜"(shortcut invariance) + anchor 해제.
3. **가장 아픈 대조 — geo가 '의미적 동일성'을 더 잘 전달한다.** geo는 코드 목적함수 없이 ARI
   0.495를 공짜로 얻으며 순도·cluster-mIoU를 **올렸다**(좋은 경로). M1이 geo 위에 더 얹은
   ARI +0.18은 대부분 나쁜 경로(침식)로 산 것.
4. **진단 맹점 (메타교훈):** D-B "M1 헤드룸"을 frozen 기준으로 측정한 것이 오판의 근원 —
   코드는 feature의 함수라서 **기하 정렬만으로 코드 일치가 대부분 따라온다**. "M1이 무엇을
   더하는가"의 올바른 기준선은 frozen이 아니라 **geo**였다. §8의 M1 GO 판정은 이 맹점 위에
   서 있었다(전제 자체는 참이었으나 비교 대상이 틀림).

### 10.4 v2 — TC3 (Teacher-Code Cross-view Consistency)

침식 채널을 구조적으로 닫는 결합: **geo 레시피 완전 유지**(token anchor full, anneal 없음) +
**teacher-고정 의미 코드**의 cross-view 예측 (`scripts/train_ssl_tc3.py`, geo-init 연속학습).

```
C_t = k-means(frozen teacher 토큰 20만, K=512)   # 고정 — 의미 기준 좌표계
L  += 0.3 · CE( softmax(F_stu_A·C_t/0.1)(p), softmax(F_tea_B·C_t/0.05)(Hp) )  # 양방향
```

- 신규 파라미터 0 (head 없음 → loss가 F에 직접), Sinkhorn 불필요(타깃 고정·다양 → 붕괴 경로 없음).
- blur 침식은 벌점화(뭉개진 feature는 sharp teacher 코드를 못 맞춤).
- **남은 이론적 침식 채널 1개(사전 등록):** *양자화 당김* — feature가 512 centroid로 스냅하며
  within-cluster 변이 소실. full token-distill이 저항해야 하며, 시그니처 = tok 상승 + erank
  비율 <0.7 지속.
- **성공 기준(사전 등록):** D-B ARI > geo 0.495 **이면서** D-C 순도 ≥ 0.854 유지. 정직한 null:
  geo와 동률(student≈teacher → 작은 gradient).

### 10.5 TC3 결과 — 사전 등록 기준 통과 (4-way 최종표)

학습(2ep, 1570 steps, geo-init): 평형 수렴 — warp 0.083, tc3 5.3→4.7, tok ~0.10에서 anchor가
드리프트 저지(양자화-당김 시그니처는 step 600에서 1회 발화 후 해제, 지속 조건 미충족).

| 축 | 지표 | frozen | geo | M1 | **TC3** |
|---|---|---|---|---|---|
| 일관성 | overlap cosine | 0.678 | 0.895 | **0.956** | 0.921 |
| | ret@1 (head-free) | 0.208 | 0.854 | **0.967** | 0.858 |
| | **D-B 코드 ARI** | 0.236 | 0.495 | 0.677* | **0.595 ✓** |
| 의미 보존 | **D-C 순도** | 0.838 | 0.854 | 0.730 | **0.862 ✓** |
| | cluster→majority mIoU | 0.269 | 0.328 | 0.197 | **0.344** |
| | single / blend mIoU | 0.326/0.445 | 0.333/0.449 | 0.276/0.318 | **0.337/0.451** |
| | single_fair / blend_fair | 0.367/0.455 | 0.370/0.450 | 0.322/0.365 | **0.379/0.473** |
| | sem@1 | 0.903 | 0.959 | 0.958 | **0.960** |

(*M1의 ARI 0.677은 침식으로 부풀려진 수치 — 순도 −0.12와 동시 발생)

**판정: TC3는 geo에 대한 엄격한 Pareto 개선.** 일관성 전 지표 ≥ geo, 의미 지표 전부 > geo —
M1이 실패한 "두 동일성의 동시 달성"을 침식 없이 이룸. ARI +0.10(0.495→0.595)은 순도 상승
(0.854→0.862)과 동반 = **좋은 경로의 코드 일치**. blend_fair 0.473(> frozen 0.455)은 어댑터가
앙상블 천장을 올린 첫 관측이나 single-split이므로 multi-seed 전 주장 유보(§3.9 교훈).

**결론:** 침식 채널을 봉쇄한 결합 설계(고정 teacher 코드 + full anchor + 기하 대응)가 정답이었다.
`ckpt_ssl_tc3` = 현 챔피언. **다음:** multi-seed de-risk(→§10.6), indoor(area_5 대기), W_TC3/τ_t
스윕, fusion-delta 재확인(M1 평가축), TC3 코드의 불확실성 신호 활용(§9.5).

### 10.6 Multi-seed de-risk — 핵심 주장 생존, 정확도 힌트 사망 (§3.9 규율)

`scripts/derisk_tc3.py`: 4 seeds × paired(인코더 3종이 seed별 train-subset 50/70·patch 샘플·
head 초기화·k-means를 공유), val 30 파노 고정. seed별 수치가 single-split 참조값 주변에 분포
(머신 정합 ✓).

| 주장 (paired Δ) | mean±std | 부호 일관 | 판정 |
|---|---|---|---|
| **ΔARI tc3−geo** | **+0.117 ± 0.015** | **4/4** | **ROBUST** — 핵심 주장 생존 |
| **Δpurity tc3−frozen** | **+0.029 ± 0.004** | **4/4** | **ROBUST** — 무침식 확정 |
| ΔARI geo−frozen | +0.218 ± 0.026 | 4/4 | ROBUST (geo 특성 재확인) |
| Δpurity tc3−geo | +0.005 ± 0.011 | 3/4 | noise — geo와 동률 (침식 없음이 요점) |
| Δsingle_fair tc3−frozen | +0.003 ± 0.005 | 3/4 | **noise** — 정확도 이득 아님 |
| Δblend_fair tc3−frozen | +0.007 ± 0.015 | 3/4 | **noise** — §10.5의 "앙상블 천장 상승 힌트(+0.018)"는 split 아티팩트 (§3.9 UPerNet 패턴 재연) |

**확정 특성:** TC3 = **의미적 동일성 어댑터** — cross-view 코드 일치를 geo 위로 +0.12 견고하게
올리고(4/4), 의미 순도는 침식 없이 유지(frozen 대비 +0.03 견고). **정확도 어댑터는 아니다** —
single/blend probe 이득은 seed 노이즈로 소멸(프로젝트 불변식 "일관성 ≠ 정확도"가 TC3에도 성립).
유보했던 blend_fair 주장을 de-risk가 정확히 처형 — §3.9 규율의 두 번째 적중.

---

## 12. VICReg 3-role 재설계 (사용자 지시, 2026-07-03) — 능동 var/cov ≠ 침식 방지

배경: loss 7항 누적을 3역할(기하강건성=invariance / 붕괴방지=var+cov / 의미=distill)로 정리,
canonical VICReg 채택. 흠 #2(휴면 var/cov)를 expander+gamma=1로 능동화. 사용자 challenge로 teacher를
role#2로 복귀(SEM 토글), 이어 사용자 지적 2건 추가 수정: inv 채널-정규화(P배 과대 제거), var/cov를
**배치단위**(BATCH개 서로 다른 장면)로. paired: SEM=distill vs none, pinned pool 810.

학습: distill≈none 궤적(sem w=1이 VICReg 25 옆에서 무력). var축 작동(std→1), **cov축 실패**
(cov 1.1→3.75 상승 = rank 수축 방치).

| DP@50 | frozen | TC3 | VICReg-distill | VICReg-none |
|---|---|---|---|---|
| 순도(D-C) | 0.838 | **0.862** | 0.753 | 0.728 |
| single(eval_ssl) | 0.326 | 0.337 | 0.254 | 0.239 |
| cluster→majority mIoU | 0.269 | **0.344** | 0.183 | 0.163 |
| ARI | 0.236 | 0.595 | 0.470 | 0.312 |

**판정:**
1. **사용자 VICReg 가설("능동 var+cov만으로 침식 방지") 반증** — none 순도 0.728은 8개 변형 중
   최악, frozen 0.838보다 한참 아래. var축은 작동하나 cov축이 과소가중(ν=1)이라 rank 수축을 못 막음.
2. distill(w=1)은 미미하게 도움(순도 0.728→0.753, single +0.015)이나 TC3 근처도 못 감 — anchor가
   **지배적 가중**이어야 하는데 1은 VICReg 25 옆에서 무력.
3. **anchor-강도 thesis 5번째 확인**(순도 순위): TC3 0.862 > geo 0.854 > frozen 0.838 > F3-EMA
   0.830 > F3-student 0.821 > **VICReg-distill 0.753 > M1 0.730 > VICReg-none 0.728**. 침식 없는
   유일한 변형은 TC3(지배적 distill). 약화시킨 전부(M1 anneal·F-3 제거·VICReg w=1) 침식.
   **능동 var/cov는 강한 semantic anchor를 대체하지 못한다** — var/cov는 자명한 붕괴(상수·저rank)를
   막지만 "유효하나 덜 의미적인 표현으로의 드리프트"는 못 막고, 의미를 붙잡는 건 teacher anchor뿐.

**메타(정직):** "3역할로 정리"의 canonical 가중(25/25/1 + sem 1)이 이 프로젝트의 핵심 자산(강한
teacher anchor)을 과소가중했다 — 깔끔함이 성능을 샀다. 사용자의 "teacher 빼지 말라" challenge가
옳았고, 약하게 유지한 것으로도 부족했다. **TC3가 챔피언 유지.**

**확정 결론(SSL 표현 축):** 이 프로젝트의 label-free pano 규모에서 SSL 어댑터는 **일관성을 개선하되
정확도는 teacher에 묶이며, 강한 teacher anchor 없이는 의미가 침식된다.** TC3(고정 teacher 코드 +
지배적 distill + 기하 consistency)가 침식 없는 유일한 레시피. **다운스트림 융합은 실디코더(UPerNet)
5-seed에서 uniform masked-mean이 학습 융합(F-2)을 SEG/DEPTH 모두 이긴다 — §9.6의 +0.078은 선형-head
아티팩트였다(§9.7).** 이 규모에서 label-free SSL의 실질 이득은 표현·융합 어느 쪽에서도 크지 않으며,
열린 상향은 스케일 사전학습(21.8k, 진행 중)과 anchor 진화(F-3 EMA 규모)다.

---

### 부록 A — 용어

- **geometric identity:** 같은 ray → 같은 raw feature (현재 warp).
- **semantic identity:** 같은 ray → 같은 잠재 의미 코드 (`(φ,N)` 불변, 본 문서).
- **M1:** overlap semantic-code agreement (핵심 기법).
- **M2:** ensemble→single 정확도 전이 (D-A 게이팅 확장).
- **보존 vs 재구조화:** distill anchor 유지(정확도 보존·천장) vs 완화(재구조화·망각 위험)의 trade.
