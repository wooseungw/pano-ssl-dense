# Pano-SSL-Dense — 연구 로그 (여정)

프로젝트가 실제로 어떻게 전개됐는지의 시간순 기록 — 던진 질문, 가설, 막다른 길, 전환점,
그리고 메타 교훈. 정제된 수치는 `RESULTS.md`를 보고, 이 파일은 *과정*을 담는다.

---

## Phase 0 — Thesis & 프레이밍

**목표:** frozen 평면 인코더(DINOv3)를, 인접 E2P 타일의 overlap을 **공짜·기하학적으로 정확한·
label-free SSL 신호**로 써서 파노라마 특징 추출기로 전이한다. **왜 scratch가 아닌가:** 파노
데이터는 DINOv3 규모로 사전학습하기엔 너무 희소 → 처음부터 학습이 아니라 **LoRA-light로 적응**.
**핵심 베팅:** 왜곡 + 타일링 필요 + overlap을 문제가 아니라 **장점**으로 승화.

초기 하드 제약: 적응은 평면 teacher의 의미 + 관계 구조를 **보존**해야 한다(데이터가 희소해
다시 배울 수 없음) → distillation anchor가 co-primary.

---

## Phase 1 — 학습 *전* 진단 (이게 모든 걸 좌우했다)

바로 학습하지 않고, frozen 인코더가 이미 뭘 하는지부터 측정했다.

1. **Seg probe, ERP-direct vs E2P-pinhole** (frozen DINOv3 linear probe): E2P de-distortion이
   도움되고, 그 이득은 **왜곡 심각도에 비례** — outdoor(DensePASS) +0.05~0.10 mIoU, indoor
   (Stanford2D3D) ≈ 0. "왜곡"이 핵심 축이라는 첫 신호.
2. **FOV 스윕:** 최적 perspective FOV ≈ **outdoor 50° / indoor 65°** (기본 90° 아님).
3. **백본 비교:** DINOv3 ≫ DINOv2-reg > DINOv2 ≫ CLIP ≈ PE(SAM-3 인코더) ≫ SAM-1. DINO의
   self-supervised dense feature가 frozen linear probe에 독보적 → DINOv3를 백본/teacher로 채택.
4. **D1 seam 진단 (전환점):** frozen 타일들이 overlap cell의 ~24~28%에서 *서로 다른* 예측 —
   그런데 재학습된 linear head가 그걸 **세탁**한다(blend된 overlap 특징이 single-tile만큼 분류됨).
   ⇒ **linear-probe mIoU는 cross-tile 불일치를 못 본다.** 이후 거의 모든 게 이 경고를 존중해야
   했다: *기본 메트릭이 SSL 효과를 가린다.*
5. **헤드룸의 FOV 민감도:** 불일치는 좁은 최적 FOV에서 *더 크고*(obliquity 아티팩트 아님),
   overlap **ensemble이 single tile보다 +0.119 mIoU** 높다(DP@50). 이 ensemble 헤드룸이
   세탁-불가 메트릭으로 측정할 타깃이 됐다.

**Phase 1 결론:** SSL을 재학습 head가 세탁 못 하는 메트릭(head-free retrieval, cross-tile
consistency, single-tile-vs-ensemble)으로 평가한다. pooled mIoU 아님.

---

## Phase 2 — SSL 구현 & 디버깅

`encoder.py`(LoRA), `losses.py`(warp-equivariance + distill + VICReg), `geometry.py`(convention-free
coordmap warp), `train_ssl.py` 구현. 학습을 신뢰하기 *전에* 감사하다 두 개의 비자명한 버그를 잡음
(기존 메모리는 DINOv2 시절 것):

- **DINOv3 attention 모듈은 LLaMA식 `q_proj/v_proj`** — DINOv2의 `query/value` 아님 → peft가
  아무것도 못 잡음(0 trainable params). 자동탐지로 수정.
- **`combined_loss`가 `gamma`를 받지만 VICReg에 전달 안 함** → VICReg가 γ=1.0로 동작, raw
  DINOv3(채널별 std ~0.07)에선 영구 포화 → **warp 신호를 압도**(warp 평탄 ~0.4, distill 상승,
  erank 붕괴). γ 전달 + **γ=0.04(VICReg를 붕괴 *바닥*으로, 타깃 아님)** 로 수정.

수정 후 **건강한 학습** — warp 0.31→0.12(overlap cosine 0.69→0.88), distill 유계 ~0.06(teacher
보존), student effective-rank가 teacher를 추적(무붕괴).

---

## Phase 3 — 전환점: 일관성 ≠ 정확도

Frozen vs LoRA, 세탁-불가 평가:

- **Semantic(single-tile mIoU): NULL** — DP 0.326→0.338, S2D3D 0.577→0.576.
- **Feature 일관성: 크고 일반화됨** — held-out overlap cosine 0.68→0.91(out), 0.72→0.88(in).

"distill anchor가 막았다" 가설을 반증(특징이 일관성 방향으로 *크게* 움직임), 프로젝트 핵심
발견을 확립: **overlap-SSL은 cross-view *일관성* 어댑터지 *정확도* 어댑터가 아니다.** 정확도는
teacher의 정보량에 묶여 있고(distill이 보존), linear head가 일관성 이득을 세탁한다. 가치는
일관성이 *곧 산출물*인 곳에서 찾아야 한다.

---

## Phase 4 — 일관성의 특성 규명

- **B — head-free(세탁할 head 없음):** dense correspondence **retrieval@1 0.21→0.86(out),
  0.25→0.62(in)**; lift(corr−rand) +0.16/+0.08(대응-특이적, collapse 아님). frozen DINOv3는
  "pose 없이 대응 자가복원 불가"인데 어댑터가 고침.
- **A — geometric(surface normal):** 각도 **정확도 평탄**(57.25°→57.84°), **cross-tile 일관성
  −15%**(35.0°→29.9°). head 있는 geometric 과제에서도 같은 패턴.
- **cosine 너머(사용자 제안):** **Hungarian@1**(엄격 1:1) 0.27→0.87 / 0.31→0.66(retrieval이
  greedy 아티팩트 아님); **CKA**(구조적, 회전 불변) outdoor 0.90→0.92(이미 높음) but **indoor
  0.52→0.83(+0.31)** — cosine이 놓친 결: indoor view들이 *구조적으로* 어긋나 있었고 SSL이 정렬.

---

## Phase 5 — 시각화

figure 모음(`viz_*`) 제작: correspondence 매칭 라인(4/14→14/14 시각화), corr/rand 히스토그램
(lift), PCA feature 파노라마(seam coherence), normal 불일치 히트맵, 캡스톤 "일관성↑ vs 정확도
평탄". 피드백으로 자기설명형으로 재작업(입력 타일 A/B, ERP 발자국, overlap 영역 음영, 지표를
overlap 위에 오버레이).

---

## Phase 6 — SOTA & 효율 현실점검

- **vs SOTA 정확도:** trainable 디코더로 frozen+dec 0.496, scratch-LoRA 0.510, **SSL-LoRA 0.513**,
  공개 SOTA 0.53~0.60(full-res, full fine-tune, UDA) 대비 아래. 의도적 경량 셋업이고, **SSL-init이
  scratch-LoRA와 동률** — 일관성≠정확도가 end-to-end에서도 유지.
- **Params/FLOPs:** trainable **2.95M**(0.59M LoRA + 2.36M decoder) vs SOTA full-train 25~85M →
  param-효율 적응. 단 **~108 GMAC/tile × ~22 tile ≈ 2.4 TMAC/ERP** → 타일링×ViT-B는 FLOPs 무거움.
  정확도가 아니라 효율이 정직한 약점 축.

---

## Phase 7 — Pointmap fusion (커밋된 DUSt3R식 과제)

E2P 타일은 광학중심 공유(parallax 없음) ⇒ 같은 점이 *같은 ray*로 보임 ⇒ fusion ghosting =
depth 불일치. Depth probe + back-projection: **정확도 평탄**(0.195→0.189 log-err), **cross-tile
depth 일관성 −14%**, **overlap-pair point gap −22%**. 일관성 어댑터가 멀티타일 pointmap을 더
coherent하게 융합 — 같은 이야기, geometric 버전.

---

## Phase 8 — Adaptive field fusion & deformable transformer

질문: overlap 타일 특징을 하나의 ERP 필드로 융합해 *한 번만* 디코드(효율), 각 위치를 *적응적*으로
채울 수 있나.

- **해상도가 진짜 레버였다(정정).** 일관 프로토콜 스윕(frozen, naive scatter, 150 파노): 32×64=
  **0.550**(coverage 0.87), 64×128=**0.557**(0.82, peak), 128×256=**0.550**(0.49). **64×128이
  sweet spot**(타일 패치 밀도 ~2°/cell와 일치); 더 키우면(128×256) 타일(32×32 패치)보다 격자가
  촘촘해 **coverage 0.49로 붕괴(hole)** → mIoU 하락. (한때 "0.40→0.57 +0.13"으로 본 건 서로 다른
  프로토콜·파노 수를 비교한 아티팩트였고, 같은 프로토콜에선 해상도 효과는 작다 +0.007.) 64×128
  필드(~0.557, per-tile pooled ~0.58에 근접)가 **단일 디코드 효율 win**.
- **adaptive obliquity 가중**(왜곡 큰 가장자리 패치 down-weight) > naive 균등 by 작은 **+0.006**.
- **geometry-guided deformable cross-attention**(학습 offset + 타일 over attention): naive와
  **동률**. 60 파노에선 과적합으로 패배(−0.03); 250 파노 + 정규화 공정 재시도 → ±0.001.
  **핵심 통찰:** E2P overlap은 *정확하고 parallax 없는* 대응이라 기하 reference가 이미 픽셀
  단위로 정확 → deformable의 "어디를 볼지 학습"이 고칠 게 없다. 학습 융합은 대응이 *불확실*할 때
  (parallax multi-view, cross-frame)나 훨씬 큰 규모에서만 값을 한다.
- **LoRA ≈ frozen** 또(일관성 ≠ 정확도).

---

## 메타 교훈 (수치가 아니라 방법론)

1. **학습 전에 진단하라.** D1 세탁 발견이 평가 전체를 재설계했고, 기본 메트릭의 false-negative
   ("SSL은 효과 없음")를 막았다.
2. **물려받은 가정을 감사하라.** run을 망칠 조용한 버그 둘(DINOv3 모듈명, γ 미전달)은 메모리를
   믿는 대신 코드를 감사해서 잡았다.
3. **축을 분리하라.** "일관성"과 "정확도"는 다른 축; 재학습 head로 둘을 뭉뚱그리면 진짜 기여가
   가려진다. 이 프로젝트 전체가 *올바른 축을 측정하는* 연습이다.
4. **동률도 정보다.** deformable 재시도, fine-tune SSL-vs-scratch, LoRA-vs-frozen 동률들은 모두
   같은 경계를 확인했다: 어댑터는 일관성을 움직이지 정확도를 움직이지 않는다.
5. **정직한 negative들이 모여 thesis가 된다.** semantic null + head-free win + geometric 일관성 +
   fusion + deformable 동률 → 하나의 견고하고 방어 가능한 특성 규명.

---

## 산출물

스크립트: `probe_seg_dinov3.py`, `diag_seam.py`, `diag_cos.py`, `diag_overlap_match.py`,
`diag_consistency_metrics.py`, `train_ssl.py`, `eval_ssl.py`, `probe_normal.py`,
`pointmap_fusion.py`, `fine_tune_seg.py`, `adaptive_field.py`, `adaptive_field_deform.py`,
`field_res_sweep.py`, `viz_*.py`. 어댑터: `ckpt_ssl_lora/`. 결과+figure: `RESULTS.md`,
`docs/figures/<script>/*.png`. 메모리: `memory/probe-seg-dinov3.md`(+ thesis/loss/geometry 노트).
