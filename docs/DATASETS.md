# Datasets — pano-ssl-dense

360°/ERP 데이터셋 인벤토리 + 취득 방법. 작성 2026-06-17.
대상 task: semantic seg · object detection · panoptic · DUSt3R pointmap · (+ depth · normal) · **region-grounding (CLIP-style image-text)**.

루트: `/data/1_personal/4_SWWOO/`. `/data`는 상시 빠듯함 — 받기 전 항상 용량 확인.

---

## 1. 보유 + 즉시 사용 가능 (on disk)

| 데이터셋 | 경로 | 규모 | GT / 용도 | 로더 |
|---|---|---|---|---|
| **Stanford2D3D** ⭐primary | `stanford2d3d/extracted_data/area_*/*/pano/` | 1155 ERP @4096×2048 | semantic·depth·normal·**global_xyz(pointmap)**·pose | `data.list_erps("stanford2d3d")`, `data.s2d3d_gt_path()` |
| **Structured3D** (scale) | `structured3d/images/Structured3D/scene_*` | **3,500 scene / 505,705 file** (추출 완료) | depth·normal·semantic (per room, `{empty,simple,full}`) + `bbox.zip`(3D bbox 보존) | `data.list_structured3d()`, `data.s3d_gt_path()` |
| **refer360 / quic360** | `refer360/data/quic360_format/` | 3,194 img / 7929·1065·5349 (train·val·test) | **region-grounding** image-text (34 카테고리 + 설명) | `data.list_grounding(split)` |
| **ZInD** | `zind/` | ~70k indoor | room layout | (TODO) |
| **DensePASS** | `densepass/DensePASS/` | 100 real outdoor (400×2048) | **outdoor** semantic, 19-cls Cityscapes trainId | `data.list_densepass()`, `data.densepass_gt_path()` |
| **HF-cache benchmarks** | (HF cache) | small | QuIC-360 · ODI-Bench · OSR-Bench · PanoEnv (eval) | — |

⚠️ Structured3D: 42 scene(1.2%)에 CRC 손상 파일 → 로더에서 decode 실패 시 try/except skip.

---

## 2. 지금 받을 수 있음 (available + 방법)

### PanoX  — ⏳ 다운로드 중 (2026-06-17, `panox/download.log`)
- HF `KevinHuang/PanoX` (public, **Apache-2.0**), **345 GB**, zip 8개 (합성 씬팩: ClothingStore, Diner50, FeudalJapan_Meshingun, ModularBakeryShop, Mountain_Environment_Set, SouthOfFrance, Warehouse, YellowParrot). 각 씬당 다수 프레임(train split). ERP multimodal(PBR/depth/normal류), indoor+outdoor. 논문 OmniX(arXiv 2510.26800). ⚠️ scene 8개 = 다양성 제한, modality는 첫 zip 추출 시 확정.
- **방법 (huggingface-cli 미설치 → wget 직접):**
  ```bash
  cd /data/1_personal/4_SWWOO/panox
  base=https://huggingface.co/datasets/KevinHuang/PanoX/resolve/main
  for f in ClothingStore Diner50 FeudalJapan_Meshingun ModularBakeryShop \
           Mountain_Environment_Set SouthOfFrance Warehouse YellowParrot; do
    wget -c "$base/$f.zip"
  done
  wget -c "$base/PanoX_filelist.json"
  ```
  - (대안) `pip install -U huggingface_hub && huggingface-cli download KevinHuang/PanoX --repo-type dataset --local-dir panox`

### Matterport3D (MP3D)  — ✅ 접근 승인됨 (2026-06-22), 아직 미다운로드
- ~90 building / ~10,800 파노(skybox). real 대규모 실내 + cross-view. RGB-D + semantic·instance(mesh) + camera pose. License: **MP3D Terms of Use** (다운로드 = 동의로 간주).
- **스크립트 / 스캔 목록:**
  - 다운로드 스크립트: `http://kaldir.vc.cit.tum.de/matterport/download_mp.py`
  - house(scan) id 목록: `http://kaldir.vc.cit.tum.de/matterport/v1/scans.txt`
- **사용법:**
  ```bash
  python download_mp.py -o <out_dir>                              # 전체(1.3TB) — ⚠️ 디스크 부족
  python download_mp.py -o <out_dir> --id 17DRP5sb8fy            # 특정 house만
  python download_mp.py -o <out_dir> --type matterport_skybox_images   # 특정 타입만(권장)
  ```
- **우리(pano-SSL)에 필요한 `--type` (전체 1.3TB 대신 이것만):**
  - `matterport_skybox_images` — 6면 skybox 큐브 → ERP 파노 구성 (핵심 RGB 소스, 전체 ~270G)
  - `region_segmentations` / `house_segmentations` — semantic·instance (mesh+region; 픽셀단위 ERP 라벨은 mesh에서 렌더/투영 필요)
  - `undistorted_normal_images` · `undistorted_depth_images` — normal·depth
  - `matterport_camera_poses` — skybox 스티칭/정렬용
- ⚠️ **실행 전 필수 GOTCHA:**
  - **스크립트는 python2 전용** — 이 환경에 python2 없음 → ① 다른 곳에서 python2 실행, or ② py3 포팅(`print` 함수화 / `urllib`→`urllib.request` / `raw_input`→`input`, 3곳).
  - **전체 1.3TB는 현재 184G에 안 들어감** → 반드시 `--type matterport_skybox_images`(+라벨) + `--id`로 서브셋.
  - **Habitat용 MP3D**(.glb mesh)는 `--type matterport_mesh` 또는 habitat 전용 zip → habitat-sim으로 무한 ERP/depth/pose/semantic 렌더 (아래 Habitat 항목과 연계).

### Puffin-4M  — 비추 (받지 않음)
- HF `KangLiao/Puffin-4M` (**449 GB**, NTU S-Lab License 1.0). 실은 **perspective 이미지+camera triplet**(ERP 파노 아님) → 평면 teacher 영역과 중복, 과대용량. 기록만.

### 기타 공개 (필요 시, 미검증 용량)
- **OmniHorizon** (2023) — 24,335 synth, depth+**normal** (normal task용). 프로젝트 페이지서 다운로드.
- **360-in-the-Wild** (2024) — 25K real, depth+view-synth.
- **OmniVQA** (2025) — 1,213 ERP + 4,852 VQA (grounding/reasoning eval, 소형).
- **PanoWorld** (github wcpcp/PanoWorld), **DA-2 "Depth Anything in Any Direction"** (github EnVision-Research/DA-2) — 360 depth 모델/데이터.

**공개 HF 데이터 일반 받는 법:**
```bash
# 직접(wget): https://huggingface.co/datasets/<repo>/resolve/main/<file>
# CLI:        huggingface-cli download <repo> --repo-type dataset --local-dir <dir>
# git-lfs:    git lfs install && git clone https://huggingface.co/datasets/<repo>
```

---

## 3. 대기 / gated (요청·라이선스 필요)

| 데이터셋 | 상태 | 비고 |
|---|---|---|
| **Matterport3D (MP3D)** | ✅ **접근 승인됨 (2026-06-22) → §2에 다운로드 방법** | ~10,800 파노, RGB-D+semantic+multi-view pose. real 규모+cross-view |
| **Habitat: HM3D / Gibson / Replica** | 라이선스 동의 + habitat-sim 렌더 | 무한 ERP 렌더 + exact depth/pose/semantic. real↔synth gap |
| **Dense360** ⭐best-fit grounding | ❌ **미공개** (github/HF 없음) | 160K ERP + 5M entity 캡션 + 1M referring + entity 마스크. arXiv 2506.14471. 릴리스 추적 |
| **ReplicaPano** | ❌ repo "download: TODO" | 2,700 real 파노, depth+layout+3D bbox. 저자 문의 |
| **Outdoor: DensePASS / SynPASS** | 필요 시 | quic360에 야외 일부 있어 *seg 라벨* 필요할 때만 |

---

## 4. 디스크 메모

- `/data` (3.7T) 상시 90%+ 근접. 받기 전 `df -h /data`.
- **Structured3D 파노라마 zip 삭제됨(2026-06-17, 211G).** 추출물(`images/`, 3500 scene)은 그대로, `bbox.zip`/`annotation_3d.zip` 보존. ⚠️ 재취득은 신청 폼 필요(공식 URL 404).
- **추출 시 `unzip` 금지** (Info-ZIP 6.00 = Zip64 >4GB 버그, 68% 조용히 누락). **`7z x` / bsdtar / python zipfile 사용.**
- 정리 가능 비-pano(타 프로젝트): `actiondetect/` 199G (PDAN 89G+detect 88G+YOWOv3 23G), `GREAT_code/`. (OV3DHOI 85G는 2026-06-17 삭제됨.)
