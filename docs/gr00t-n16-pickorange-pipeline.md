# LightwheelAI/leisaac-pick-orange 데이터셋으로 GR00T N1.6 파인튜닝 — 전체 가이드

이 문서는 처음 보는 사람도 다른 컴퓨터에서 그대로 따라할 수 있게 모든 단계를 빠짐없이 적은 가이드다. 학습은 Linux + 데이터센터 GPU(예: H100)에서, 평가는 Windows + RT 코어 있는 GPU(예: RTX A4000)에서 진행하는 분리 운영을 가정한다.

## 0. 무엇을 만드는가

- **입력**: Hugging Face 의 `LightwheelAI/leisaac-pick-orange` 데이터셋 (LeRobot v2.1 포맷, 60 episodes, 36 293 frames, 30 fps, dual camera, 6-dim action/state, 작업: "Grab orange and place into plate")
- **모델**: NVIDIA `GR00T-N1.6-3B` (3B parameter VLA, diffusion 정책 헤드)
- **결과물**: SO-ARM101 로봇이 LeIsaac PickOrange 시뮬 환경에서 오렌지를 집어 접시에 올리도록 동작하는 파인튜닝된 정책 체크포인트
- **소요 시간** (대략): 데이터 변환 ~15 분 + 학습 ~1 시간 + 평가 셋업 ~10 분

## 1. 사전 요구사항

### 1.1 하드웨어

#### 학습 머신 (서버)
- **NVIDIA GPU**: VRAM 24 GB 이상 (H100 80 GB, A100 40/80 GB, RTX 4090 24 GB, RTX 5090 32 GB, RTX 6000 Ada 48 GB 등)
  - **데이터센터 GPU(H100/A100) 도 학습엔 문제없다.** 평가에서만 RT 코어가 필요.
- **디스크**: 최소 100 GB 여유
  - 베이스 모델 가중치 ~6 GB
  - 데이터셋 원본 + H.264 변환본 ~1.5 GB
  - 체크포인트 ~14 GB × 5 개 보존
- **RAM**: 32 GB 이상

#### 평가 머신 (클라이언트)
- **NVIDIA GPU with RT 코어**: RTX A4000/A5000/A6000, RTX 6000 Ada, GeForce RTX 30/40/50 시리즈, L40/L40S/L4, A40/A30
  - **주의**: H100, A100 은 RT 코어가 없어서 Isaac Sim 카메라 sensor 가 동작하지 않는다 (`vkCreateRayTracingPipelinesKHR` 실패). 학습 머신과 평가 머신을 분리하는 이유.
- **VRAM**: 8 GB 이상 (Isaac Sim + 정책 클라이언트 호출만)

#### 네트워크
- 학습 머신과 평가 머신이 같은 사내망에 있어야 함 (학습 머신의 5555/tcp 포트가 평가 머신에서 닿아야 함).

### 1.2 OS / 소프트웨어

#### 학습 머신
- **OS**: Linux (Ubuntu 22.04 / 24.04 권장)
- **NVIDIA Driver**: 그래픽스 라이브러리 포함 풀 인스톨 권장. 만약 `--no-opengl-files` 로 설치된 compute-only 환경이면 7.4 절 트러블슈팅 참조.
- **Python**: 3.11 (`uv sync` 가 자동 설치)
- **uv** (Python 환경/의존성 관리자): https://docs.astral.sh/uv/getting-started/installation/
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **ffmpeg** (av1 → h264 변환용)
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```
- **git**

#### 평가 머신
- **OS**: Windows 11 (Isaac Sim 5.1 Windows 빌드 기준)
- **NVIDIA Driver**: GeForce/Studio 또는 NVIDIA RTX Enterprise 드라이버 (그래픽스 풀 스택 기본 포함)
- **Python**: 3.11
- **uv** (Windows 설치):
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **git**

### 1.3 계정

- **Hugging Face 계정** — 데이터셋·베이스 모델 다운로드. 가입 후 https://huggingface.co/settings/tokens 에서 read 토큰 발급.
- **Weights & Biases (wandb) 계정** — 학습 로깅 (무료). https://wandb.ai/authorize 에서 API key 발급.
- **GitHub 액세스** — Isaac-GR00T 클론.

## 2. 저장소 셋업 (학습 머신)

### 2.1 본 저장소 클론 + 환경 동기화

```bash
git clone <이 저장소 URL> robotics_manipulation
cd robotics_manipulation
uv sync
```

`uv sync` 가 `pyproject.toml` 과 `uv.lock` 에 정의된 모든 의존성 (Isaac Sim 5.1, Isaac Lab 2.3, leisaac, lerobot, h5py < 3.16, torch 2.7.0+cu128 등) 을 `.venv/` 에 설치한다. 최초 실행 시 Isaac Sim 번들 다운로드로 수 GB 가 발생할 수 있다.

설치 검증:

```bash
uv run python -c "import isaacsim, lerobot, leisaac, h5py; \
print('isaacsim', isaacsim.__version__); \
print('h5py HDF5', h5py.version.hdf5_version)"
```

`h5py HDF5` 가 `1.14.x` 로 출력되면 Isaac Sim ABI 와 호환된 상태. `2.x` 가 나오면 README 의 `h5py와 Isaac Sim의 HDF5 ABI 불일치` 섹션을 참고해 `uv add "h5py<3.16"` 로 다운그레이드.

### 2.2 Hugging Face 로그인

```bash
uv run hf auth login
```

발급한 read 토큰 입력. 데이터셋과 베이스 모델 다운로드에 사용.

### 2.3 Isaac-GR00T 클론

GR00T 학습 코드는 NVIDIA 의 별도 저장소에 있다. 본 저장소 루트에 함께 둔다.

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
git checkout e8e625f4f21898c506a1d8f7d20a289c97a52acf
cd ..
```

`e8e625f` 는 LeIsaac 공식 문서가 GR00T N1.6 대상으로 명시한 commit hash. 이걸로 고정해야 LeIsaac 클라이언트와 인터페이스가 일치.

### 2.4 GR00T 의존성 설치 (`--no-deps` 가 핵심)

기존 `.venv` 의 torch 2.7.0+cu128 / Isaac Lab / leisaac 가 GR00T 가 끌고 오는 다른 버전 torch 에 의해 깨지지 않도록 **`--no-deps`** 로 설치한다.

```bash
uv pip install -e ./Isaac-GR00T --no-deps
```

GR00T import 시 빠진 라이브러리를 만나면 그때 추가 (`--no-deps` 유지):

```bash
uv pip install peft accelerate diffusers wandb tyro --no-deps
```

검증 — 다음이 `OK` 출력하면 통과:

```bash
uv run python -c "from gr00t.experiment.launch_finetune import main; print('OK')"
```

만약 추가로 빠진 모듈이 보고되면 같은 방식으로 (`uv pip install <name> --no-deps`) 보충.

### 2.5 wandb 로그인 (1회)

```bash
uv run wandb login
```

API key 입력 (https://wandb.ai/authorize 에서 복사).

## 3. 데이터셋 다운로드 (학습 머신)

저장소 루트에서:

```bash
hf download --repo-type dataset \
  LightwheelAI/leisaac-pick-orange \
  --local-dir ./datasets/leisaac-so101-pick-orange
```

다운로드 후 디렉토리 (~700 MB):

```
datasets/leisaac-so101-pick-orange/
├── data/chunk-000/
│   ├── episode_000000.parquet
│   └── ... (총 60개)
├── meta/
│   ├── episodes.jsonl          ← v2.1 포맷 (GR00T 호환)
│   ├── episodes_stats.jsonl
│   ├── info.json
│   └── tasks.jsonl
├── videos/chunk-000/
│   ├── observation.images.front/
│   │   └── episode_000000.mp4 ... (60개, av1 codec)
│   └── observation.images.wrist/
│       └── episode_000000.mp4 ...
└── README.md
```

데이터셋 핵심 정보:
- **codebase_version**: `v2.1` (GR00T 가 요구하는 포맷. v3.0 이면 변환 필요했지만 이 데이터셋은 처음부터 v2.1 이라 그대로 사용 가능)
- **action / observation.state**: float32, shape (6,), names = `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`
- **observation.images.front / .wrist**: 480 × 640, av1 codec, 30 fps
- **task**: `"Grab orange and place into plate"`

## 4. 데이터셋 GR00T 호환 변환 (학습 머신)

데이터셋 비디오가 av1 코덱이라 GR00T 데이터 로더(Decord)가 디코딩하지 못하는 경우가 있다. **H.264 로 재인코딩한 복사본** 을 만들면서 GR00T 가 요구하는 `modality.json` 을 자동으로 함께 작성한다. 원본은 그대로 보존.

### 4.1 변환 스크립트

본 저장소에 `scripts/convert/transcode_av1_to_h264.sh` 가 들어있다 (없다면 4.4 절의 내용으로 직접 작성). 이 스크립트는:

1. `datasets/leisaac-so101-pick-orange/data`, `meta`, `README.md` 를 `datasets/leisaac-so101-pick-orange-h264/` 로 그대로 복사
2. `videos/**/*.mp4` 를 `ffmpeg -c:v libx264 -crf 23 -preset fast -pix_fmt yuv420p` 로 재인코딩하여 같은 트리 구조에 저장
3. `meta/info.json` 의 `"video.codec": "av1"` → `"h264"` 갱신
4. `meta/modality.json` 을 dual-cam 매핑으로 작성

### 4.2 실행

```bash
chmod +x scripts/convert/transcode_av1_to_h264.sh
./scripts/convert/transcode_av1_to_h264.sh
```

소요 시간: 약 10–15 분 (CPU 코어 수에 따라). 진행 중에는 ffmpeg 로그가 거의 안 보이지만 (`-loglevel error`) 정상.

### 4.3 결과 검증

#### 4.3.1 디렉토리 구조

```bash
ls datasets/leisaac-so101-pick-orange-h264/meta/
# 출력에 다음이 모두 보여야 한다:
# episodes.jsonl  episodes_stats.jsonl  info.json  modality.json  tasks.jsonl
```

#### 4.3.2 비디오 코덱

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name \
        datasets/leisaac-so101-pick-orange-h264/videos/chunk-000/observation.images.front/episode_000000.mp4
# 출력: codec_name=h264
```

#### 4.3.3 modality.json 내용

```bash
cat datasets/leisaac-so101-pick-orange-h264/meta/modality.json
```

다음과 일치해야 한다:

```json
{
  "state":      {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
  "action":     {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
  "video":      {"front": {"original_key": "observation.images.front"},
                 "wrist": {"original_key": "observation.images.wrist"}},
  "annotation": {"human.task_description": {"original_key": "task_index"}}
}
```

이 매핑의 의미: LeRobot 데이터셋의 컬럼명 (`observation.images.front`, `task_index` 등) 을 GR00T 의 modality 키 (`video.front`, `annotation.human.task_description` 등) 로 변환.

#### 4.3.4 데이터 로더 실제 로딩 테스트

```bash
cd Isaac-GR00T
uv run python scripts/load_dataset.py \
  --dataset-path ../datasets/leisaac-so101-pick-orange-h264 \
  --plot-state-action
cd ..
```

state/action 6 차원 곡선 plot 과 비디오 frame dump 가 정상 출력되면 OK. 만약 `scripts/load_dataset.py` 의 정확한 위치가 다르면:
```bash
find Isaac-GR00T -name 'load_dataset*' -type f
```
로 찾아서 그 경로를 사용.

### 4.4 변환 스크립트 본문 (참고용 — 만약 저장소에 없으면 직접 작성)

```bash
#!/usr/bin/env bash
set -euo pipefail
SRC="datasets/leisaac-so101-pick-orange"
DST="datasets/leisaac-so101-pick-orange-h264"

# 1. data / meta / README 그대로 복사
mkdir -p "$DST"
cp -rn "$SRC/data" "$SRC/meta" "$SRC/README.md" "$DST/"

# 2. videos: ffmpeg av1 → h264
find "$SRC/videos" -type f -name '*.mp4' -print0 |
while IFS= read -r -d '' f; do
  out="${f/$SRC/$DST}"
  mkdir -p "$(dirname "$out")"
  ffmpeg -y -loglevel error -i "$f" \
         -c:v libx264 -crf 23 -preset fast -pix_fmt yuv420p \
         -an "$out"
done

# 3. info.json codec 표기 갱신
sed -i 's/"video.codec": "av1"/"video.codec": "h264"/g' "$DST/meta/info.json"

# 4. modality.json 작성
cat > "$DST/meta/modality.json" <<'JSON'
{
  "state":      {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
  "action":     {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
  "video":      {"front": {"original_key": "observation.images.front"},
                 "wrist": {"original_key": "observation.images.wrist"}},
  "annotation": {"human.task_description": {"original_key": "task_index"}}
}
JSON

echo "OK: $DST 변환 완료"
```

## 5. 학습 (학습 머신, GPU 1 장)

### 5.1 학습 스크립트

본 저장소에 `scripts/imitation_learning/finetune_groot_n16.sh` 가 들어있다 (없다면 5.4 절). 이 스크립트는 NVIDIA 의 SO-100 dual-cam canonical recipe (`Isaac-GR00T/examples/SO100/finetune_so100.sh`) 를 그대로 본 데이터셋에 어댑팅한 것이다.

핵심 인자:
- `--base_model_path nvidia/GR00T-N1.6-3B` : Hugging Face 에서 베이스 모델 자동 다운로드 (~6 GB, 1회만)
- `--dataset_path ../datasets/leisaac-so101-pick-orange-h264` : 4 절에서 만든 H.264 변환본
- `--modality_config_path examples/SO100/so100_config.py` : Isaac-GR00T 가 제공하는 dual-cam SO-100 modality 설정. 우리 데이터의 카메라 키 (`front`, `wrist`) / 모달리티 키 (`single_arm`, `gripper`) 와 그대로 일치하므로 수정 없이 사용
- `--embodiment_tag NEW_EMBODIMENT` : SO-101 은 GR00T 사전 정의 embodiment 가 없어 NEW_EMBODIMENT 사용
- `--num_gpus 1`, `--global_batch_size 32`, `--max_steps 10000`, `--learning_rate 1e-4`, `--warmup_ratio 0.05`, `--weight_decay 1e-5`, `--save_steps 1000`, `--save_total_limit 5`
- `--use_wandb` : wandb 로깅 활성

### 5.2 Smoke test 먼저 (50 step, ~5 분)

본 학습 전에 데이터 로더 + forward pass 가 끝까지 가는지 확인.

```bash
chmod +x scripts/imitation_learning/finetune_groot_n16.sh
# max_steps 와 output_dir 만 임시로 바꿔 50 step 만 실행
sed 's/--max_steps 10000/--max_steps 50/; s/groot_n16_pickorange/groot_n16_smoke/' \
    scripts/imitation_learning/finetune_groot_n16.sh | bash
```

판정:
- 50 step 안에 loss 가 초기 (~1.0) 보다 낮아지면 OK
- OOM, decoder 미스매치, modality 키 누락 등의 에러로 죽으면 트러블슈팅 (7 절)

검증 끝나면 정리:
```bash
rm -rf outputs/train/groot_n16_smoke
```

### 5.3 본 학습 (10 k step, ~1 시간)

```bash
./scripts/imitation_learning/finetune_groot_n16.sh
```

#### 모니터링 (다른 터미널들에서)

**터미널 A** — 학습 콘솔 (위 명령어가 도는 곳).

**터미널 B** — GPU 사용량:
```bash
nvidia-smi -l 5
```
H100 80 GB 에서 batch 32 면 50–70 % VRAM 사용 예상.

**브라우저** — wandb dashboard. 학습 시작 직후 콘솔에 출력되는 `wandb: 🚀 View run at https://wandb.ai/<entity>/<project>/...` URL 접속. 실시간 loss curve / image sample / GPU 사용량.

#### 체크포인트

`outputs/train/groot_n16_pickorange/checkpoint-{step}` 에 1000 step 마다 저장 (각 ~14 GB). `--save_total_limit 5` 로 최근 5개만 보존, 이전 것은 자동 삭제.

#### 성공 기준

- 학습이 OOM/crash 없이 10 k step 까지 완료
- `final loss < 0.05` (참고: rajeshramana 사례 0.017)
- 가장 최근 (`checkpoint-10000`) 이 일반적으로 가장 좋은 정책

### 5.4 학습 스크립트 본문 (참고용 — 만약 저장소에 없으면 직접 작성)

```bash
#!/usr/bin/env bash
set -ex
cd "$(dirname "$0")/../../Isaac-GR00T"

CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path ../datasets/leisaac-so101-pick-orange-h264 \
    --modality_config_path examples/SO100/so100_config.py \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus 1 \
    --output_dir ../outputs/train/groot_n16_pickorange \
    --save_steps 1000 \
    --save_total_limit 5 \
    --max_steps 10000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 32 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 4 \
    --use_wandb
```

## 6. 평가 (분리 운영: 서버 = 학습 머신, 클라이언트 = 평가 머신)

학습 머신이 H100/A100 같은 RT 코어 없는 GPU 면 Isaac Sim 카메라 sensor 가 동작하지 않는다. 그래서:
- **추론 서버**는 학습 머신 (H100) 에서 띄우고 (학습된 체크포인트가 거기 있음)
- **Isaac Sim 평가 클라이언트**는 평가 머신 (Windows + RTX A4000) 에서 실행
- 두 머신은 사내망 TCP 5555 로 통신

```
[학습 머신 — Linux H100]                  [평가 머신 — Windows A4000]
GR00T 추론 서버                              Isaac Sim 클라이언트
inference_service.py     ←─ TCP 5555 ─→     scripts/evaluation/policy_inference.py
(체크포인트 로드)                             (LeIsaac PickOrange 환경)
```

### 6.1 추론 서버 (학습 머신)

```bash
cd /path/to/robotics_manipulation
uv run python Isaac-GR00T/scripts/inference_service.py \
    --server \
    --model_path outputs/train/groot_n16_pickorange \
    --denoising-steps 4 \
    --port 5555
```

옵션 의미:
- `--model_path` : 학습 출력 디렉토리. 자동으로 가장 마지막 체크포인트를 사용
- `--denoising-steps 4` : diffusion sampling step. 4 면 빠르고 품질 OK (canonical 기본값)
- `--port 5555` : 클라이언트가 접속할 포트

방화벽 — 학습 머신의 5555/tcp 가 평가 머신 IP 에서 닿게 사내 네트워크 정책에 맞춰 허용. 보통 `sudo ufw allow from <평가머신IP> to any port 5555/tcp`.

### 6.2 Isaac Sim 평가 클라이언트 (평가 머신)

평가 머신에서 본 저장소를 한 번 더 클론하고 `uv sync`. 그 다음 PowerShell 에서:

```powershell
uv run scripts/evaluation/policy_inference.py `
    --task=LeIsaac-SO101-PickOrange-v0 `
    --policy_type=gr00tn1.6 `
    --policy_host=<학습머신IP> `
    --policy_port=5555 `
    --eval_rounds=10 `
    --enable_cameras
```

`<학습머신IP>` 는 학습 머신의 사내 IP. `--eval_rounds=10` 은 평가 episode 수.

이 클라이언트는 leisaac 의 `Gr00t16ServicePolicyClient` 를 자동 호출해 GR00T 추론 서버와 통신한다. observation 을 매 step 서버에 보내고, 서버에서 받은 action 을 Isaac Sim 환경에 적용.

### 6.3 평가 결과 해석

콘솔에 episode 별 결과가 출력된다 (실제 메트릭 이름은 코드 로그 참고):
- success rate (전체 성공률)
- grasp / place 단계별 성공
- 평균 episode 길이

LeIsaac 데모 화면을 보고 싶으면 `--livestream=2` 도 함께 추가하고 평가 머신에서 Omniverse Streaming Client / 호환 WebRTC 클라이언트로 접속.

## 7. 트러블슈팅

### 7.1 학습 OOM

H100 80 GB 에서 `--global_batch_size 32` 가 OOM 이면 단계적 축소:
```
--global_batch_size 16   # 1차
--global_batch_size 8    # 2차
+ --no-tune-diffusion-model   # 3차 (diffusion decoder 동결, 메모리 ~30% 절약)
```

### 7.2 nvjitlink 충돌

학습 시작 시 `libnvJitLink.so.12: cannot open shared object` 류 에러:
```bash
uv pip install nvidia-nvjitlink-cu12==12.8.61 --no-deps
```

### 7.3 Decord AV1 디코딩 실패

`RuntimeError: AVCodec ... not found` 또는 비디오 frame 이 모두 검은색이면:
- 4 절 변환을 건너뛰었거나 `--dataset_path` 가 원본 디렉토리 (`leisaac-so101-pick-orange`) 를 가리킴
- 해결: `-h264` 디렉토리로 교체

### 7.4 학습 머신의 NVIDIA 드라이버 graphics 라이브러리 누락 (`vkCreateInstance failed`)

Isaac Sim / 도커 컨테이너 기동 시 다음 같은 에러:
```
[Error] [carb.graphics-vulkan.plugin] VkResult: ERROR_INCOMPATIBLE_DRIVER
[Error] [carb.graphics-vulkan.plugin] vkCreateInstance failed.
                                       Vulkan 1.1 is not supported, ...
```

`.run` 인스톨러로 `--no-opengl-files` 와 함께 설치된 호스트에서 발생. 본 저장소 `README.md` 의 `Docker 컨테이너에서 Vulkan 초기화 실패 (Linux)` 섹션을 따라 `.run` 을 `--no-kernel-module --install-libglvnd --silent` 로 재실행해 GL 라이브러리를 추가 설치. 학습 자체에는 영향 없으나 Isaac Sim 사용 시 막힘.

### 7.5 평가 머신이 RT 코어 없는 GPU (`vkCreateRayTracingPipelinesKHR failed`)

데이터센터 GPU (H100/A100) 에서 Isaac Sim 카메라 sensor 가 죽음. RT 코어 있는 GPU 에서만 평가. README `카메라 sensor 가 raytracing pipeline 생성 실패 (RT 코어 없는 GPU)` 섹션 참조.

### 7.6 wandb 비활성화하고 싶음

학습 인자에서 `--use_wandb` 만 제거. transformers Trainer 가 자동으로 tensorboard log 를 `outputs/train/.../runs/` 에 저장하므로 다음으로 볼 수 있음:
```bash
uv run tensorboard --logdir outputs/train/groot_n16_pickorange/runs --port 6006
```

### 7.7 GR00T import 시 추가 모듈 누락

2.4 절의 `peft accelerate diffusers wandb tyro` 외에 다른 모듈을 요구하면 같은 패턴으로 (`uv pip install <name> --no-deps`) 보충. 예: `bitsandbytes`, `omegaconf`, `hydra-core`, `einops`. **`torch` 는 절대 따로 설치하지 말 것** — 기존 2.7.0+cu128 을 깨뜨리면 Isaac Lab 도 함께 깨짐.

### 7.8 추론 서버 / Isaac Sim 클라이언트 직렬화 호환

`Gr00t16ServicePolicyClient` (leisaac) 와 `inference_service.py` (Isaac-GR00T) 사이 직렬화가 어긋나면 평가 머신 콘솔에 `Connection reset` / `pickle / msgpack decode error` 가 뜸. 가능한 대응:
- 서버를 native gr00t 인터페이스로 바꿔서 (`gr00t/eval/run_gr00t_server.py --use_sim_policy_wrapper`) 실행
- 또는 평가 머신 측에서 `scripts/evaluation/policy_inference.py` 의 정책 클라이언트 분기를 확인 (`--policy_type=gr00tn1.6` 이 어떤 클래스를 인스턴스화하는지)

## 8. 참고 자료

- [LeIsaac available_policy 공식 문서](https://lightwheelai.github.io/leisaac/resources/available_policy)
- [rajeshramana/groot-n1.6-pick-orange Hugging Face model card](https://huggingface.co/rajeshramana/groot-n1.6-pick-orange)
- [GROOT_LEARNINGS.md (rajeshramana)](https://github.com/rajeshramana24/leisaac-pick-orange-learnings/blob/main/GROOT_LEARNINGS.md)
- [Leisaac LeRobot Gr00t IsaacSim 입문 (한국어 블로그)](https://velog.io/@choonsik_mom/Leisaac-LeRobot-Gr00t-IsaacSim%EC%9C%BC%EB%A1%9C-%EC%9E%85%EB%AC%B8%ED%95%98%EB%8A%94-VLA-Finetuning)
- `Isaac-GR00T/examples/SO100/` — canonical SO-100 dual-cam recipe (수정 없이 그대로 활용)
- 본 저장소 `README.md` Troubleshooting 섹션 — 환경 셋업 / Vulkan / RT 코어 관련 사전 검증 사항
