# LeIsaac × GR00T 통합 Docker 컨테이너 — 4-역할 사용 가이드

Windows 11 호스트와 Linux Docker 컨테이너를 묶어 다음 4 가지 역할을 단일 이미지(`leisaac-gr00t-dev`) 로 처리한다.

| # | 역할 | 어디서 | 핵심 명령 |
|---|------|--------|----------|
| 1 | **원격 Teleoperation**            | Windows leader → 컨테이너 sim    | `teleop_leader_bridge.ps1` + `serve_leisaac.ps1 -Record` |
| 2 | **GR00T 파인튜닝**                | 컨테이너                         | `finetune_gr00t.ps1` |
| 3 | **GR00T sim 추론**                | 컨테이너 ↔ 컨테이너              | `serve_gr00t.ps1` + `serve_leisaac.ps1 -Eval` |
| 4 | **GR00T → 실기 Follower 원격 제어** | 컨테이너 → Windows follower      | `serve_gr00t.ps1` + `inference_follower_bridge.ps1` |

## 아키텍처

```
Windows 11 Host PC                          ┌─── Linux Docker Container: leisaac-gr00t-dev ───┐
                                            │  /workspace/gr00t/.venv   (Python 3.10)         │
                                            │  /workspace/leisaac/.venv (Python 3.11)         │
┌── Use case 1: Teleop ──┐                  │                                                 │
│ Leader Arm @ COM7      │ ZMQ PUB :5556    │                                                 │
│ teleop_leader_bridge.ps1├─────────────────►│  teleop_se3_agent.py --remote_endpoint=...    │
└────────────────────────┘                  │   → SO101LeaderRemote                          │
                                            │                                                 │
                                            │  IsaacSim 5.1 GUI ──► WebRTC :8211             │
┌── Use case 2: Train ───┐                  │                                                 │
│ ./datasets bind mount  ├─────────────────►│  launch_finetune.py                            │
│ ./outputs bind mount   │◄──────────────── │   → ./outputs/gr00t_finetune/<ts>/checkpoint-*  │
└────────────────────────┘                  │                                                 │
                                            │                                                 │
┌── Use case 3: Sim eval ┐                  │  policy_inference.py                            │
│ (호스트 입력 없음)     │                  │   --policy_host=gr00t-server                    │
└────────────────────────┘                  │   ──► gr00t-server:5555                         │
                                            │                                                 │
┌── Use case 4: Real eval┐                  │                                                 │
│ Follower @ COM8        │ ZMQ REQ :5555    │                                                 │
│ Cameras (front, wrist) ├─────────────────►│  gr00t-server (run_gr00t_server.py)            │
│ inference_follower_    │◄─────────────── │   → action chunk (16-step)                      │
│   bridge.py            │                  │                                                 │
└────────────────────────┘                  └─────────────────────────────────────────────────┘
```

이미지 안에 두 venv 가 병존한다 — GR00T 가 Python 3.10 만 지원하고 IsaacSim/LeIsaac 이 3.11 을 요구하기 때문에 단일 venv 통합이 불가능하다. 둘은 같은 컨테이너의 GPU·HF 캐시·`./datasets`·`./outputs` 마운트를 공유하고 정책 추론은 모두 ZMQ (`tcp://...:5555`) 로 통신한다.

## 전제 조건

- **Docker Desktop** (WSL2 backend) 실행 중, NVIDIA Container Toolkit 활성화
- **NVIDIA 드라이버** 설치 — `nvidia-smi` 로 GPU 가시 확인
- **Isaac-GR00T 서브트리** — `git clone --revision e8e625f4f21898c506a1d8f7d20a289c97a52acf https://github.com/NVIDIA/Isaac-GR00T` (`docs/simluation_demo.md` 참조)
- **Hugging Face 토큰** — `nvidia/GR00T-N1.6-3B` 가 gated 라 필요 (HF 페이지에서 라이선스 동의)
- **Omniverse Streaming Client** — Use case 1·3 livestream 시청용 (또는 Chrome/Edge 브라우저)
- **SO-ARM101 leader / follower 시리얼 포트** — Windows 장치 관리자에서 COM 번호 확인
- **카메라 2대** — Use case 4 시 front / wrist 둘 다 PC USB 포트에 직접 연결 (USB 허브 금지, `README.md` 의 "카메라 대역폭 제한" 섹션 참조)

## 0. 환경 변수 셋업

```powershell
Copy-Item .env.example .env
notepad .env   # HF_TOKEN, GR00T_MODEL_PATH 등 채우기
```

## 1. 이미지 빌드 (최초 1회)

```powershell
pwsh docker/leisaac-gr00t/build.ps1
```

빌드 단계는 다음 순으로 진행되며 첫 빌드는 30~60 분 소요된다.

1. NGC PyTorch 베이스 (이미 캐시돼 있으면 스킵)
2. apt 시스템 패키지 (X11/EGL/Vulkan)
3. uv + Python 3.10 / 3.11 설치
4. GR00T venv 빌드 (`flash-attn`, `deepspeed`, `pytorch3d` source build)
5. LeIsaac venv 빌드 (`uv sync --frozen` — IsaacSim 5.1 다운로드 수 GB)

빌드 완료 후 `docker images leisaac-gr00t-dev` 로 확인. 예상 사이즈 25~35 GB.

## 2. 데이터셋 준비 (Use case 1·2)

기존 LightwheelAI 데모셋:

```powershell
hf download --repo-type dataset LightwheelAI/leisaac-pick-orange `
    --local-dir ./datasets/leisaac-so101-pick-orange
```

LeIsaac HDF5 → LeRobot V2 변환 (Use case 2 학습 입력용):

```powershell
docker compose run --rm gr00t-trainer `
    /workspace/gr00t/.venv/bin/python `
    /workspace/scripts/convert/isaaclab2lerobot.py `
        --hdf5_root /workspace/datasets `
        --repo_id <username>/SoArm_pick_and_place
```

## 3. Use case 1 — 원격 Teleoperation

PowerShell 두 개를 띄운다.

**창 1 — Windows 호스트 (leader 시리얼 → ZMQ)**

```powershell
pwsh scripts/teleop_leader_bridge.ps1 -Port COM7
# `Recalibrate` 가 필요하면:
pwsh scripts/teleop_leader_bridge.ps1 -Port COM7 -Recalibrate
```

`SO101-Leader connected.` 와 calibration 로그가 끝나면 ZMQ PUB 가 `tcp://0.0.0.0:5556` 에서 listen.

**창 2 — Linux 컨테이너 (LeIsaac sim + remote SUB)**

```powershell
pwsh scripts/serve_leisaac.ps1 -Record
# 다른 task / endpoint:
pwsh scripts/serve_leisaac.ps1 `
    -Task LeIsaac-SO101-PickOrange-v0 `
    -LeaderEndpoint tcp://host.docker.internal:5556 `
    -DatasetFile /workspace/datasets/dataset.hdf5 `
    -Record
```

**창 3 — WebRTC 시청**

`http://localhost:8211/streaming/webrtc-client?server=localhost` 에 접속하거나 Omniverse Streaming Client 로 `localhost` 입력. 첫 프레임이 출력되기까지 Kit extension 로딩으로 30 초 정도 검은 화면이 정상.

**컨트롤** (livestream 화면 위에 출력)

| 키 | 동작 |
|----|------|
| `B` | 텔레오퍼레이션 시작 |
| `R` | 리셋 (실패로 기록) |
| `N` | 리셋 (성공으로 기록) |
| `Ctrl+C` | 종료 |

녹화 결과는 호스트 `./datasets/dataset.hdf5` 에 저장된다 (bind mount).

## 4. Use case 2 — GR00T 파인튜닝

데이터셋을 LeRobot V2 형식으로 변환했다는 전제. modality config 는 `Isaac-GR00T/examples/SO100/so100_config.py` 를 참고해 SO-101 용으로 한 부 만들어 `./scripts/configs/so101_modality.py` 에 둔다.

```powershell
pwsh scripts/finetune_gr00t.ps1 `
    -DatasetPath /workspace/datasets/leisaac-so101-pick-orange `
    -ModalityConfig /workspace/scripts/configs/so101_modality.py `
    -EmbodimentTag NEW_EMBODIMENT `
    -MaxSteps 10000 `
    -BatchSize 32 `
    -LearningRate 1e-4
```

VRAM 16 GB (A4000) 에서 OOM 이 나면:

```powershell
pwsh scripts/finetune_gr00t.ps1 ... `
    -BatchSize 16 -GradAccumSteps 2
```

체크포인트 산출물은 호스트 `./outputs/gr00t_finetune/<timestamp>/checkpoint-XXXX/` 에 떨어진다 (bind mount).

## 5. Use case 3 — sim 추론 (학습된 / 사전학습 정책)

**창 1 — 정책 서버 기동**

사전학습 모델:

```powershell
pwsh scripts/serve_gr00t.ps1 -Detach
```

학습된 체크포인트로 바꾸려면 `.env` 또는 환경변수에 컨테이너 경로 지정:

```powershell
$env:GR00T_MODEL_PATH = "/workspace/outputs/gr00t_finetune/20260428_120000/checkpoint-10000"
pwsh scripts/serve_gr00t.ps1 -Recreate -Detach
```

서버 로그에 `Server is ready and listening on tcp://0.0.0.0:5555` 가 떠야 한다 — `docker compose logs -f gr00t-server`.

**창 2 — sim 평가 클라이언트**

```powershell
pwsh scripts/serve_leisaac.ps1 -Eval
# 다른 task / language instruction:
pwsh scripts/serve_leisaac.ps1 -Eval `
    -Task LeIsaac-SO101-PickOrange-v0 `
    -LanguageInstruction "Pick up the orange and place it on the plate" `
    -EvalRounds 10
```

같은 compose 네트워크 안에서 `gr00t-server` 호스트명으로 resolve 되므로 wrapper 가 `--policy_host=gr00t-server` 를 자동 전달.

WebRTC 시청은 Use case 1 과 동일: `http://localhost:8211/...`.

## 6. Use case 4 — GR00T 추론 → Windows 실기 Follower 원격 제어

**창 1 — 정책 서버 기동** (Use case 3 과 동일)

```powershell
pwsh scripts/serve_gr00t.ps1 -Detach
```

**창 2 — Windows 호스트에서 follower 브릿지 실행**

```powershell
pwsh scripts/inference_follower_bridge.ps1 -FollowerPort COM8 -FrontCam 0 -WristCam 1
# 다른 task 지시 / 다른 호스트:
pwsh scripts/inference_follower_bridge.ps1 `
    -FollowerPort COM8 `
    -FrontCam 0 -WristCam 1 `
    -Task "Sort the blocks by color" `
    -PolicyHost localhost
```

스크립트 동작:

1. lerobot `SO101Follower` 로 follower 시리얼 + 카메라 2대 동시 연결
2. 30 Hz 루프:
   - `follower.get_observation()` → 6개 joint pos + (front, wrist) RGB 프레임
   - GR00T 서버 `tcp://localhost:5555` 에 ZMQ REQ
   - 16-step action chunk 수신 → `follower.send_action({...})` 으로 적용

`Ctrl+C` 또는 `--MaxSeconds N` 인자로 종료.

## 검증

| # | 단계 | 명령 | 예상 결과 |
|---|------|------|-----------|
| 1 | 이미지 빌드 | `docker images leisaac-gr00t-dev` | size 25-35 GB |
| 2 | GPU 인식 | `docker compose run --rm gr00t-server nvidia-smi` | A4000 visible |
| 3 | LeIsaac venv | `docker compose run --rm leisaac-sim /workspace/leisaac/.venv/bin/python -c "import isaacsim, leisaac, h5py; print(isaacsim.__version__, h5py.version.hdf5_version)"` | `5.1.0 1.14.6` |
| 4 | GR00T venv | `docker compose run --rm gr00t-server /workspace/gr00t/.venv/bin/python -c "import gr00t, flash_attn, deepspeed; print(gr00t.__version__)"` | 정상 출력 |
| 5 | Use case 1 | leader bridge 후 sim 컨테이너에서 livestream 화면에 가상 팔로워가 leader 따라 움직임. `./datasets/dataset.hdf5` 생성 |
| 6 | Use case 2 | `./outputs/gr00t_finetune/.../checkpoint-1000/` 생성, loss 감소 로그 |
| 7 | Use case 3 | sim 컨테이너에서 `Service is running.` 출력, action_horizon=16 응답, 가상 팔로워가 정책대로 움직임 |
| 8 | Use case 4 | `Test-NetConnection localhost -Port 5555` → `True`, follower 가 카메라/joint state 기반으로 자율 동작 |

## 트러블슈팅

- **`Service is not running`** — gr00t-server 가 안 떠 있거나 `--host 127.0.0.1` 로 바인드됨. `docker compose logs gr00t-server` 확인.
- **HF 401 (gated 모델)** — `.env` 의 `HF_TOKEN` 누락. HF 페이지에서 GR00T-N1.6 라이선스 동의 후 토큰 발급.
- **WebRTC 검은 화면** — Kit extension 로딩 ~30 초 정상. 그 이후에도 검은 화면이면 GPU 인식 실패 (`nvidia-smi` 컨테이너에서 확인).
- **CUDA OOM (학습)** — `-BatchSize 16 -GradAccumSteps 2` 로 줄이거나 `--policy.device=cuda` 가 단일 GPU 만 잡도록.
- **카메라 1 대만 잡힘 (Use case 4)** — USB 허브 사용 중. PC USB 포트에 직접 꽂을 것 (`README.md` 카메라 섹션 참조).
- **빌드 실패** — `pwsh docker/leisaac-gr00t/build.ps1 -NoCache -Pull`.
- **ZMQ timeout** — `-TimeoutMs 30000` 으로 늘리고 첫 호출 시 모델 워밍업 기다리기.
- **submodule dirty** — `Isaac-GR00T/docker/src/` 가 빌드 후 남아있다면 `Isaac-GR00T/docker/build.sh` 가 정리 안 한 것. 본 통합 Dockerfile 은 그 경로를 건드리지 않으므로 무시 가능.

## 부록 — 데이터셋 변환 가이드

LeIsaac sim 으로 녹화한 HDF5 파일을 GR00T 학습에 쓰려면 다음 두 단계 변환이 필요하다.

1. **HDF5 → LeRobot V2 (parquet + mp4)**
   - `scripts/convert/isaaclab2lerobot.py` 또는 `Isaac-GR00T/scripts/lerobot_conversion/convert_v3_to_v2.py` 사용
   - 출력 구조:
     ```
     <repo_id>/
     ├── meta/{episodes.jsonl, info.json, modality.json, tasks.jsonl}
     ├── data/chunk-000/episode_NNNNNN.parquet
     └── videos/chunk-000/observation.images.<cam>/episode_NNNNNN.mp4
     ```

2. **`meta/modality.json` 작성** — `Isaac-GR00T/examples/SO100/so100_config.py` 참조. SO-101 의 경우 `state.single_arm` (5 차원) + `state.gripper` (1 차원) + `video.front` + `video.wrist` 로 구성.

자세한 스키마는 `Isaac-GR00T/getting_started/data_preparation.md` 참고.
