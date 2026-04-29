# 시뮬레이션에서 GR00T N1.6 추론 테스트

> **NOTE — 통합 컨테이너 가이드**
>
> 이 문서는 **Windows 호스트에서 Isaac Sim 을 native 로 돌리고, GR00T 추론만 Linux 컨테이너로 격리**하는 초기 구성을 설명한다.
> 이후 IsaacSim·LeIsaac·GR00T 를 **하나의 통합 이미지**(`leisaac-gr00t-dev`) 로 묶고 Teleop / Finetune / Sim eval / 실기 follower eval 4 가지 역할을 모두 커버하도록 확장됐다. 새로운 사용 흐름은 [`docs/leisaac-gr00t-container.md`](leisaac-gr00t-container.md) 를 참고할 것.
>
> 본 문서의 명령은 호환성을 위해 그대로 두되, `gr00t-dev` 이미지 / `gr00t-server` 컨테이너명은 통합 이미지(`leisaac-gr00t-dev`) / compose 서비스명(`gr00t-server`) 로 사실상 대체된다.

Windows 호스트에서 Isaac Sim(LeIsaac) 기반 평가를 돌리면서, GR00T N1.6 정책 추론 서버는 Linux Docker 컨테이너로 격리 기동한다. Windows 에서는 `flash-attn`, `deepspeed`, `torchcodec` 휠 제공/빌드가 불가능하므로 추론 쪽만 컨테이너화한다. 두 프로세스는 ZMQ REQ-REP (TCP 5555) 로 통신한다.

## 전제 조건

- Docker Desktop (WSL2 backend) 실행 중
- NVIDIA 드라이버 설치 완료 — `nvidia-smi` 로 GPU 가시 확인
- Git Bash (빌드 시 `build.sh` 실행용)
- Isaac-GR00T 서브모듈이 문서 지정 커밋에 고정되어 있음: `e8e625f4f21898c506a1d8f7d20a289c97a52acf`

## 1단계 — `gr00t-dev` 이미지 빌드 (최초 1 회)

Git Bash 에서 실행:

```bash
cd /d/Workspaces/robotics_manipulation/Isaac-GR00T/docker
bash build.sh
```

`build.sh` 는 레포 루트를 `docker/src/gr00t` 로 복사한 뒤 `nvcr.io/nvidia/pytorch:25.04-py3` 베이스 위에 `flash-attn`, `deepspeed`, `torchcodec`, `pyzmq`, `pytorch3d` 등을 설치한다. 결과 이미지명: `gr00t-dev` (수 GB).

## 2단계 — Policy Server 컨테이너 기동

세 가지 중 한 가지를 선택한다. 셋 다 동일하게 `--host 0.0.0.0` 으로 바인드해 호스트에서 접속 가능하게 하고, HuggingFace 모델 캐시를 `%USERPROFILE%\.cache\huggingface` 로 마운트해 재다운로드를 방지한다.

### 2-A. PowerShell 래퍼 (권장)

```powershell
pwsh scripts/serve_gr00t.ps1
# 백그라운드로 기동하려면
pwsh scripts/serve_gr00t.ps1 -Detach
# gated 모델일 경우
pwsh scripts/serve_gr00t.ps1 -HfToken $env:HF_TOKEN
```

### 2-B. docker compose

```powershell
docker compose up gr00t-server
# 백그라운드
docker compose up -d gr00t-server
docker compose logs -f gr00t-server
```

### 2-C. 직접 `docker run`

```powershell
docker run --rm -it `
  --gpus all `
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 `
  -p 5555:5555 `
  -v ${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface `
  -e HF_HOME=/root/.cache/huggingface `
  --name gr00t-server `
  gr00t-dev `
  /workspace/gr00t/.venv/bin/python /workspace/gr00t/gr00t/eval/run_gr00t_server.py `
    --embodiment-tag GR1 `
    --model-path nvidia/GR00T-N1.6-3B `
    --host 0.0.0.0 `
    --port 5555
```

> 왜 venv python 절대경로를 직접 지정하는가: 이미지의 엔트리포인트(`/opt/nvidia/nvidia_entrypoint.sh`) 는 NGC 배너 출력 후 argv 를 exec 만 하므로 venv 를 자동 활성화하지 않는다. 단순히 `python` 을 부르면 시스템 Python 3.12 가 잡히지만 gr00t 는 `/workspace/gr00t/.venv` (Python 3.10) 에만 설치돼 있어 `ModuleNotFoundError` 가 난다. `--ipc=host` 와 memlock/stack ulimit 는 NGC PyTorch 이미지가 권장하는 runtime 플래그다.

기동 성공 시 컨테이너 로그:

```
Server is ready and listening on tcp://0.0.0.0:5555
```

## 3단계 — Isaac Sim 클라이언트 실행 (호스트)

서버 listening 확인 후 PowerShell 에서:

```powershell
uv run scripts/evaluation/policy_inference.py `
    --task=LeIsaac-SO101-PickOrange-v0 `
    --eval_rounds=10 `
    --policy_type=gr00tn1.6 `
    --policy_host=localhost `
    --policy_port=5555 `
    --policy_timeout_ms=15000 `
    --policy_action_horizon=16 `
    --policy_language_instruction="Pick up the orange and place it on the plate" `
    --device=cuda `
    --enable_cameras
```

`--policy_timeout_ms` 는 ping 뿐 아니라 `get_action` 에도 적용된다. 첫 호출은 CUDA 커널 컴파일로 느려지므로 15000 ms 이상을 권장한다.

## 검증

| # | 확인 항목 | 방법 |
|---|-----------|------|
| 1 | 이미지 존재 | `docker images gr00t-dev` |
| 2 | 서버 기동 | `docker logs -f gr00t-server` → `Server is ready and listening on tcp://0.0.0.0:5555` |
| 3 | 포트 개방 | `Test-NetConnection localhost -Port 5555` → `TcpTestSucceeded : True` |
| 4 | Ping 응답 | 클라이언트 로그에 `Service is running.` |
| 5 | 추론 루프 진입 | 서버 로그에 `get_action` 수신, action 길이 = `policy_action_horizon` |
| 6 | VRAM | `nvidia-smi -l 2` → A4000 15GB 한계 내 (Sim ~1.6GB + GR00T 3B bf16 ~6-10GB) |

## 트러블슈팅

- **`RuntimeError: Service is not running`**: 컨테이너가 안 떴거나 `--host` 가 `127.0.0.1` 로 남음. `docker ps` 로 컨테이너 상태와 `-p 5555:5555` 포트 매핑 확인.
- **HF 401 에러**: 모델이 gated. HuggingFace 계정에서 라이선스 동의 후 `HF_TOKEN` 을 전달 (2-A 래퍼는 `-HfToken`, compose 는 `.env` 의 `HF_TOKEN`).
- **CUDA OOM**: Isaac Sim 렌더링 옵션을 낮추거나, 서버를 별도 장비로 분리 후 `--policy_host` 를 해당 IP 로 지정.
- **빌드 실패**: `docker system prune -a` 후 `bash build.sh --no-cache`.

## 배경

- Isaac-GR00T 커밋 핀: `e8e625f4f21898c506a1d8f7d20a289c97a52acf`
- 통신: ZMQ REQ-REP over TCP, 포트 5555
- 서버 엔드포인트: `ping`, `get_action`, `reset`, `get_modality_config`, `kill`
- 관련 코드:
  - 서버 엔트리: `Isaac-GR00T/gr00t/eval/run_gr00t_server.py`
  - 서버 구현: `Isaac-GR00T/gr00t/policy/server_client.py`
  - 정책 로더: `Isaac-GR00T/gr00t/policy/gr00t_policy.py`
  - 클라이언트: `.venv/Lib/site-packages/leisaac/policy/service_policy_clients.py`
  - 클라이언트 베이스: `.venv/Lib/site-packages/leisaac/policy/base.py`
