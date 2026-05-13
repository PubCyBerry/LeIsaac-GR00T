# SO-ARM101 VLA Control System

## 환경 구성

- CUDA 12.8
- Python 3.11
- PyTorch 2.7
- LeRobot 0.4.4
- Isaac Sim 5.1.0
- Isaac Lab 2.3.0

## 환경 설정(WSL2 Ubuntu 24.04 / miniforge)

miniforge 및 기타 의존성 설치

```bash
# miniforge 설치
sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt update
sudo apt-get install --no-install-recommends  libevdev-dev ffmpeg speech-dispatcher espeak-ng vulkan-tools mesa-vulkan-drivers libxkbcommon-x11-0 -y
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

source $HOME/.bashrc
conda create -y -n lerobot python=3.11
conda activate lerobot
conda install evdev -c conda-forge -y
conda install ffmpeg -c conda-forge -y
conda install pip -y

git clone -b v0.5.1 https://github.com/huggingface/lerobot
cd lerobot

pip install -e ".[feetech]"
```

## 환경 설정(WSL2 Ubuntu 24.04 / uv)

uv 및 기타 의존성 설치

```bash
# uv 설치
sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt update
sudo apt-get install --no-install-recommends  libevdev-dev ffmpeg speech-dispatcher espeak-ng vulkan-tools mesa-vulkan-drivers libxkbcommon-x11-0 -y
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.bashrc

uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate

uv sync --group teleop
```

## 환경 설정(Docker)

`.env.example` 파일을 `.env` 파일로 복사한 후 포트 정보, API 키 등을 입력한 후 컨테이너를 실행

```bash
# setup-motors / calibration /teleoperation / record / dataset-viz
docker compose -f docker/docker-compose.yaml up teleop
```


## (WSL)SO-101 Arm, 카메라 포트 연결

1. `usbipd` 설치

```powershell
winget install usdipd
```

2. 관리자 권한으로 powershell 실행 후 Leader, Follower Arm, 카메라 포트 연결

```powershell
# 포트 목록 조회
usbipd list
# 최초 1회만 실행
usbipd bind --busid <leader-port>
usbipd bind --busid <follower-port>
usbipd bind --busid <wrist-cam-port>
usbipd bind --busid <belly-cam-port>
# usb 재연결할 때마다 / WSL 리부트할 때마다 실행
usbipd attach --wsl --busid <leader-port>
usbipd attach --wsl --busid <follower-port>
usbipd attach --wsl --busid <wrist-cam-port>
usbipd attach --wsl --busid <belly-cam-port>
# Windows로 포트를 되돌릴 경우:
usbipd detach --busid <port>
```

3. WSL로 돌아와서 각 포트에 읽기/쓰기 권한 부여

```bash
# Leader Arm, Follower Arm USB
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
# Wrist Cam, Belly Cam
sudo chmod 666 /dev/video0 /dev/video2
sudo usermod -aG dialout $USER
```

## Rerun Viewer 사용법

Rerun viewer를 사용해 데이터를 시각화할 수 있음.

1. Host PC에서 `pip install rerun-sdk==0.26.2`로 rerun viewer 설치(컨테이너 내부 rerun 버전과 일치해야 함)

2. `rerun`으로 rerun viewer 실행.

3. `lerobot-teleoperate` 또는 `lerobot-record` 실행하면 자동으로 화면 연동됨.

## 실기기 Teleoperation

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=${FOLLOWER_PORT} \
    --robot.id=${FOLLOWER_ID:-konan_robot} \
    --robot.cameras="{
        wrist: {type: opencv, index_or_path: ${WRIST_CAM_DEV}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}, fourcc: MJPG},
        belly: {type: opencv, index_or_path: ${BELLY_CAM_DEV}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}, fourcc: MJPG},
        }" \
    --teleop.type=so101_leader \
    --teleop.port=${LEADER_PORT} \
    --teleop.id=${LEADER_ID:-konan_teleop} \
    --display_data=true
```

## 데이터셋 기록

1. Hugging Face CLI로 token 추가(토큰은 [Hugging Face settings](https://huggingface.co/settings/tokens)에서 생성 가능)

```bash
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')
echo $HF_USER
```

2. 에피소드 생성 - 최소 50개, 위치당 10개 에피소드 생성 권장

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=${FOLLOWER_PORT} \
    --robot.id=${FOLLOWER_ID:-konan_robot} \
    --robot.cameras="{
        wrist: {type: opencv, index_or_path: ${WRIST_CAM_DEV}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}, fourcc: MJPG},
        belly: {type: opencv, index_or_path: ${BELLY_CAM_DEV}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}, fourcc: MJPG},
        }" \
    --teleop.type=so101_leader \
    --teleop.port=${LEADER_PORT} \
    --teleop.id=${LEADER_ID:-konan_teleop} \
    --dataset.single_task=${SINGLE_TASK} \
    --dataset.repo_id=${HF_USER}/${SINGLE_TASK} \
    --dataset.num_episodes=${NUM_EPISODES} \
    --dataset.episode_time_s=${EPISODE_TIME_S} \
    --dataset.reset_time_s=${RESET_TIME_S} \
    --dataset.push_to_hub=${PUSH_TO_HUB}
    --dataset.fps=${CAM_FPS} \
    --display_data=false \
```
