# inference_follower_bridge.ps1 — Use case 4: 컨테이너 GR00T 서버로 Windows follower 암 원격 제어 wrapper.
#
# 전제: `docker compose up -d gr00t-server` 가 먼저 실행돼 5555 포트가 listen 중.
#
# 사용:
#   pwsh scripts/inference_follower_bridge.ps1
#   pwsh scripts/inference_follower_bridge.ps1 -FollowerPort COM8 -FrontCam 0 -WristCam 1
#   pwsh scripts/inference_follower_bridge.ps1 -PolicyHost 192.168.1.20 -Task "Sort the blocks by color"

param(
    [string]$PolicyHost   = "localhost",
    [int]   $PolicyPort   = 5555,
    [int]   $TimeoutMs    = 15000,
    [string]$FollowerPort = "COM8",
    [string]$FollowerId   = "follower_arm",
    [int]   $FrontCam     = 0,
    [int]   $WristCam     = 1,
    [int]   $CamWidth     = 640,
    [int]   $CamHeight    = 480,
    [int]   $CamFps       = 30,
    [int]   $StepHz       = 30,
    [double]$MaxSeconds   = 0.0,
    [string]$Task         = "Pick up the orange and place it on the plate"
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $repoRoot

Write-Host "==> Use case 4: GR00T 추론 → 실기 follower 제어"
Write-Host "    Policy server : tcp://$PolicyHost`:$PolicyPort"
Write-Host "    Follower port : $FollowerPort"
Write-Host "    Cameras       : front=$FrontCam, wrist=$WristCam ($CamWidth x $CamHeight @ $CamFps fps)"

$pyArgs = @(
    "scripts/inference_follower_bridge.py",
    "--policy_host=$PolicyHost",
    "--policy_port=$PolicyPort",
    "--policy_timeout_ms=$TimeoutMs",
    "--follower_port=$FollowerPort",
    "--follower_id=$FollowerId",
    "--front_cam_index=$FrontCam",
    "--wrist_cam_index=$WristCam",
    "--cam_width=$CamWidth",
    "--cam_height=$CamHeight",
    "--cam_fps=$CamFps",
    "--step_hz=$StepHz",
    "--max_seconds=$MaxSeconds",
    "--task", $Task
)

& uv run @pyArgs
