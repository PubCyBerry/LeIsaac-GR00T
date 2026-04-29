# serve_leisaac.ps1 — LeIsaac 시뮬레이션 컨테이너 기동 wrapper.
#
# 두 가지 역할을 동일 스크립트로 처리:
#   * (기본) Use case 1: 원격 텔레오퍼레이션 + 데이터셋 기록
#   * (-Eval) Use case 3: gr00t-server 추론 클라이언트로 sim 평가
#
# 사용:
#   # Use case 1 — 기록 모드
#   pwsh scripts/serve_leisaac.ps1 -Record
#
#   # Use case 1 — 다른 task / 다른 leader endpoint
#   pwsh scripts/serve_leisaac.ps1 -Task LeIsaac-SO101-PickOrange-v0 `
#                                  -LeaderEndpoint tcp://192.168.1.10:5556 -Record
#
#   # Use case 3 — sim 평가 (먼저 docker compose up -d gr00t-server 필요)
#   pwsh scripts/serve_leisaac.ps1 -Eval

param(
    [string]$Task           = "LeIsaac-SO101-PickOrange-v0",
    [string]$LeaderEndpoint = "tcp://host.docker.internal:5556",
    [string]$DatasetFile    = "/workspace/datasets/dataset.hdf5",
    [int]   $Livestream     = 2,
    [string]$LanguageInstruction = "Pick up the orange and place it on the plate",
    [int]   $EvalRounds     = 10,
    [int]   $ActionHorizon  = 16,
    [int]   $TimeoutMs      = 15000,
    [string]$PolicyType     = "gr00tn1.6",
    [string]$PolicyHost     = "gr00t-server",
    [int]   $PolicyPort     = 5555,
    [switch]$Record,
    [switch]$Eval,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $repoRoot

# 컨테이너 안의 LeIsaac venv python
$python = "/workspace/leisaac/.venv/bin/python"

if ($Eval) {
    Write-Host "==> Use case 3: sim eval against gr00t-server ($PolicyHost`:$PolicyPort)"
    $cmd = @(
        $python,
        "/workspace/scripts/evaluation/policy_inference.py",
        "--task=$Task",
        "--eval_rounds=$EvalRounds",
        "--policy_type=$PolicyType",
        "--policy_host=$PolicyHost",
        "--policy_port=$PolicyPort",
        "--policy_timeout_ms=$TimeoutMs",
        "--policy_action_horizon=$ActionHorizon",
        "--policy_language_instruction=$LanguageInstruction",
        "--device=cuda",
        "--enable_cameras",
        "--livestream", $Livestream
    )
} else {
    Write-Host "==> Use case 1: remote teleoperation (LEADER_ENDPOINT=$LeaderEndpoint)"
    $cmd = @(
        $python,
        "/workspace/scripts/environments/teleoperation/teleop_se3_agent.py",
        "--task=$Task",
        "--teleop_device=so101leader",
        "--remote_endpoint=$LeaderEndpoint",
        "--device=cuda",
        "--enable_cameras",
        "--livestream", $Livestream
    )
    if ($Record) { $cmd += @("--record", "--dataset_file=$DatasetFile") }
    if ($Resume) { $cmd += "--resume" }
}

Write-Host "==> docker compose run --service-ports --rm leisaac-sim ..."
docker compose run --service-ports --rm leisaac-sim @cmd
