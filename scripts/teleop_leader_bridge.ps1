# teleop_leader_bridge.ps1 — Use case 1: Windows 호스트에서 SO-ARM101 leader 암 시리얼 포트를 ZMQ PUB 으로 발행.
#
# Linux 컨테이너의 leisaac-sim 이 `SO101LeaderRemote(endpoint="tcp://host.docker.internal:5556")` 로 SUB 한다.
#
# 사용:
#   pwsh scripts/teleop_leader_bridge.ps1                    # 기본 (COM7, 5556 포트)
#   pwsh scripts/teleop_leader_bridge.ps1 -Port COM5         # 다른 COM 포트
#   pwsh scripts/teleop_leader_bridge.ps1 -Recalibrate       # leader 재보정

param(
    [string]$Port        = "COM7",
    [string]$Bind        = "tcp://0.0.0.0:5556",
    [int]   $Rate        = 50,
    [string]$Id          = "leader_arm",
    [switch]$Recalibrate
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $repoRoot

Write-Host "==> Starting SO-101 leader → ZMQ bridge"
Write-Host "    Serial port : $Port"
Write-Host "    ZMQ bind    : $Bind"
Write-Host "    Rate        : $Rate Hz"

$pyArgs = @(
    "scripts/environments/teleoperation/so101_joint_state_server.py",
    "--port",  $Port,
    "--bind",  $Bind,
    "--rate",  $Rate,
    "--id",    $Id
)
if ($Recalibrate) { $pyArgs += "--recalibrate" }

& uv run @pyArgs
