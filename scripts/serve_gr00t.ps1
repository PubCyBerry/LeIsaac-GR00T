#Requires -Version 5.1
<#
.SYNOPSIS
  GR00T N1.6 Policy Server 를 통합 LeIsaac × GR00T 컨테이너에서 기동한다.

.DESCRIPTION
  내부적으로 `docker compose up gr00t-server` 를 호출한다 (필요한 마운트·GPU 설정·extra_hosts 가 docker-compose.yml 에 이미 정의돼 있다).
  로컬 학습 산출물 체크포인트를 사용하려면 -ModelPath 에 컨테이너 경로(/workspace/outputs/...) 를 전달.

.PARAMETER ModelPath
  HuggingFace 모델 ID 또는 컨테이너 내부 체크포인트 경로. 기본: nvidia/GR00T-N1.6-3B.

.PARAMETER EmbodimentTag
  GR00T embodiment tag. 학습 시 사용한 tag 와 일치시킬 것. 기본: NEW_EMBODIMENT.

.PARAMETER HfToken
  HuggingFace 액세스 토큰 (gated 모델용). 미지정이면 .env 의 HF_TOKEN 을 그대로 사용.

.PARAMETER Detach
  지정 시 백그라운드로 실행 (`up -d`). 미지정 시 foreground.

.PARAMETER Recreate
  실행 전에 기존 컨테이너를 강제로 다시 만든다.

.EXAMPLE
  pwsh scripts/serve_gr00t.ps1
  pwsh scripts/serve_gr00t.ps1 -Detach
  pwsh scripts/serve_gr00t.ps1 -ModelPath /workspace/outputs/gr00t_finetune/20260428_120000/checkpoint-10000
  pwsh scripts/serve_gr00t.ps1 -HfToken $env:HF_TOKEN
#>
param(
    [string]$ModelPath     = "",
    [string]$EmbodimentTag = "",
    [string]$HfToken       = "",
    [switch]$Detach,
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $repoRoot

# docker-compose 가 ${VAR} 로 참조하는 환경변수만 세팅. 미지정 항목은 .env 또는 yml 의 default 사용.
if ($ModelPath)     { $env:GR00T_MODEL_PATH     = $ModelPath }
if ($EmbodimentTag) { $env:GR00T_EMBODIMENT_TAG = $EmbodimentTag }
if ($HfToken)       { $env:HF_TOKEN             = $HfToken }

$composeArgs = @("compose")

if ($Recreate) {
    Write-Host "[serve_gr00t] 기존 gr00t-server 컨테이너 제거"
    docker compose rm -fs gr00t-server | Out-Null
}

$composeArgs += @("up")
if ($Detach) { $composeArgs += "-d" }
$composeArgs += "gr00t-server"

Write-Host "[serve_gr00t] docker $($composeArgs -join ' ')"
docker @composeArgs
