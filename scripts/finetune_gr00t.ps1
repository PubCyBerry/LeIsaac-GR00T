# finetune_gr00t.ps1 — Use case 2: GR00T N1.6 파인튜닝.
#
# 컨테이너의 gr00t-trainer 서비스에서 launch_finetune.py 를 실행한다.
# 데이터셋 / 모달리티 config / 출력 경로는 모두 컨테이너 내부 경로를 전달해야 한다 — bind mount 덕에
# 호스트 ./datasets, ./outputs 가 컨테이너 /workspace/datasets, /workspace/outputs 로 보인다.
#
# 사용:
#   pwsh scripts/finetune_gr00t.ps1 `
#       -DatasetPath /workspace/datasets/leisaac-so101-pick-orange `
#       -ModalityConfig /workspace/scripts/configs/so101_modality.py `
#       -MaxSteps 10000 -BatchSize 32

param(
    [string]$BaseModel        = "nvidia/GR00T-N1.6-3B",
    [Parameter(Mandatory=$true)]
    [string]$DatasetPath,
    [string]$ModalityConfig   = "",
    [string]$OutputDir        = "",
    [string]$EmbodimentTag    = "NEW_EMBODIMENT",
    [int]   $MaxSteps         = 10000,
    [int]   $BatchSize        = 32,
    [int]   $GradAccumSteps   = 1,
    [double]$LearningRate     = 1e-4,
    [int]   $SaveSteps        = 1000,
    [int]   $NumWorkers       = 4,
    [int]   $NumGpus          = 1,
    [string]$HfToken          = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $repoRoot

if (-not $OutputDir) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = "/workspace/outputs/gr00t_finetune/$stamp"
}
if ($HfToken) { $env:HF_TOKEN = $HfToken }

Write-Host "==> Use case 2: GR00T finetune"
Write-Host "    Base model       : $BaseModel"
Write-Host "    Dataset (in-ctr) : $DatasetPath"
Write-Host "    Modality config  : $ModalityConfig"
Write-Host "    Output dir       : $OutputDir"
Write-Host "    Steps / batch    : $MaxSteps / $BatchSize (grad-accum=$GradAccumSteps)"

$cmd = @(
    "/workspace/gr00t/.venv/bin/python",
    "/workspace/gr00t/gr00t/experiment/launch_finetune.py",
    "--base_model_path", $BaseModel,
    "--dataset_path",    $DatasetPath,
    "--embodiment_tag",  $EmbodimentTag,
    "--output_dir",      $OutputDir,
    "--max_steps",       $MaxSteps,
    "--global_batch_size", $BatchSize,
    "--gradient_accumulation_steps", $GradAccumSteps,
    "--learning_rate",   $LearningRate,
    "--save_steps",      $SaveSteps,
    "--dataloader_num_workers", $NumWorkers,
    "--num_gpus",        $NumGpus
)
if ($ModalityConfig) { $cmd += @("--modality_config_path", $ModalityConfig) }

Write-Host "==> docker compose run --rm gr00t-trainer ..."
docker compose run --rm gr00t-trainer @cmd
