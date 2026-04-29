# build.ps1 — LeIsaac × GR00T 통합 이미지 빌드
#
# 사용:
#   pwsh docker/leisaac-gr00t/build.ps1               # 일반 빌드
#   pwsh docker/leisaac-gr00t/build.ps1 -NoCache      # 캐시 무시
#   pwsh docker/leisaac-gr00t/build.ps1 -Tag my-tag   # 태그 변경

param(
    [string]$Tag = "leisaac-gr00t-dev",
    [switch]$NoCache,
    [switch]$Pull
)

$ErrorActionPreference = "Stop"

$repoRoot   = Resolve-Path "$PSScriptRoot\..\.."
$dockerfile = Join-Path $PSScriptRoot "Dockerfile"

Write-Host "==> Building image '$Tag'"
Write-Host "    Dockerfile : $dockerfile"
Write-Host "    Context    : $repoRoot"

$buildArgs = @(
    "build",
    "--platform", "linux/amd64",
    "-t", $Tag,
    "-f", $dockerfile
)
if ($NoCache) { $buildArgs += "--no-cache" }
if ($Pull)    { $buildArgs += "--pull" }
$buildArgs += $repoRoot

& docker @buildArgs
if ($LASTEXITCODE -ne 0) { throw "docker build failed (exit $LASTEXITCODE)" }

Write-Host "==> Image '$Tag' build complete."
docker images $Tag
