# Sets up the ant tracker on Windows 10/11.
# Run from the repository folder in PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
# Safe to run again (skips what is already there).

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$ModelUrl  = "https://github.com/canafaraci/Taka_Ant_Detection/releases/download/model-v2/ours_v2_best.pt"
$ModelSha  = "4d1b65afa9a8eb30068fc49e7e94b74ccf67f2cf9161433604b2ddf882039d3e"
$ModelPath = "FINAL\checkpoints\ours_v2_best.pt"

function Step($msg) { Write-Host "`n=== $msg" -ForegroundColor Cyan }
function Have($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

Step "Python 3.12"
$py = $null
if (Have "py") { try { & py -3.12 -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $py = "py -3.12" } } catch {} }
if (-not $py) {
    Write-Host "Installing Python 3.12 with winget..."
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    $py = "py -3.12"
}
Write-Host "Using: $py"

Step "ffmpeg (video decoding)"
if (-not (Have "ffmpeg")) {
    winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    Write-Host "ffmpeg installed. If a later step says ffmpeg is missing, close and reopen PowerShell and run this script again."
} else { Write-Host "ffmpeg found." }

Step "Virtual environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) { Invoke-Expression "$py -m venv .venv" }
$vpy = ".\.venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip

Step "PyTorch"
$gpu = $false
if (Have "nvidia-smi") { try { nvidia-smi -L | Out-Null; if ($LASTEXITCODE -eq 0) { $gpu = $true } } catch {} }
if ($gpu) {
    Write-Host "NVIDIA GPU found - installing the CUDA build."
    & $vpy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "No NVIDIA GPU found - installing the CPU build (tracking will be slow)."
    & $vpy -m pip install torch torchvision
}

Step "Other packages"
& $vpy -m pip install -r FINAL\requirements.txt pandas matplotlib

Step "Model weights"
$needDownload = $true
if (Test-Path $ModelPath) {
    $h = (Get-FileHash $ModelPath -Algorithm SHA256).Hash.ToLower()
    if ($h -eq $ModelSha) { $needDownload = $false; Write-Host "Model already present." }
}
if ($needDownload) {
    Write-Host "Downloading model..."
    Invoke-WebRequest -Uri $ModelUrl -OutFile $ModelPath
    $h = (Get-FileHash $ModelPath -Algorithm SHA256).Hash.ToLower()
    if ($h -ne $ModelSha) { throw "Model download is corrupt (checksum mismatch). Delete $ModelPath and run again." }
}

Step "Check"
& $vpy -c "import torch, cv2, tkinter; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('opencv', cv2.__version__, '| tk', tkinter.TkVersion)"
& $vpy FINAL\pipeline.py track-ants -- --help | Select-Object -First 3

Write-Host "`nDone. Double-click 'Ant Tracker.bat' to open the app." -ForegroundColor Green
