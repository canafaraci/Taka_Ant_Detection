# Sets up the ant tracker on Windows 10/11. No administrator rights needed.
# Run from the repository folder in PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
# Safe to run again (skips what is already there).

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # much faster downloads in Windows PowerShell
Set-Location $PSScriptRoot
$ModelUrl  = "https://github.com/canafaraci/Taka_Ant_Detection/releases/download/model-v2/ours_v2_best.pt"
$ModelSha  = "4d1b65afa9a8eb30068fc49e7e94b74ccf67f2cf9161433604b2ddf882039d3e"
$ModelPath = "FINAL\checkpoints\ours_v2_best.pt"
$Torch     = "https://download.pytorch.org/whl"

function Step($msg) { Write-Host "`n=== $msg" -ForegroundColor Cyan }
function Have($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User") + ";" +
                "$env:LOCALAPPDATA\Microsoft\WinGet\Links"
}
function Run($exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $exe $($argv -join ' ')" }
}
function Find-Python {
    # The python.exe in WindowsApps is only a Microsoft Store shortcut, so look for a real install.
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:ProgramFiles\Python312\python.exe"
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    if (Have "py") {
        try {
            $p = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $p -and (Test-Path $p.Trim())) { return $p.Trim() }
        } catch {}
    }
    return $null
}
function Test-Cuda($vpy) {
    $ErrorActionPreference = "Continue"   # a failed test must return False, not stop the script
    # Runs real PyTorch kernels (not just a library call), so an unsupported GPU fails here.
    $code = "import torch, torch.nn.functional as F; assert torch.cuda.is_available(); x=torch.randn(1,3,64,64,device='cuda'); w=torch.randn(8,3,3,3,device='cuda'); y=F.relu(F.conv2d(x,w)).sum()+ (x*2+1).mean(); torch.cuda.synchronize(); print('CUDA OK on', torch.cuda.get_device_name(0))"
    & $vpy -c $code | Out-Host
    return ($LASTEXITCODE -eq 0)
}

Step "Python 3.12"
$py = Find-Python
if (-not $py) {
    Write-Host "Installing Python 3.12 for this user with winget..."
    winget install -e --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
    Refresh-Path
    $py = Find-Python
    if (-not $py) { throw "Python 3.12 was not found after installing. Close PowerShell, open a new one and run this script again." }
}
Write-Host "Using: $py"

Step "ffmpeg (video decoding)"
Refresh-Path
if (-not (Have "ffmpeg")) {
    winget install -e --id Gyan.FFmpeg --scope user --accept-source-agreements --accept-package-agreements
    Refresh-Path
    if (-not (Have "ffmpeg")) {
        winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
        Refresh-Path
    }
}
if (Have "ffmpeg") { Write-Host "ffmpeg found: $((Get-Command ffmpeg).Source)" }
else { Write-Host "ffmpeg not found yet. Close and reopen PowerShell, then run this script again." -ForegroundColor Yellow }

Step "Virtual environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) { Run $py @("-m", "venv", ".venv") }
$vpy = (Resolve-Path ".venv\Scripts\python.exe").Path
Run $vpy @("-m", "pip", "install", "--upgrade", "pip")

Step "PyTorch"
$cap = $null
if (Have "nvidia-smi") {
    try {
        $capText = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1).Trim()
        $cap = [double]::Parse($capText, [Globalization.CultureInfo]::InvariantCulture)
    } catch {}
}
$cudaOk = $false
if ($cap) {
    Write-Host "NVIDIA GPU found (compute capability $cap)."
    $already = $false
    if (Test-Path ".venv\Lib\site-packages\torch") { $already = Test-Cuda $vpy }
    if ($already) { $cudaOk = $true }
    else {
        # Newer cards (RTX 20xx and later) use CUDA 12.8; older cards such as the GTX 10xx series need CUDA 12.6.
        $index = if ($cap -ge 7.5) { "cu128" } else { "cu126" }
        Write-Host "Installing the CUDA build ($index, about 2.5 GB)..."
        Run $vpy @("-m", "pip", "install", "--upgrade", "torch", "torchvision", "--index-url", "$Torch/$index")
        $cudaOk = Test-Cuda $vpy
        if (-not $cudaOk) {
            Write-Host "This PyTorch build does not run on this GPU. Trying PyTorch 2.7.1 (CUDA 11.8), which supports older cards..." -ForegroundColor Yellow
            Run $vpy @("-m", "pip", "install", "--force-reinstall", "torch==2.7.1", "torchvision==0.22.1", "--index-url", "$Torch/cu118")
            $cudaOk = Test-Cuda $vpy
        }
    }
}
if (-not $cudaOk) {
    if ($cap) { Write-Host "The GPU could not be used; installing the CPU build instead." -ForegroundColor Yellow }
    else { Write-Host "No NVIDIA GPU found - using the CPU build." }
    if ($cap -or -not (Test-Path ".venv\Lib\site-packages\torch")) {
        Run $vpy @("-m", "pip", "install", "--force-reinstall", "torch", "torchvision", "--index-url", "$Torch/cpu")
    } else { Write-Host "PyTorch already installed." }
}

Step "Other packages"
Run $vpy @("-m", "pip", "install", "-r", "FINAL\requirements.txt", "pandas", "matplotlib")

Step "Model weights"
$needDownload = $true
if (Test-Path $ModelPath) {
    $h = (Get-FileHash $ModelPath -Algorithm SHA256).Hash.ToLower()
    if ($h -eq $ModelSha) { $needDownload = $false; Write-Host "Model already present." }
}
if ($needDownload) {
    Write-Host "Downloading model (76 MB)..."
    try { Invoke-WebRequest -Uri $ModelUrl -OutFile $ModelPath -UseBasicParsing }
    catch { throw "Could not download the model from $ModelUrl. Check that the 'model-v2' release exists on GitHub with ours_v2_best.pt attached." }
    $h = (Get-FileHash $ModelPath -Algorithm SHA256).Hash.ToLower()
    if ($h -ne $ModelSha) { throw "Model download is corrupt (checksum mismatch). Delete $ModelPath and run again." }
    Write-Host "Model downloaded and checked."
}

Step "Check"
Run $vpy @("-c", "import torch, cv2, tkinter; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('opencv', cv2.__version__, '| tk', tkinter.TkVersion)")
& $vpy FINAL\pipeline.py track-ants -- --help | Out-Null
if ($LASTEXITCODE -ne 0) { throw "The tracker does not start - send the messages above to Can." }
Write-Host "Tracker starts OK."

Step "Speed test (picks GPU or CPU for this computer)"
Run $vpy @("FINAL\check_speed.py")

Write-Host "`nDone. Double-click 'Ant Tracker.bat' to open the app." -ForegroundColor Green
