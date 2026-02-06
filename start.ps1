# TalkToMe - Automated Setup and Launch Script
# Run this script in PowerShell: .\start.ps1

param(
    [switch]$SkipInstall,
    [switch]$SkipOllama,
    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   TalkToMe - AI Avatar Application    " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# =============================================================================
# Helper Functions
# =============================================================================

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host ">> $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "   [WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "   [ERROR] $Message" -ForegroundColor Red
}

function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# =============================================================================
# Step 1: Check Python Installation
# =============================================================================

Write-Step "Checking Python installation..."

$PythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    if (Test-Command $cmd) {
        $version = & $cmd --version 2>&1
        if ($version -match "Python 3\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -ge 10) {
                $PythonCmd = $cmd
                Write-Host "   Found: $version" -ForegroundColor Gray
                break
            }
        }
    }
}

if (-not $PythonCmd) {
    Write-Error "Python 3.10+ is required but not found."
    Write-Host "   Please install Python from https://www.python.org/downloads/" -ForegroundColor Gray
    exit 1
}

# =============================================================================
# Step 2: Create/Activate Virtual Environment
# =============================================================================

Write-Step "Setting up virtual environment..."

$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvPip = Join-Path $VenvPath "Scripts\pip.exe"
$VenvActivate = Join-Path $VenvPath "Scripts\Activate.ps1"

if (-not (Test-Path $VenvPython)) {
    Write-Host "   Creating virtual environment..." -ForegroundColor Gray
    & $PythonCmd -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment"
        exit 1
    }
}

Write-Host "   Activating virtual environment..." -ForegroundColor Gray
. $VenvActivate

# =============================================================================
# Step 3: Install Dependencies
# =============================================================================

if (-not $SkipInstall) {
    Write-Step "Installing dependencies (this may take a few minutes)..."
    
    # Upgrade pip
    Write-Host "   Upgrading pip..." -ForegroundColor Gray
    & $VenvPython -m pip install --upgrade pip --quiet
    
    # Check if PyTorch is installed with CUDA
    $TorchInstalled = & $VenvPython -c "import torch; print(torch.cuda.is_available())" 2>$null
    
    if ($TorchInstalled -ne "True") {
        Write-Host "   Installing PyTorch with CUDA support..." -ForegroundColor Gray
        & $VenvPip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "CUDA PyTorch installation failed, trying CPU version..."
            & $VenvPip install torch torchvision torchaudio --quiet
        }
    } else {
        Write-Host "   PyTorch with CUDA already installed" -ForegroundColor Gray
    }
    
    # Install main requirements
    Write-Host "   Installing project dependencies..." -ForegroundColor Gray
    & $VenvPip install -r requirements.txt --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install requirements.txt"
        exit 1
    }
    
    # Install additional dependencies for Ditto
    Write-Host "   Installing Ditto dependencies..." -ForegroundColor Gray
    & $VenvPip install mediapipe einops resampy --quiet
    
    Write-Host "   Dependencies installed successfully!" -ForegroundColor Gray
} else {
    Write-Host "   Skipping dependency installation (--SkipInstall)" -ForegroundColor Gray
}

# =============================================================================
# Step 4: Initialize Git Submodules
# =============================================================================

Write-Step "Checking Git submodules..."

$DittoPath = Join-Path $ProjectRoot "ditto"
if (-not (Test-Path (Join-Path $DittoPath "core"))) {
    if (Test-Command "git") {
        Write-Host "   Initializing Ditto submodule..." -ForegroundColor Gray
        git submodule update --init --recursive
    } else {
        Write-Warning "Git not found. Please run: git submodule update --init --recursive"
    }
} else {
    Write-Host "   Ditto submodule already initialized" -ForegroundColor Gray
}

# =============================================================================
# Step 5: Download Model Checkpoints
# =============================================================================

Write-Step "Checking model checkpoints..."

$CheckpointsPath = Join-Path $ProjectRoot "checkpoints"
$DittoPytorchPath = Join-Path $CheckpointsPath "ditto_pytorch"

if (-not (Test-Path $DittoPytorchPath)) {
    Write-Host "   Downloading Ditto model checkpoints (~3GB)..." -ForegroundColor Gray
    Write-Host "   This may take several minutes depending on your connection." -ForegroundColor Gray
    
    & $VenvPython -c @"
from huggingface_hub import snapshot_download
snapshot_download(
    'digital-avatar/ditto-talkinghead',
    local_dir='checkpoints',
    allow_patterns=['ditto_pytorch/*', 'ditto_cfg/v0.4_hubert_cfg_pytorch.pkl']
)
print('Download complete!')
"@
    
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed to download checkpoints automatically."
        Write-Host "   Please download manually from: https://huggingface.co/digital-avatar/ditto-talkinghead" -ForegroundColor Gray
    }
} else {
    Write-Host "   Model checkpoints already present" -ForegroundColor Gray
}

# =============================================================================
# Step 6: Create .env if missing
# =============================================================================

Write-Step "Checking configuration..."

$EnvFile = Join-Path $ProjectRoot ".env"
$EnvExample = Join-Path $ProjectRoot ".env.example"

if (-not (Test-Path $EnvFile)) {
    if (Test-Path $EnvExample) {
        Copy-Item $EnvExample $EnvFile
        Write-Host "   Created .env from .env.example" -ForegroundColor Gray
    } else {
        # Create default .env
        @"
# Local Whisper STT Settings
WHISPER_MODEL=large-v3-turbo
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b

# Server Configuration
HOST=0.0.0.0
PORT=8765
"@ | Out-File -FilePath $EnvFile -Encoding utf8
        Write-Host "   Created default .env configuration" -ForegroundColor Gray
    }
} else {
    Write-Host "   Configuration file exists" -ForegroundColor Gray
}

# =============================================================================
# Step 7: Check Ollama
# =============================================================================

if (-not $SkipOllama) {
    Write-Step "Checking Ollama..."
    
    $OllamaRunning = $false
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction SilentlyContinue
        $OllamaRunning = $true
    } catch {
        $OllamaRunning = $false
    }
    
    if ($OllamaRunning) {
        Write-Host "   Ollama is running" -ForegroundColor Gray
        
        # Check if model is available
        $models = (Invoke-WebRequest -Uri "http://localhost:11434/api/tags" | ConvertFrom-Json).models
        $hasModel = $models | Where-Object { $_.name -match "mistral" }
        
        if (-not $hasModel) {
            Write-Warning "mistral:7b model not found. Pulling model..."
            if (Test-Command "ollama") {
                ollama pull mistral:7b
            } else {
                Write-Host "   Run manually: ollama pull mistral:7b" -ForegroundColor Gray
            }
        }
    } else {
        Write-Warning "Ollama is not running!"
        Write-Host "   Please start Ollama in another terminal: ollama serve" -ForegroundColor Gray
        Write-Host "   Then pull the model: ollama pull mistral:7b" -ForegroundColor Gray
        Write-Host ""
        
        $continue = Read-Host "   Continue anyway? (y/N)"
        if ($continue -ne "y" -and $continue -ne "Y") {
            exit 0
        }
    }
}

# =============================================================================
# Step 8: Launch Application
# =============================================================================

Write-Step "Launching TalkToMe..."
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Server starting on port 8765        " -ForegroundColor Cyan
Write-Host "   Open: http://localhost:8765         " -ForegroundColor Cyan
Write-Host "   Press Ctrl+C to stop                " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Launch the server
& $VenvPython -m src.server
