#!/bin/bash
# TalkToMe - Automated Setup and Launch Script for Linux/Mac
# Run: chmod +x start.sh && ./start.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}   TalkToMe - AI Avatar Application    ${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# =============================================================================
# Helper Functions
# =============================================================================

step() {
    echo ""
    echo -e "${GREEN}>> $1${NC}"
}

warn() {
    echo -e "${YELLOW}   [WARNING] $1${NC}"
}

error() {
    echo -e "${RED}   [ERROR] $1${NC}"
}

# =============================================================================
# Step 1: Check Python Installation
# =============================================================================

step "Checking Python installation..."

PYTHON_CMD=""
for cmd in python3 python; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd --version 2>&1)
        if [[ $version =~ Python\ 3\.([0-9]+) ]]; then
            minor=${BASH_REMATCH[1]}
            if [ "$minor" -ge 10 ]; then
                PYTHON_CMD=$cmd
                echo "   Found: $version"
                break
            fi
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    error "Python 3.10+ is required but not found."
    echo "   Please install Python from https://www.python.org/downloads/"
    exit 1
fi

# =============================================================================
# Step 2: Create/Activate Virtual Environment
# =============================================================================

step "Setting up virtual environment..."

VENV_PATH="$PROJECT_ROOT/.venv"
VENV_PYTHON="$VENV_PATH/bin/python"
VENV_PIP="$VENV_PATH/bin/pip"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "   Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_PATH"
fi

echo "   Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# =============================================================================
# Step 3: Install Dependencies
# =============================================================================

if [ "$1" != "--skip-install" ]; then
    step "Installing dependencies (this may take a few minutes)..."
    
    # Upgrade pip
    echo "   Upgrading pip..."
    $VENV_PYTHON -m pip install --upgrade pip --quiet
    
    # Check if PyTorch is installed with CUDA
    TORCH_CUDA=$($VENV_PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    
    if [ "$TORCH_CUDA" != "True" ]; then
        echo "   Installing PyTorch with CUDA support..."
        $VENV_PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet 2>/dev/null || {
            warn "CUDA PyTorch installation failed, trying CPU version..."
            $VENV_PIP install torch torchvision torchaudio --quiet
        }
    else
        echo "   PyTorch with CUDA already installed"
    fi
    
    # Install main requirements
    echo "   Installing project dependencies..."
    $VENV_PIP install -r requirements.txt --quiet
    
    # Install additional dependencies for Ditto
    echo "   Installing Ditto dependencies..."
    $VENV_PIP install mediapipe einops resampy --quiet
    
    echo "   Dependencies installed successfully!"
else
    echo "   Skipping dependency installation (--skip-install)"
fi

# =============================================================================
# Step 4: Initialize Git Submodules
# =============================================================================

step "Checking Git submodules..."

if [ ! -d "$PROJECT_ROOT/ditto/core" ]; then
    if command -v git &> /dev/null; then
        echo "   Initializing Ditto submodule..."
        git submodule update --init --recursive
    else
        warn "Git not found. Please run: git submodule update --init --recursive"
    fi
else
    echo "   Ditto submodule already initialized"
fi

# =============================================================================
# Step 5: Download Model Checkpoints
# =============================================================================

step "Checking model checkpoints..."

if [ ! -d "$PROJECT_ROOT/checkpoints/ditto_pytorch" ]; then
    echo "   Downloading Ditto model checkpoints (~3GB)..."
    echo "   This may take several minutes depending on your connection."
    
    $VENV_PYTHON -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'digital-avatar/ditto-talkinghead',
    local_dir='checkpoints',
    allow_patterns=['ditto_pytorch/*', 'ditto_cfg/v0.4_hubert_cfg_pytorch.pkl']
)
print('Download complete!')
" || {
        warn "Failed to download checkpoints automatically."
        echo "   Please download manually from: https://huggingface.co/digital-avatar/ditto-talkinghead"
    }
else
    echo "   Model checkpoints already present"
fi

# =============================================================================
# Step 6: Create .env if missing
# =============================================================================

step "Checking configuration..."

if [ ! -f "$PROJECT_ROOT/.env" ]; then
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        echo "   Created .env from .env.example"
    else
        cat > "$PROJECT_ROOT/.env" << EOF
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
EOF
        echo "   Created default .env configuration"
    fi
else
    echo "   Configuration file exists"
fi

# =============================================================================
# Step 7: Check Ollama
# =============================================================================

step "Checking Ollama..."

if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "   Ollama is running"
else
    warn "Ollama is not running!"
    echo "   Please start Ollama in another terminal: ollama serve"
    echo "   Then pull the model: ollama pull mistral:7b"
    echo ""
    read -p "   Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# =============================================================================
# Step 8: Launch Application
# =============================================================================

step "Launching TalkToMe..."
echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}   Server starting on port 8765        ${NC}"
echo -e "${CYAN}   Open: http://localhost:8765         ${NC}"
echo -e "${CYAN}   Press Ctrl+C to stop                ${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Launch the server
$VENV_PYTHON -m src.server
