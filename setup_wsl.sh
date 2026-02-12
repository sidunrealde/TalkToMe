#!/bin/bash
# =============================================================================
#  TalkToMe — WSL2 Setup Script
#  Sets up Python venv, installs all deps, copies checkpoints, builds TRT
#  engines, pre-compiles Cython, creates .env, and writes a run script.
#
#  Usage (run from INSIDE WSL2 Ubuntu):
#    cd /mnt/f/Projects/LLM/TalkToMe   # or wherever the repo lives on Windows
#    chmod +x setup_wsl.sh
#    ./setup_wsl.sh
#
#  Prerequisites:
#    - WSL2 with Ubuntu 22.04+ (systemd enabled)
#    - NVIDIA driver installed on Windows (GPU passthrough)
#    - Ollama running on Windows with OLLAMA_HOST=0.0.0.0:11434
#    - Model checkpoints already downloaded into checkpoints/
# =============================================================================

set -e  # Exit on any error

# ---------------------------------------------------------------------------
# 0. Colors & helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

ok()   { echo -e "${GREEN}✓ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠ $*${NC}"; }
fail() { echo -e "${RED}✗ $*${NC}"; }
info() { echo -e "${CYAN}→ $*${NC}"; }

# ---------------------------------------------------------------------------
# 1. Detect paths
# ---------------------------------------------------------------------------
echo "=============================================="
echo "  TalkToMe — WSL2 Setup"
echo "=============================================="
echo ""

# Detect where the script lives (the Windows repo mount)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if echo "$SCRIPT_DIR" | grep -q "^/mnt/"; then
    WINDOWS_PROJECT_DIR="$SCRIPT_DIR"
    info "Windows repo detected at: $WINDOWS_PROJECT_DIR"
else
    # Fallback: assume standard path
    WINDOWS_PROJECT_DIR="/mnt/f/Projects/LLM/TalkToMe"
    warn "Could not auto-detect Windows path, using: $WINDOWS_PROJECT_DIR"
fi

PROJECT_NAME="TalkToMe"
WSL_PROJECT_DIR="$HOME/$PROJECT_NAME"
VENV_DIR="$WSL_PROJECT_DIR/venv"

info "WSL project dir: $WSL_PROJECT_DIR"
echo ""

# ---------------------------------------------------------------------------
# 2. Preflight checks
# ---------------------------------------------------------------------------
echo "Step 1: Preflight checks..."

# Must be WSL
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    fail "This script must run inside WSL2."
    echo "  Install with:  wsl --install -d Ubuntu"
    exit 1
fi
ok "Running in WSL2"

# Must have GPU
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    ok "NVIDIA GPU detected"
else
    fail "nvidia-smi not found — check Windows NVIDIA driver & wsl --update"
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. System packages
# ---------------------------------------------------------------------------
echo ""
echo "Step 2: Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-venv python3-dev python3-pip \
    git git-lfs \
    ffmpeg libsndfile1 portaudio19-dev \
    libespeak-ng1 espeak-ng \
    openssl build-essential rsync \
    coturn \
    > /dev/null 2>&1
ok "System dependencies installed (including coturn TURN server)"

# ---------------------------------------------------------------------------
# 4. Copy project files to WSL-native filesystem
#    (native ext4 is ~10x faster than /mnt/ for Python)
# ---------------------------------------------------------------------------
echo ""
echo "Step 3: Syncing project files to WSL-native filesystem..."

mkdir -p "$WSL_PROJECT_DIR"

# Sync source code (fast — excludes big stuff)
rsync -a --delete \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='checkpoints' \
    --exclude='node_modules' \
    --exclude='.env' \
    "$WINDOWS_PROJECT_DIR/" "$WSL_PROJECT_DIR/"
ok "Source files synced"

# ---------------------------------------------------------------------------
# 5. Copy checkpoints (large — skip if unchanged)
# ---------------------------------------------------------------------------
echo ""
echo "Step 4: Copying model checkpoints (may take a while on first run)..."
mkdir -p "$WSL_PROJECT_DIR/checkpoints"

for dir in ditto_cfg ditto_onnx ditto_pytorch; do
    SRC="$WINDOWS_PROJECT_DIR/checkpoints/$dir"
    DST="$WSL_PROJECT_DIR/checkpoints/$dir"
    if [ -d "$SRC" ]; then
        info "  Syncing $dir..."
        rsync -a "$SRC/" "$DST/"
    else
        warn "  $dir not found at $SRC — skipping"
    fi
done

# Copy existing TRT engines if present (avoids re-building)
TRT_SRC="$WINDOWS_PROJECT_DIR/checkpoints/ditto_trt"
TRT_DST="$WSL_PROJECT_DIR/checkpoints/ditto_trt"
if [ -d "$TRT_SRC" ]; then
    info "  Syncing ditto_trt (pre-built engines)..."
    rsync -a "$TRT_SRC/" "$TRT_DST/"
fi

# Copy avatars
if [ -d "$WINDOWS_PROJECT_DIR/avatars" ]; then
    rsync -a "$WINDOWS_PROJECT_DIR/avatars/" "$WSL_PROJECT_DIR/avatars/"
fi

ok "Checkpoints synced"

# ---------------------------------------------------------------------------
# 6. Python virtual environment
# ---------------------------------------------------------------------------
echo ""
echo "Step 5: Setting up Python virtual environment..."

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    info "Existing venv found — reusing"
else
    info "Creating new venv..."
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR" --without-pip
    source "$VENV_DIR/bin/activate"
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 > /dev/null
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools -q
ok "Virtual environment ready ($(python3 --version))"

# ---------------------------------------------------------------------------
# 7. Install Python dependencies
# ---------------------------------------------------------------------------
echo ""
echo "Step 6: Installing Python dependencies..."

# PyTorch + CUDA 12.4
info "Installing PyTorch (CUDA 12.4)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q

# TensorRT
info "Installing TensorRT..."
pip install tensorrt -q

# ONNX Runtime GPU
info "Installing ONNX Runtime GPU..."
pip install onnxruntime-gpu -q

# Base requirements
info "Installing base requirements..."
pip install -r "$WSL_PROJECT_DIR/requirements.txt" -q

# WSL-specific extras
info "Installing WSL extras (Cython, imageio, etc.)..."
pip install -r "$WSL_PROJECT_DIR/requirements_wsl.txt" -q

ok "All Python dependencies installed"

# ---------------------------------------------------------------------------
# 8. Pre-compile Cython blend module
# ---------------------------------------------------------------------------
echo ""
echo "Step 7: Pre-compiling Cython modules..."
cd "$WSL_PROJECT_DIR"
python3 -c "
import sys, os
sys.path.insert(0, os.path.join('$WSL_PROJECT_DIR', 'ditto'))
from ditto.core.utils.blend import blend_images_cy
print('Cython blend module compiled OK')
" 2>/dev/null && ok "Cython modules compiled" || warn "Cython compilation had warnings (non-fatal)"

# ---------------------------------------------------------------------------
# 9. Build TensorRT engines (if ONNX models exist and engines don't)
# ---------------------------------------------------------------------------
echo ""
echo "Step 8: Building TensorRT engines..."

ONNX_DIR="$WSL_PROJECT_DIR/checkpoints/ditto_onnx"
TRT_DIR="$WSL_PROJECT_DIR/checkpoints/ditto_trt"
mkdir -p "$TRT_DIR"

if [ ! -d "$ONNX_DIR" ]; then
    warn "No ONNX models found at $ONNX_DIR — skipping TRT build"
    warn "You'll need to use PyTorch mode (see .env)"
else
    # Define models: name, precision, extra polygraphy args
    declare -A MODELS=(
        ["appearance_extractor"]="fp16"
        ["motion_extractor"]="fp32"
        ["stitch_network"]="fp16"
        ["decoder"]="fp16"
        ["lmdm_v0.4_hubert"]="fp32"
        ["insightface_det"]="fp16"
        ["landmark106"]="fp16"
        ["landmark203"]="fp16"
        ["blaze_face"]="fp16"
        ["face_mesh"]="fp16"
    )

    # hubert needs special dynamic shapes
    HUBERT_ARGS="--trt-min-shapes input_values:[1,3200] --trt-opt-shapes input_values:[1,16000] --trt-max-shapes input_values:[1,48000]"

    BUILT=0
    SKIPPED=0
    FAILED=0

    for model in "${!MODELS[@]}"; do
        prec="${MODELS[$model]}"
        onnx="$ONNX_DIR/${model}.onnx"
        engine="$TRT_DIR/${model}_${prec}.engine"

        if [ ! -f "$onnx" ]; then
            continue
        fi

        if [ -f "$engine" ]; then
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        info "  Building $model ($prec)..."
        EXTRA_ARGS=""
        if [ "$prec" = "fp16" ]; then
            EXTRA_ARGS="--fp16"
        fi

        if polygraphy convert "$onnx" -o "$engine" \
            --convert-to trt $EXTRA_ARGS 2>/dev/null; then
            BUILT=$((BUILT + 1))
        else
            FAILED=$((FAILED + 1))
            warn "  Failed to build $model"
        fi
    done

    # Build hubert separately (dynamic shapes)
    HUBERT_ONNX="$ONNX_DIR/hubert.onnx"
    HUBERT_ENGINE="$TRT_DIR/hubert_fp32.engine"
    if [ -f "$HUBERT_ONNX" ] && [ ! -f "$HUBERT_ENGINE" ]; then
        info "  Building hubert (fp32, dynamic shapes)..."
        if polygraphy convert "$HUBERT_ONNX" -o "$HUBERT_ENGINE" \
            --convert-to trt \
            --trt-min-shapes input_values:[1,3200] \
            --trt-opt-shapes input_values:[1,16000] \
            --trt-max-shapes input_values:[1,48000] 2>/dev/null; then
            BUILT=$((BUILT + 1))
        else
            FAILED=$((FAILED + 1))
            warn "  Failed to build hubert"
        fi
    elif [ -f "$HUBERT_ENGINE" ]; then
        SKIPPED=$((SKIPPED + 1))
    fi

    ok "TRT build complete: $BUILT built, $SKIPPED skipped, $FAILED failed"

    # --- warp_network: symlink PyTorch weights (no TRT support for GridSample3D) ---
    WARP_PTH_SRC="$WSL_PROJECT_DIR/checkpoints/ditto_pytorch/models/warp_network.pth"
    WARP_PTH_DST="$TRT_DIR/warp_network.pth"
    if [ -f "$WARP_PTH_SRC" ] && [ ! -f "$WARP_PTH_DST" ]; then
        cp "$WARP_PTH_SRC" "$WARP_PTH_DST"
        ok "Copied warp_network.pth to TRT dir (hybrid mode)"
    fi

    # --- Copy hubert.onnx for the streaming module ---
    if [ -f "$ONNX_DIR/hubert.onnx" ] && [ ! -f "$TRT_DIR/hubert.onnx" ]; then
        cp "$ONNX_DIR/hubert.onnx" "$TRT_DIR/hubert.onnx"
    fi
    # --- Copy warp_network.onnx for ONNX-fallback modes ---
    if [ -f "$ONNX_DIR/warp_network.onnx" ] && [ ! -f "$TRT_DIR/warp_network.onnx" ]; then
        cp "$ONNX_DIR/warp_network.onnx" "$TRT_DIR/warp_network.onnx"
    fi
fi

# ---------------------------------------------------------------------------
# 10. Generate hybrid TRT config (TRT + PyTorch warp_network)
# ---------------------------------------------------------------------------
echo ""
echo "Step 9: Generating TRT hybrid config..."

CFG_DIR="$WSL_PROJECT_DIR/checkpoints/ditto_cfg"
HYBRID_CFG="$CFG_DIR/v0.4_hubert_cfg_trt_hybrid_online.pkl"

if [ -f "$HYBRID_CFG" ]; then
    ok "Hybrid config already exists"
else
    # Generate from the base TRT config, replacing warp_network engine with .pth
    python3 - << 'PYEOF'
import pickle, os, copy

cfg_dir = os.path.expanduser("~/TalkToMe/checkpoints/ditto_cfg")
base_cfg_path = os.path.join(cfg_dir, "v0.4_hubert_cfg_trt_online.pkl")
out_path = os.path.join(cfg_dir, "v0.4_hubert_cfg_trt_hybrid_online.pkl")

if not os.path.exists(base_cfg_path):
    # Fallback: try non-online version
    base_cfg_path = os.path.join(cfg_dir, "v0.4_hubert_cfg_trt.pkl")

if not os.path.exists(base_cfg_path):
    print("WARNING: No base TRT config found. Hybrid config not generated.")
    exit(0)

with open(base_cfg_path, "rb") as f:
    cfg = pickle.load(f)

# Deep copy and modify warp_network to use .pth instead of .engine
cfg_new = copy.deepcopy(cfg)
for section in cfg_new:
    if isinstance(section, dict):
        for key, val in section.items():
            if isinstance(val, str) and "warp_network" in val and val.endswith(".engine"):
                section[key] = val.replace(".engine", ".pth").replace("_fp16", "").replace("_fp32", "")
                print(f"  Replaced {key}: {val} -> {section[key]}")

# Set online_mode
for section in cfg_new:
    if isinstance(section, dict) and "online_mode" in section:
        section["online_mode"] = True

with open(out_path, "wb") as f:
    pickle.dump(cfg_new, f)

print(f"Generated: {out_path}")
PYEOF
    ok "Hybrid config generated"
fi

# ---------------------------------------------------------------------------
# 11. Detect Ollama on Windows host & create .env
# ---------------------------------------------------------------------------
echo ""
echo "Step 10: Creating .env configuration..."

# Auto-detect how to reach Ollama on the Windows host.
# WSL2 mirrored networking:  localhost reaches Windows → use localhost.
# WSL2 NAT networking:       must go via the default-gateway IP.
OLLAMA_URL=""

# 1. Try localhost first (works with mirrored networking)
if curl -s --connect-timeout 3 "http://localhost:11434/" >/dev/null 2>&1; then
    OLLAMA_URL="http://localhost:11434"
    ok "Ollama reachable at $OLLAMA_URL (mirrored/localhost)"
fi

# 2. Fall back to gateway IP (NAT networking)
if [ -z "$OLLAMA_URL" ]; then
    WIN_GATEWAY=$(ip route show default 2>/dev/null | awk '{print $3}')
    if [ -z "$WIN_GATEWAY" ]; then
        WIN_GATEWAY="172.31.0.1"
        warn "Could not detect gateway, defaulting to $WIN_GATEWAY"
    fi
    info "Windows host gateway: $WIN_GATEWAY"

    if curl -s --connect-timeout 3 "http://${WIN_GATEWAY}:11434/" >/dev/null 2>&1; then
        OLLAMA_URL="http://${WIN_GATEWAY}:11434"
        ok "Ollama reachable at $OLLAMA_URL (via gateway)"
    else
        # Default to localhost and warn
        OLLAMA_URL="http://localhost:11434"
        warn "Ollama not reachable at http://${WIN_GATEWAY}:11434 or localhost"
        warn "Make sure Ollama is running on Windows with OLLAMA_HOST=0.0.0.0:11434"
        warn "  Set via Windows env var: [System.Environment]::SetEnvironmentVariable('OLLAMA_HOST','0.0.0.0:11434','User')"
    fi
fi

# Determine best Ditto mode
if [ -d "$TRT_DIR" ] && ls "$TRT_DIR"/*.engine 1>/dev/null 2>&1; then
    DITTO_CFG="checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_hybrid_online.pkl"
    DITTO_CKPT="checkpoints/ditto_trt"
    info "Using TRT hybrid mode (fastest)"
else
    DITTO_CFG="checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
    DITTO_CKPT="checkpoints/ditto_pytorch"
    info "Using PyTorch mode (no TRT engines found)"
fi

# Preserve existing GROQ key if .env already exists
EXISTING_GROQ=""
if [ -f "$WSL_PROJECT_DIR/.env" ]; then
    EXISTING_GROQ=$(grep -oP 'GROQ_API_KEY=\K.*' "$WSL_PROJECT_DIR/.env" 2>/dev/null || true)
fi
if [ -z "$EXISTING_GROQ" ] || [ "$EXISTING_GROQ" = "your_groq_api_key_here" ]; then
    EXISTING_GROQ="your_groq_api_key_here"
fi

cat > "$WSL_PROJECT_DIR/.env" << EOF
# TalkToMe WSL2 Configuration  (auto-generated by setup_wsl.sh)

# Groq API key for cloud STT — get one at https://console.groq.com
GROQ_API_KEY=${EXISTING_GROQ}

# Ollama (running on Windows host)
OLLAMA_BASE_URL=${OLLAMA_URL}
OLLAMA_MODEL=mistral:7b

# Ditto video generation
DITTO_CONFIG_PATH=${DITTO_CFG}
DITTO_CHECKPOINT_PATH=${DITTO_CKPT}
DITTO_REALTIME=true

# Server
HOST=0.0.0.0
PORT=8765
EOF
ok ".env created"

# ---------------------------------------------------------------------------
# 12. Configure coturn TURN server for WSL2 WebRTC
#     In WSL2 mirrored networking, direct UDP between Windows and WSL is
#     broken on the LAN IP.  Only TCP via 127.0.0.1 (shared loopback)
#     works bidirectionally.  coturn relays all WebRTC media over TCP.
# ---------------------------------------------------------------------------
echo ""
echo "Step 11: Configuring TURN relay server..."

TURN_CONF="$WSL_PROJECT_DIR/turn/turnserver.conf"
TURN_PORT=3479
TURN_USER="talktome"
TURN_PASS="talktome123"
TURN_REALM="talktome"

mkdir -p "$WSL_PROJECT_DIR/turn"
cat > "$TURN_CONF" << TURNEOF
# coturn config — auto-generated by setup_wsl.sh
# Listens only on 127.0.0.1 (shared loopback between Windows & WSL2)
listening-ip=127.0.0.1
listening-port=${TURN_PORT}
relay-ip=127.0.0.1
min-port=49152
max-port=49252
realm=${TURN_REALM}
lt-cred-mech
user=${TURN_USER}:${TURN_PASS}
no-tls
no-dtls
no-cli
log-file=/tmp/coturn.log
simple-log
pidfile=/tmp/coturn-talktome.pid
TURNEOF
ok "TURN config written to $TURN_CONF"
info "  Port: $TURN_PORT  |  User: $TURN_USER  |  Relay: 127.0.0.1"

# Disable system coturn service (we manage it ourselves in run.sh)
sudo systemctl disable coturn 2>/dev/null || true
sudo systemctl stop coturn 2>/dev/null || true

# ---------------------------------------------------------------------------
# 13. Create run.sh convenience script
# ---------------------------------------------------------------------------
echo ""
echo "Step 12: Creating run script..."

cat > "$WSL_PROJECT_DIR/run.sh" << 'RUNEOF'
#!/bin/bash
# ============================================================================
#  TalkToMe — Run Script  (auto-generated by setup_wsl.sh)
#  Starts coturn TURN relay + Python server, cleans up on exit.
#
#  Usage:  ./run.sh              (foreground)
#          ./run.sh --background  (tmux, survives terminal close)
# ============================================================================
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
source venv/bin/activate

TURN_CONF="$ROOT/turn/turnserver.conf"
TURN_PID="/tmp/coturn-talktome.pid"

# ------------------------------------------------------------------
# coturn management
# ------------------------------------------------------------------
start_turn() {
    # Skip if not in WSL (non-WSL doesn't need TURN)
    if ! grep -qi microsoft /proc/version 2>/dev/null; then
        return 0
    fi
    if [ ! -f "$TURN_CONF" ]; then
        echo "[TURN] Config not found at $TURN_CONF — skipping (run setup_wsl.sh)"
        return 0
    fi

    # Kill any leftover coturn from a previous run
    if [ -f "$TURN_PID" ] && kill -0 "$(cat "$TURN_PID")" 2>/dev/null; then
        echo "[TURN] Stopping previous coturn (pid $(cat $TURN_PID))..."
        kill "$(cat "$TURN_PID")" 2>/dev/null || true
        sleep 1
    fi

    echo "[TURN] Starting coturn relay on 127.0.0.1..."
    turnserver -c "$TURN_CONF" &
    TURN_SERVER_PID=$!
    sleep 1

    if kill -0 "$TURN_SERVER_PID" 2>/dev/null; then
        echo "[TURN] ✓ coturn running (pid $TURN_SERVER_PID)"
    else
        echo "[TURN] ✗ coturn failed to start — check /tmp/coturn.log"
        echo "[TURN]   (WebRTC may not connect in WSL2 without TURN)"
    fi
}

stop_turn() {
    if [ -f "$TURN_PID" ] && kill -0 "$(cat "$TURN_PID")" 2>/dev/null; then
        echo "[TURN] Stopping coturn (pid $(cat $TURN_PID))..."
        kill "$(cat "$TURN_PID")" 2>/dev/null || true
    fi
    # Also catch any strays we started
    if [ -n "${TURN_SERVER_PID:-}" ]; then
        kill "$TURN_SERVER_PID" 2>/dev/null || true
    fi
}

# Clean up on exit (Ctrl+C, kill, etc.)
cleanup() {
    echo ""
    echo "Shutting down..."
    stop_turn
    # Kill any child processes
    jobs -p | xargs -r kill 2>/dev/null || true
    echo "Goodbye!"
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------
# Start everything
# ------------------------------------------------------------------
start_turn

if [ "$1" = "--background" ] || [ "$1" = "-b" ]; then
    tmux kill-session -t talktome 2>/dev/null || true
    # In background mode, start a fresh tmux that also manages coturn
    tmux new-session -d -s talktome \
        "cd $ROOT && source venv/bin/activate && python -m src.server 2>&1 | tee /tmp/talktome.log"
    echo ""
    echo "Server started in tmux session 'talktome'"
    echo "  View logs:  tmux attach -t talktome"
    echo "  Or:         tail -f /tmp/talktome.log"
    echo "  Stop:       tmux kill-session -t talktome"
else
    python -m src.server
fi
RUNEOF
chmod +x "$WSL_PROJECT_DIR/run.sh"
ok "run.sh created"

# ---------------------------------------------------------------------------
# 14. Final verification
# ---------------------------------------------------------------------------
echo ""
echo "Step 13: Verifying installation..."

cd "$WSL_PROJECT_DIR"
python3 -c "
import sys
errors = []

# Core
try: import torch; assert torch.cuda.is_available(), 'CUDA not available'
except Exception as e: errors.append(f'PyTorch/CUDA: {e}')

try: import pipecat
except Exception as e: errors.append(f'Pipecat: {e}')

try: import fastapi
except Exception as e: errors.append(f'FastAPI: {e}')

try: from kokoro_onnx import Kokoro
except Exception as e: errors.append(f'Kokoro TTS: {e}')

# Ditto deps
try: import filetype
except Exception as e: errors.append(f'filetype: {e}')

try: import imageio
except Exception as e: errors.append(f'imageio: {e}')

try: import cv2
except Exception as e: errors.append(f'opencv: {e}')

try: import einops
except Exception as e: errors.append(f'einops: {e}')

try: import Cython
except Exception as e: errors.append(f'Cython: {e}')

try: import scipy
except Exception as e: errors.append(f'scipy: {e}')

try: import librosa
except Exception as e: errors.append(f'librosa: {e}')

try: import skimage
except Exception as e: errors.append(f'scikit-image: {e}')

# TRT
try: import tensorrt
except Exception as e: errors.append(f'TensorRT: {e}')

try: import onnxruntime
except Exception as e: errors.append(f'ONNX Runtime: {e}')

# Report
if errors:
    print('Issues found:')
    for e in errors:
        print(f'  ✗ {e}')
    sys.exit(1)
else:
    print('All dependencies verified OK')
"

if [ $? -eq 0 ]; then
    ok "All verifications passed"
else
    warn "Some verifications failed — see above"
fi

# ---------------------------------------------------------------------------
# Done!
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo -e "${GREEN}  Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "Project location: $WSL_PROJECT_DIR"
echo ""
echo "To run the server:"
echo "  cd $WSL_PROJECT_DIR"
echo "  ./run.sh                  # foreground (auto-starts TURN relay)"
echo "  ./run.sh --background     # tmux (persistent)"
echo ""
echo "Access from Windows: https://localhost:8765"
echo ""
echo -e "${YELLOW}Important:${NC}"
echo "  1. Update .env with your GROQ_API_KEY (for cloud STT)"
echo "  2. Make sure Ollama is running on Windows:"
echo "     - Set env var: OLLAMA_HOST=0.0.0.0:11434"
echo "     - Then restart Ollama"
echo "  3. Upload an avatar image in the web UI"
echo ""
echo -e "${CYAN}WebRTC Note:${NC}"
echo "  The TURN relay server (coturn) starts automatically with ./run.sh"
echo "  No separate setup needed — WebRTC works out of the box in WSL2."
echo ""
echo -e "${CYAN}LAN Access (other devices on your network):${NC}"
echo "  Run from Windows PowerShell (Administrator):"
echo "    cd $(echo $WINDOWS_PROJECT_DIR | sed 's|/mnt/f|F:|')"
echo "    .\\setup_lan.ps1"
echo "  This sets up port forwarding, firewall, and SSL certs."
echo ""
