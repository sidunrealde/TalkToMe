# 🎭 TalkToMe - AI Avatar with Real-Time Lip Sync

A conversational AI avatar web application that brings portrait images to life with realistic lip-sync animation. Uses Pipecat for real-time conversation orchestration and Ditto TalkingHead for audio-driven facial animation.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- 🎭 **Realistic Lip-Sync** - Animate any portrait photo with accurate mouth movements
- 🎤 **Speech-to-Text** - Real-time transcription with Groq Whisper
- 🤖 **LLM Conversations** - Powered by Ollama (local) or any OpenAI-compatible API
- 🔊 **Natural TTS** - High-quality voice synthesis with Kokoro TTS
- 🎯 **Voice Activity Detection** - Smart interruption handling
- 🌐 **Network Access** - HTTPS support for accessing from other devices
- 🎨 **Multiple Voices** - Choose from various voice options

---

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 3090, RTX 4090)
- **VRAM**: Minimum 8GB recommended
- **RAM**: 16GB+ recommended

### Software
- Python 3.10+
- NVIDIA CUDA Toolkit 12.x
- Ollama (for local LLM)
- Groq API key (for STT)

---

## 🚀 Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/TalkToMe.git
cd TalkToMe
git submodule update --init --recursive
```

### Step 2: Create Python Environment

**Option A: Using venv (Recommended)**
```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

**Option B: Using Conda**
```bash
conda create -n TalkToMe python=3.10 -y
conda activate TalkToMe
```

### Step 3: Install PyTorch with CUDA

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt

# Additional dependencies for Ditto
pip install mediapipe einops resampy
```

### Step 5: Download Ditto Model Checkpoints

```bash
pip install huggingface_hub

# Download PyTorch checkpoints (recommended)
python -c "from huggingface_hub import snapshot_download; snapshot_download('digital-avatar/ditto-talkinghead', local_dir='checkpoints', allow_patterns=['ditto_pytorch/*', 'ditto_cfg/v0.4_hubert_cfg_pytorch.pkl'])"
```

### Step 6: Install and Start Ollama

1. Download Ollama from [ollama.ai](https://ollama.ai)
2. Install and run:

```bash
# Start Ollama server
ollama serve

# In another terminal, pull a model
ollama pull mistral:7b
```

### Step 7: Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env with your settings
```

Edit `.env` file:
```env
# Required: Get your API key from https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b

# Server Configuration
HOST=0.0.0.0
PORT=8765
```

### Step 8: Run the Application

```bash
python -m src.server
```

### Step 9: Open in Browser

Navigate to `http://localhost:8765`

1. 📷 Upload an avatar image (clear frontal face photo works best)
2. 🎙️ Select a voice
3. 🔗 Click "Connect"
4. 💬 Start talking or type messages!

---

## 🌐 Network Access (Access from Other Devices)

The server supports HTTPS for accessing from other devices on your network.

### Generate SSL Certificates

**Windows (PowerShell - Run as Administrator):**
```powershell
# Create certs directory
mkdir certs -Force

# Generate certificate
$cert = New-SelfSignedCertificate -DnsName "YOUR_IP", "localhost" -CertStoreLocation "cert:\CurrentUser\My" -KeyExportPolicy Exportable -NotAfter (Get-Date).AddYears(5)

# Export to PFX
$pwd = ConvertTo-SecureString -String "talktome" -Force -AsPlainText
Export-PfxCertificate -Cert $cert -FilePath "certs\server.pfx" -Password $pwd

# Convert to PEM format (requires cryptography package)
pip install cryptography
python -c "from cryptography.hazmat.primitives.serialization import pkcs12, Encoding, PrivateFormat, NoEncryption; data = open('certs/server.pfx', 'rb').read(); key, cert, _ = pkcs12.load_key_and_certificates(data, b'talktome'); open('certs/key.pem', 'wb').write(key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption())); open('certs/cert.pem', 'wb').write(cert.public_bytes(Encoding.PEM))"
```

**Linux/Mac:**
```bash
mkdir -p certs
openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem -out certs/cert.pem -days 1825 -nodes -subj "/CN=localhost"
```

### Add Firewall Rule (Windows - Run as Administrator)

```powershell
New-NetFirewallRule -DisplayName "TalkToMe Avatar Server" -Direction Inbound -LocalPort 8765 -Protocol TCP -Action Allow
```

### Access from Other Devices

1. Find your IP: `ipconfig` (Windows) or `ip addr` (Linux)
2. Access: `https://YOUR_IP:8765`
3. Accept the self-signed certificate warning in browser

> **Note:** You must use HTTPS (not HTTP) for microphone access from non-localhost devices.

---

## � WSL2 Setup (TensorRT Acceleration)

For maximum performance, you can run TalkToMe inside **WSL2** with TensorRT-accelerated Ditto inference. This uses your Windows NVIDIA GPU via CUDA passthrough and gives ~3-5x faster frame generation than pure PyTorch.

### Why WSL?

- TensorRT pip wheels ship native Linux `.so` libraries — no separate SDK install needed
- The Ditto pipeline uses a **hybrid mode**: 10 models run as TensorRT engines, while `warp_network` uses PyTorch (GridSample3D has no TRT support)
- WSL2's native ext4 filesystem is ~10x faster than `/mnt/` for Python imports

### Prerequisites

| Requirement | Details |
|---|---|
| **WSL2** | Ubuntu 22.04+ (`wsl --install -d Ubuntu`) |
| **NVIDIA Driver** | Installed on Windows (GPU passthrough automatic in WSL2) |
| **Ollama** | Running on Windows with `OLLAMA_HOST=0.0.0.0:11434` |
| **Checkpoints** | Already downloaded into `checkpoints/` (see Step 5 above) |
| **Groq API Key** | For cloud STT — [console.groq.com](https://console.groq.com) |

### One-Command Setup

The `setup_wsl.sh` script handles everything automatically:

```bash
# Open a WSL terminal
wsl

# Navigate to the repo (mounted from Windows)
cd /mnt/f/Projects/LLM/TalkToMe   # adjust to your path

# Run setup
chmod +x setup_wsl.sh
./setup_wsl.sh
```

The script will:
1. Install system packages (`python3-dev`, `ffmpeg`, `espeak-ng`, `build-essential`, etc.)
2. Sync project files to `~/TalkToMe` on WSL's native filesystem
3. Copy model checkpoints (ONNX, PyTorch, config files)
4. Create a Python venv and install all dependencies (PyTorch CUDA 12.4, TensorRT, Pipecat, Kokoro, etc.)
5. Pre-compile the Cython blend module
6. Build TensorRT engines from ONNX models (~10-20 min on first run, skips existing)
7. Copy `warp_network.pth` for hybrid mode
8. Generate the TRT hybrid config pickle
9. Auto-detect the Windows host IP and create `.env` with correct Ollama URL
10. Create a `run.sh` convenience script
11. Verify all dependencies

### Ollama Setup for WSL

WSL2 runs in a separate network namespace, so Ollama on Windows must listen on all interfaces:

**Windows PowerShell (run once):**
```powershell
# Set Ollama to listen on all interfaces
[System.Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0:11434', 'User')

# Restart Ollama (close tray icon, then reopen)
```

The setup script auto-detects the WSL→Windows gateway IP and configures `.env` accordingly.

### Running the Server

```bash
# From WSL
cd ~/TalkToMe

# Foreground (see all logs)
./run.sh

# Background (survives terminal close)
./run.sh --background

# View background logs
tail -f /tmp/talktome.log

# Stop background server
tmux kill-session -t talktome
```

Access from Windows Chrome: **http://localhost:8765**

> **Note:** When running in WSL, use HTTP (not HTTPS). `localhost` is a secure context in Chrome, so microphone/WebRTC still works without SSL certificates.

### Ditto Backend Modes

The setup script auto-selects the best mode based on available files:

| Mode | Config | Checkpoint Dir | Speed | Notes |
|------|--------|----------------|-------|-------|
| **TRT Hybrid** ⚡ | `v0.4_hubert_cfg_trt_hybrid_online.pkl` | `ditto_trt` | Fastest | 10 TRT engines + PyTorch warp |
| PyTorch | `v0.4_hubert_cfg_pytorch.pkl` | `ditto_pytorch` | Slower | No TRT needed, works everywhere |
| Full TRT | `v0.4_hubert_cfg_trt_online.pkl` | `ditto_trt` | Fastest | Requires GridSample3D TRT plugin |

To switch modes, edit `~/TalkToMe/.env`:
```env
# TRT Hybrid (recommended)
DITTO_CONFIG_PATH=checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_hybrid_online.pkl
DITTO_CHECKPOINT_PATH=checkpoints/ditto_trt

# PyTorch fallback
# DITTO_CONFIG_PATH=checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl
# DITTO_CHECKPOINT_PATH=checkpoints/ditto_pytorch
```

### Re-running Setup

The script is safe to re-run — it reuses the existing venv, skips already-built TRT engines, and preserves your Groq API key in `.env`.

---

## �📁 Project Structure

```
TalkToMe/
├── avatars/                    # Avatar images
├── certs/                      # SSL certificates (for HTTPS)
│   ├── cert.pem
│   └── key.pem
├── checkpoints/                # Model weights (gitignored)
│   ├── ditto_cfg/              # Ditto config pickles
│   ├── ditto_pytorch/          # PyTorch model files
│   ├── ditto_onnx/             # ONNX model files
│   └── ditto_trt/              # TensorRT engines (built by setup_wsl.sh)
├── ditto/                      # Ditto TalkingHead submodule
├── src/
│   ├── server.py               # FastAPI server + WebRTC signaling
│   ├── pipeline.py             # Pipecat conversation pipeline
│   ├── config.py               # Configuration management
│   └── services/
│       ├── ditto_realtime.py   # Real-time lip-sync service
│       ├── kokoro_tts.py       # Kokoro TTS service
│       ├── static_avatar.py    # Static avatar fallback
│       └── avatar_manager.py   # Avatar image management
├── web/                        # Frontend (served as static files)
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── setup_wsl.sh                # WSL2 one-command setup script
├── requirements.txt            # Base Python dependencies
├── requirements_wsl.txt        # WSL-specific extras (TRT, Cython, etc.)
├── .env.example                # Example environment config
├── .gitattributes              # Line ending rules
└── .gitignore
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for STT | **Required** |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | LLM model name | `mistral:7b` |
| `DITTO_CONFIG_PATH` | Ditto config pickle | `checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl` |
| `DITTO_CHECKPOINT_PATH` | Ditto model directory | `checkpoints/ditto_pytorch` |
| `DITTO_REALTIME` | Enable real-time Ditto | `true` |
| `HOST` | Server bind address | `0.0.0.0` |
| `PORT` | Server port | `8765` |

### Ditto Backend Options

Edit `src/config.py` to switch backends:

| Backend | Config File | Checkpoint Dir | Notes |
|---------|-------------|----------------|-------|
| **PyTorch** | `v0.4_hubert_cfg_pytorch.pkl` | `ditto_pytorch` | ✅ Recommended, works out of box |
| ONNX | `v0.4_hubert_cfg_hybrid_onnx.pkl` | `ditto_onnx` | Faster, requires ONNX Runtime |
| TensorRT | `v0.4_hubert_cfg_trt_online.pkl` | `ditto_trt` | Fastest, requires TensorRT setup |

---

## 🔧 Troubleshooting

### "No face detected" / Ditto Initialization Failed
- Use a clear, frontal face photo with good lighting
- Ensure the face is clearly visible and not obscured
- Try a different avatar image
- Check for eye_info.py errors in logs (indicates face detection issues)

### CUDA Out of Memory
- Close other GPU applications
- Reduce output resolution in config
- Try a smaller LLM model (`ollama pull phi3`)

### WebRTC Connection Fails
- Check firewall settings (port 8765 must be open)
- Ensure STUN server is accessible
- Try a different browser (Chrome/Edge recommended)

### Audio/Microphone Not Working from Other Devices
- **Must use HTTPS** (not HTTP) for microphone access from non-localhost
- Accept the self-signed certificate warning in browser
- Check browser permissions for microphone
- Ensure SSL certificates are properly generated

### Ditto Import Errors
```bash
# Ensure submodule is initialized
git submodule update --init --recursive

# Install missing dependencies
pip install mediapipe einops resampy
```

### Ollama Connection Refused
```bash
# Make sure Ollama is running
ollama serve

# Check if model is downloaded
ollama list
ollama pull mistral:7b
```

### Ollama Connection Refused (WSL)
WSL2 can't reach Windows `localhost`. Ollama must listen on all interfaces:
```powershell
# Windows PowerShell
[System.Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0:11434', 'User')
# Then restart Ollama
```
The setup script auto-detects the gateway IP and writes it to `.env`.

### Ditto "No module named 'filetype'" (or imageio, pyximport, etc.)
Missing WSL-specific dependencies. Re-run the setup script:
```bash
cd /mnt/f/Projects/LLM/TalkToMe
./setup_wsl.sh
```
Or install manually: `pip install -r requirements_wsl.txt`

### TensorRT Engines Won't Build
- Ensure `nvidia-smi` works inside WSL (`wsl nvidia-smi`)
- Check that ONNX models exist in `checkpoints/ditto_onnx/`
- Re-run `setup_wsl.sh` — it skips already-built engines

### "eye_info.py" IndexError
This error occurs when MediaPipe cannot detect facial landmarks. Try:
- Use a different avatar image with clearer face visibility
- Ensure proper lighting in the image
- Use images with natural eye appearance (no sunglasses)

---

## 🎨 Avatar Image Tips

For best results, use avatar images that:
- ✅ Have a **clear frontal view** of the face
- ✅ Have **good lighting** (no harsh shadows)
- ✅ Show the **full face** (not cropped)
- ✅ Have a **neutral or slight smile** expression
- ✅ Are **high resolution** (512x512 or larger)
- ❌ Avoid sunglasses or face coverings
- ❌ Avoid extreme angles or poses
- ❌ Avoid heavy makeup or face paint

---

## 📚 Tech Stack

- **[Pipecat](https://github.com/pipecat-ai/pipecat)** - Real-time conversation orchestration
- **[Ditto TalkingHead](https://github.com/digital-avatar/ditto)** - Audio-driven facial animation
- **[Ollama](https://ollama.ai)** - Local LLM inference
- **[Groq](https://groq.com)** - Fast STT (Whisper)
- **[Kokoro TTS](https://github.com/hexgrad/kokoro)** - High-quality text-to-speech
- **[FastAPI](https://fastapi.tiangolo.com)** - Web server
- **[WebRTC](https://webrtc.org)** - Real-time audio/video streaming

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Ditto TalkingHead](https://github.com/digital-avatar/ditto) for the amazing lip-sync technology
- [Pipecat](https://github.com/pipecat-ai/pipecat) for the conversation pipeline framework
- [Ollama](https://ollama.ai) for making local LLMs accessible

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
