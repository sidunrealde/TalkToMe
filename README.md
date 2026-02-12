# 🎭 TalkToMe — AI Avatar with Real-Time Lip Sync

A conversational AI avatar web application that brings portrait images to life with realistic lip-sync animation. Uses **Pipecat** for real-time conversation orchestration and **Ditto TalkingHead** for audio-driven facial animation, running on **WSL2** with **TensorRT** GPU acceleration.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green)
![TensorRT](https://img.shields.io/badge/TensorRT-10.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- 🎭 **Realistic Lip-Sync** — Animate any portrait photo with accurate mouth movements driven by audio
- 🎤 **Speech-to-Text** — Real-time transcription with Groq Whisper (cloud)
- 🤖 **LLM Conversations** — Powered by Ollama (local) or any OpenAI-compatible API
- 🔊 **Natural TTS** — High-quality voice synthesis with Kokoro TTS (local, no API needed)
- 🎯 **Voice Activity Detection** — Silero VAD for smart interruption handling
- 🌐 **WebRTC Streaming** — Real-time audio/video via browser, with automatic TURN relay for WSL2
- 🎨 **Multiple Voices** — Choose from various Kokoro voice options
- 🧠 **Customizable Personality** — YAML-based personality system for the AI avatar
- ⚡ **TensorRT Hybrid Mode** — 10 TRT engines + PyTorch warp for maximum inference speed

---

## 📐 Architecture Overview

```
┌───────────────────────────────────────────────────────────────────┐
│  Windows Host                                                     │
│  ┌──────────┐  ┌──────────┐                                      │
│  │  Chrome   │  │  Ollama  │  (LLM on localhost:11434)            │
│  │ WebRTC    │  │ mistral  │                                      │
│  └─────┬────┘  └────┬─────┘                                      │
│        │ HTTPS       │ HTTP                                       │
│        │ :8765       │ :11434                                     │
├────────┼─────────────┼────────────────────────────────────────────┤
│  WSL2 (Ubuntu)       │                                            │
│        │             │                                            │
│  ┌─────▼─────────────▼───────────────────────────┐               │
│  │  FastAPI + Pipecat Pipeline                    │               │
│  │                                                │               │
│  │  Browser Audio ──► Groq Whisper STT (cloud)    │               │
│  │                        │                       │               │
│  │                        ▼                       │               │
│  │                    Ollama LLM ◄────────────────┤               │
│  │                        │                       │               │
│  │                        ▼                       │               │
│  │                    Kokoro TTS (local)           │               │
│  │                        │                       │               │
│  │                        ▼                       │               │
│  │             Ditto TalkingHead (TensorRT)       │               │
│  │                        │                       │               │
│  │                        ▼                       │               │
│  │              WebRTC Video+Audio ──► Browser    │               │
│  └────────────────────────────────────────────────┘               │
│                                                                   │
│  ┌──────────┐  TURN relay for WebRTC (auto-managed)              │
│  │  coturn  │  127.0.0.1:3479 TCP                                │
│  └──────────┘                                                     │
└───────────────────────────────────────────────────────────────────┘
```

---

## 📋 Prerequisites

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GPU with CUDA (6 GB VRAM) | RTX 3090 / RTX 4090 |
| **VRAM** | 6 GB | 8+ GB |
| **RAM** | 16 GB | 32 GB |
| **OS** | Windows 11 22H2+ | Windows 11 23H2+ |

### Software (Windows Side)

| Software | Purpose | Install |
|----------|---------|---------|
| **WSL2 + Ubuntu** | Linux runtime for TensorRT | `wsl --install -d Ubuntu` |
| **NVIDIA GPU Driver** | GPU passthrough to WSL2 | [nvidia.com/drivers](https://www.nvidia.com/drivers) |
| **Ollama** | Local LLM inference | [ollama.ai](https://ollama.ai) |
| **Git** | Clone the repository | [git-scm.com](https://git-scm.com) |
| **Chrome / Edge** | Browser with WebRTC support | Already installed |

### API Keys

| Key | Purpose | Get it at |
|-----|---------|-----------|
| **Groq API Key** | Cloud speech-to-text (Whisper) | [console.groq.com](https://console.groq.com) |

> **Note:** Groq offers a generous free tier. No other paid API keys are required — the LLM (Ollama) and TTS (Kokoro) both run locally.

---

## 🚀 Full Setup Guide (Step by Step)

This guide walks you through setting up TalkToMe from a fresh Windows machine. The entire AI pipeline runs inside **WSL2** for TensorRT compatibility.

### Step 1: Install WSL2 with Ubuntu

Open **PowerShell as Administrator** and run:

```powershell
# Install WSL2 with Ubuntu (reboot may be required)
wsl --install -d Ubuntu

# After reboot, verify
wsl --version
```

During first launch Ubuntu will ask you to create a **username and password**. Remember these — you'll need the password for `sudo` commands later.

**Verify GPU passthrough works:**
```bash
# Open a WSL terminal
wsl

# Check that the GPU is visible
nvidia-smi
```

You should see your NVIDIA GPU listed with driver version. If not, update your Windows NVIDIA driver from [nvidia.com/drivers](https://www.nvidia.com/drivers) and then run `wsl --update`.

---

### Step 2: Install and Configure Ollama (Windows)

Ollama provides the local LLM that powers the avatar's conversations.

1. **Download and install** from [ollama.ai](https://ollama.ai)

2. **Configure Ollama to listen on all interfaces** (so WSL2 can reach it):

   ```powershell
   # PowerShell — set environment variable permanently
   [System.Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0:11434', 'User')
   ```

3. **Restart Ollama** — close the Ollama tray icon completely, then reopen Ollama

4. **Pull a model:**

   ```powershell
   ollama pull mistral:7b
   ```

> **Model choices:** `mistral:7b` is a good balance of quality and speed. For faster responses try `phi3` or `llama3.2:3b`. For better quality try `llama3.1:8b`.

---

### Step 3: Clone the Repository

```powershell
# Clone to your preferred location
cd F:\Projects\LLM
git clone https://github.com/yourusername/TalkToMe.git
cd TalkToMe

# Initialize the Ditto submodule
git submodule update --init --recursive
```

---

### Step 4: Download Model Checkpoints

The Ditto TalkingHead models (~2 GB) need to be downloaded from HuggingFace:

```powershell
# Install the HuggingFace CLI (in PowerShell or any Python env)
pip install huggingface_hub

# Download all checkpoint directories
python -c "from huggingface_hub import snapshot_download; snapshot_download('digital-avatar/ditto-talkinghead', local_dir='checkpoints', allow_patterns=['ditto_pytorch/*', 'ditto_onnx/*', 'ditto_cfg/*'])"
```

After downloading, verify your `checkpoints/` folder looks like this:

```
checkpoints/
├── ditto_cfg/                  # Config pickle files
│   ├── v0.4_hubert_cfg_pytorch.pkl
│   └── v0.4_hubert_cfg_trt_online.pkl
├── ditto_onnx/                 # ONNX models (→ converted to TRT engines by setup)
│   ├── appearance_extractor.onnx
│   ├── blaze_face.onnx
│   ├── decoder.onnx
│   ├── face_mesh.onnx
│   ├── hubert.onnx
│   ├── insightface_det.onnx
│   ├── landmark106.onnx
│   ├── landmark203.onnx
│   ├── lmdm_v0.4_hubert.onnx
│   ├── motion_extractor.onnx
│   ├── stitch_network.onnx
│   └── warp_network.onnx
└── ditto_pytorch/              # PyTorch model weights
    ├── aux_models/
    │   ├── 2d106det.onnx
    │   ├── det_10g.onnx
    │   ├── face_landmarker.task
    │   ├── hubert_streaming_fix_kv.onnx
    │   └── landmark203.onnx
    └── models/
        ├── appearance_extractor.pth
        ├── decoder.pth
        ├── lmdm_v0.4_hubert.pth
        ├── motion_extractor.pth
        ├── stitch_network.pth
        └── warp_network.pth
```

---

### Step 5: Get a Groq API Key

Groq provides fast cloud-based speech-to-text (Whisper).

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up or log in (free tier is sufficient)
3. Navigate to **API Keys** → **Create API Key**
4. **Copy the key** — you'll need it in Step 7

---

### Step 6: Run the WSL Setup Script

This is the main setup step. The `setup_wsl.sh` script automates everything inside WSL2:

```bash
# Open a WSL terminal
wsl

# Navigate to the repo (Windows drive is mounted under /mnt/)
cd /mnt/f/Projects/LLM/TalkToMe

# Make executable and run
chmod +x setup_wsl.sh
./setup_wsl.sh
```

> ⏱️ **First run takes 15–30 minutes** depending on internet speed and GPU (TensorRT engine builds are the slowest part).

#### What the script does automatically:

| Step | Action | Time |
|------|--------|------|
| 1 | **Preflight checks** — verifies WSL2 and NVIDIA GPU | Instant |
| 2 | **System packages** — installs `python3`, `ffmpeg`, `espeak-ng`, `coturn`, `build-essential` | 1–2 min |
| 3 | **Sync files** — copies project to WSL-native ext4 filesystem (`~/TalkToMe`) | 30 sec |
| 4 | **Checkpoints** — copies ONNX, PyTorch, and config files | 1–2 min |
| 5 | **Python venv** — creates virtual environment at `~/TalkToMe/venv` | 30 sec |
| 6 | **Python deps** — installs PyTorch (CUDA 12.4), TensorRT, Pipecat, Kokoro, etc. | 5–10 min |
| 7 | **Cython** — pre-compiles the Ditto blend module | 10 sec |
| 8 | **TensorRT engines** — builds 10 TRT engines from ONNX models | 5–15 min |
| 9 | **Hybrid config** — generates the TRT hybrid config pickle | Instant |
| 10 | **Environment** — auto-detects Windows gateway IP, creates `.env` | Instant |
| 11 | **TURN relay** — configures coturn for WebRTC in WSL2 | Instant |
| 12 | **Run script** — creates `~/TalkToMe/run.sh` with auto-TURN management | Instant |
| 13 | **Verification** — tests all Python imports and CUDA availability | 5 sec |

> **Re-running is safe.** The script reuses the existing venv, skips already-built TRT engines, and preserves your Groq API key in `.env`.

#### Why WSL-native filesystem?

The script copies files from `/mnt/f/...` (Windows NTFS, accessed via 9P — very slow for Python) to `~/TalkToMe/` on WSL's native ext4 filesystem, which is **~10x faster** for imports and I/O. Always run the server from `~/TalkToMe`, not from `/mnt/`.

---

### Step 7: Set Your Groq API Key

After the setup script finishes, edit the generated `.env` file:

```bash
# Still inside WSL
nano ~/TalkToMe/.env
```

Find the line:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace `your_groq_api_key_here` with your actual Groq API key from Step 5.

Save and exit: `Ctrl+O` → `Enter` → `Ctrl+X`

---

### Step 8: Prepare an Avatar Image

You can either:
- **Upload through the web UI** (easiest) — just click the upload area after opening the app
- **Pre-place an image** in the `avatars/` directory

**Tips for best lip-sync results:**
- ✅ Clear frontal view of the face
- ✅ Good, even lighting (no harsh shadows)
- ✅ Neutral or slight smile expression
- ✅ High resolution (512×512 pixels or larger)
- ❌ Avoid sunglasses or face coverings
- ❌ Avoid extreme angles or side profiles
- ❌ Avoid heavy makeup or face paint

---

### Step 9: Start the Server

```bash
cd ~/TalkToMe
./run.sh
```

You should see output like:

```
[TURN] Starting coturn relay on 127.0.0.1...
[TURN] ✓ coturn running (pid 502)

============================================================
🔒 HTTPS Server running at:
   Local:   https://localhost:8765
   Network: https://192.168.0.176:8765
============================================================
```

The server automatically:
1. **Starts the coturn TURN relay** for WebRTC (WSL2 requires this)
2. **Detects WSL2 networking** and patches ICE candidates
3. **Loads the Ditto TensorRT pipeline** on first browser connection

---

### Step 10: Open in Browser

1. Open **Chrome** (or Edge) on Windows
2. Navigate to **https://localhost:8765**
3. **Accept the certificate warning** — click "Advanced" → "Proceed to localhost (unsafe)"
   > This is expected — the server uses a self-signed SSL certificate for HTTPS, which is required for WebRTC microphone access.
4. Upload an avatar image (or use the default if one is pre-loaded)
5. Select a voice from the dropdown
6. Click **"Connect"**
7. **Start talking!** Or type a message in the text box.

> **First connection takes ~15 seconds** while Ditto loads TensorRT engines into GPU memory. Subsequent connections are instant.

---

## 🏃 Daily Usage (After Initial Setup)

Once everything is set up, your daily workflow is just three steps:

```bash
# 1. Make sure Ollama is running on Windows (check system tray icon)

# 2. Start the server
wsl -e bash -c "cd ~/TalkToMe && ./run.sh"

# 3. Open https://localhost:8765 in Chrome
```

### run.sh Options

```bash
# Foreground mode — see all logs, Ctrl+C to stop
./run.sh

# Background mode — runs in tmux, survives terminal close
./run.sh --background

# View background logs
tail -f /tmp/talktome.log

# Stop background server
tmux kill-session -t talktome
```

### Quick-Launch from Windows PowerShell

You don't need to enter WSL interactively:

```powershell
# Start server (blocks terminal, shows logs)
wsl -e bash -c "cd ~/TalkToMe && ./run.sh"

# Or run in background
wsl -e bash -c "cd ~/TalkToMe && ./run.sh --background"
```

---

## 🌐 LAN Access (Access from Other Devices)

By default the server is accessible only from the Windows host via `localhost`. To access from phones, tablets, or other computers on your local network:

```powershell
# Run from Windows PowerShell as Administrator
cd F:\Projects\LLM\TalkToMe
.\setup_lan.ps1
```

This script performs:
1. **Enables WSL2 mirrored networking** — gives WSL the same IP as the Windows host
2. **Generates SSL certificates** with your LAN IP in the Subject Alternative Name
3. **Copies certs** into the WSL project directory
4. **Adds Windows Firewall rules** for TCP port 8765 and UDP (WebRTC media)

After setup, **restart WSL** (`wsl --shutdown` from PowerShell, then start it again) and run the server. Access from any device on your network:

```
https://YOUR_LAN_IP:8765
```

> **To undo LAN setup:** `.\setup_lan.ps1 -Teardown`

---

## 🧠 Customizing the AI Personality

The avatar's personality, backstory, and behavioral rules are defined in `personality.yaml` at the project root:

```yaml
name: "Nova"
tagline: "A curious and warm AI companion who loves good conversation."

personality:
  - friendly and approachable
  - gently witty with a dry sense of humour
  - concise – prefers short, punchy replies

backstory: |
  You were brought to life inside a small indie lab by a solo developer
  who wanted an AI companion that actually felt like a person...

instructions:
  - Keep responses concise (1-3 sentences when possible).
  - If you don't know something, say so honestly.
  - Match the user's energy – be playful when they're playful.

voice_style: "warm, conversational, moderate pace"
```

Edit this file to change the avatar's name, personality traits, backstory, and behavior. Changes take effect on the next browser connection — no server restart needed.

After editing, sync to WSL:
```powershell
Copy-Item personality.yaml \\wsl$\Ubuntu\home\YOUR_USERNAME\TalkToMe\personality.yaml
```

---

## ⚡ Ditto Backend Modes

The setup script auto-selects **TRT Hybrid** mode for maximum performance:

| Mode | Config Pickle | Checkpoint Dir | Speed | Notes |
|------|---------------|----------------|-------|-------|
| **TRT Hybrid** ⚡ | `v0.4_hubert_cfg_trt_hybrid_online.pkl` | `ditto_trt` | **Fastest** | 10 TRT engines + PyTorch warp_network |
| PyTorch | `v0.4_hubert_cfg_pytorch.pkl` | `ditto_pytorch` | Slower | No TRT needed, works everywhere |
| Full TRT | `v0.4_hubert_cfg_trt_online.pkl` | `ditto_trt` | Fastest | Requires GridSample3D TRT plugin |

To switch modes, edit `~/TalkToMe/.env`:

```env
# TRT Hybrid (recommended — default after setup)
DITTO_CONFIG_PATH=checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_hybrid_online.pkl
DITTO_CHECKPOINT_PATH=checkpoints/ditto_trt

# PyTorch fallback (if TRT engines won't build)
# DITTO_CONFIG_PATH=checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl
# DITTO_CHECKPOINT_PATH=checkpoints/ditto_pytorch
```

> **Why hybrid?** TensorRT's `GridSample3D` operator doesn't support the warp_network's 5D grid. The hybrid approach runs 10 models as TRT engines and only `warp_network` in PyTorch, giving near-full TRT speed without needing a custom plugin.

---

## ⚙️ Configuration Reference

### Environment Variables (`~/TalkToMe/.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for cloud STT | **Required** |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` (auto-detected for WSL) |
| `OLLAMA_MODEL` | LLM model name | `mistral:7b` |
| `DITTO_CONFIG_PATH` | Ditto config pickle path | (set by setup script) |
| `DITTO_CHECKPOINT_PATH` | Ditto model directory | (set by setup script) |
| `DITTO_REALTIME` | Enable real-time lip-sync | `true` |
| `HOST` | Server bind address | `0.0.0.0` |
| `PORT` | Server port | `8765` |
| `PERSONALITY_PATH` | Path to personality YAML | `personality.yaml` |

### TURN Relay (Automatic — No Setup Needed)

The server auto-manages a **coturn TURN relay** for WebRTC connectivity in WSL2. The config lives in `turn/turnserver.conf` and is generated by `setup_wsl.sh`. You should never need to touch this.

| Setting | Value | Why |
|---------|-------|-----|
| Listen IP | `127.0.0.1` | Shared loopback between Windows and WSL2 |
| Port | `3479` | Avoids conflict with system coturn on 3478 |
| Transport | TCP | WSL2 mirrored mode breaks direct UDP between Windows ↔ WSL |
| Auth | `talktome` / `talktome123` | Long-term credential mechanism |

> **Why TURN?** In WSL2 mirrored networking mode, UDP packets between Windows and WSL don't work reliably on the LAN IP. The TURN relay routes all WebRTC media through TCP on the shared `127.0.0.1` loopback, which works perfectly.

---

## 📁 Project Structure

```
TalkToMe/
├── src/                            # Python source code
│   ├── server.py                   #   FastAPI server, WebRTC signaling, ICE/TURN auto-config
│   ├── pipeline.py                 #   Pipecat conversation pipeline orchestration
│   ├── config.py                   #   Configuration loader (.env + personality.yaml)
│   └── services/
│       ├── ditto_realtime.py       #   Real-time Ditto lip-sync (TRT/PyTorch hybrid)
│       ├── kokoro_tts.py           #   Kokoro TTS service (local ONNX inference)
│       ├── avatar_manager.py       #   Avatar image upload and management
│       ├── static_avatar.py        #   Static avatar fallback (no lip-sync)
│       ├── edge_tts_service.py     #   Edge TTS alternative
│       └── ditto_video.py          #   Ditto video generation utilities
├── web/                            # Frontend (served as static files)
│   ├── index.html                  #   Main page
│   ├── app.js                      #   WebRTC client, UI logic, ICE config fetch
│   └── styles.css                  #   Styling
├── ditto/                          # Ditto TalkingHead git submodule
├── checkpoints/                    # Model weights (~2 GB, gitignored)
│   ├── ditto_cfg/                  #   Config pickles (PyTorch, TRT, hybrid)
│   ├── ditto_onnx/                 #   ONNX models (source for TRT conversion)
│   ├── ditto_pytorch/              #   PyTorch model weights (.pth)
│   └── ditto_trt/                  #   TensorRT engines (.engine, built by setup)
├── avatars/                        # Avatar images (uploaded via web UI)
├── certs/                          # SSL certificates (for HTTPS / LAN access)
├── turn/                           # TURN relay configuration (auto-generated)
│   └── turnserver.conf             #   coturn config file
├── personality.yaml                # AI avatar personality definition
├── setup_wsl.sh                    # ★ WSL2 one-command setup script (main setup)
├── setup_lan.ps1                   # LAN access setup (Windows, run as admin)
├── requirements.txt                # Base Python dependencies
├── requirements_wsl.txt            # WSL-specific extras (TRT, Cython, etc.)
├── .env.example                    # Example environment configuration
└── .gitignore
```

**Generated files (inside WSL at `~/TalkToMe/`):**
- `run.sh` — Server launch script with coturn lifecycle management
- `.env` — Environment configuration with auto-detected Ollama URL
- `venv/` — Python virtual environment
- `checkpoints/ditto_trt/*.engine` — Built TensorRT engines
- `checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_hybrid_online.pkl` — Hybrid config

---

## 🔧 Troubleshooting

### Setup Issues

#### `nvidia-smi` not found inside WSL
Your Windows NVIDIA driver may be outdated or WSL needs an update:
```powershell
# From PowerShell
wsl --update
# Then update your NVIDIA driver from nvidia.com/drivers
```

#### Setup script fails during "Installing Python dependencies"
Usually a transient network issue or pip conflict. Just re-run — the script is idempotent:
```bash
cd /mnt/f/Projects/LLM/TalkToMe
./setup_wsl.sh
```

#### TensorRT engines won't build
- Verify `nvidia-smi` works inside WSL
- Ensure ONNX models exist: `ls ~/TalkToMe/checkpoints/ditto_onnx/*.onnx`
- Delete engines to force rebuild: `rm ~/TalkToMe/checkpoints/ditto_trt/*.engine`
- Re-run `./setup_wsl.sh`
- **Fallback:** If TRT consistently fails, switch to PyTorch mode — edit `~/TalkToMe/.env` (see [Ditto Backend Modes](#-ditto-backend-modes))

#### "Step 8" takes forever
TensorRT engine building is GPU-bound and can take 5–15 minutes on first run. This is a one-time cost — engines are cached and reused on subsequent runs.

---

### Connection Issues

#### Groq STT error: "Invalid API Key"
Your Groq API key is missing or incorrect:
```bash
nano ~/TalkToMe/.env
# Fix the GROQ_API_KEY=... line to your actual key
```

#### Ollama error: "All connection attempts failed"
WSL2 can't reach `localhost` on the Windows host. Ollama must listen on all interfaces:

1. **Set the environment variable (Windows PowerShell):**
   ```powershell
   [System.Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0:11434', 'User')
   ```

2. **Restart Ollama** — close the tray icon completely, then reopen

3. **Verify from WSL:**
   ```bash
   # Find the Windows gateway IP
   ip route show default | awk '{print $3}'
   # Test connectivity (replace with your IP)
   curl http://172.x.x.x:11434/
   ```

The setup script auto-detects this gateway IP and writes the correct `OLLAMA_BASE_URL` to `.env`.

#### WebRTC not connecting (no video/audio after clicking Connect)
The coturn TURN relay should start automatically with the server. Check:
```bash
# Is coturn running?
ss -tlnH 'sport = :3479'

# Check coturn logs
cat /tmp/coturn.log

# Restart the server (coturn auto-starts with it)
cd ~/TalkToMe && ./run.sh
```

In the server logs, you should see:
```
[TURN] ✓ coturn started (pid XXX)
```

And when the browser connects, the SDP offer should show `type=relay` candidates.

#### Browser shows certificate error
This is expected with self-signed certificates:
1. Click **"Advanced"**
2. Click **"Proceed to localhost (unsafe)"**
3. The page will load normally

#### Browser shows old version / stale UI
Clear the browser cache: `Ctrl+Shift+Delete` → check "Cached images and files" → Clear

---

### Runtime Issues

#### "No face detected" / Ditto initialization failed
- Use a clear **frontal face photo** with good lighting
- The face must be clearly visible (no sunglasses, masks, or extreme angles)
- Try a different avatar image
- Check logs for `eye_info.py` errors (indicates face landmark detection failure)

#### CUDA out of memory
- Close other GPU applications (games, other AI tools)
- Use a smaller LLM: `ollama pull phi3` then edit `.env`: `OLLAMA_MODEL=phi3`
- Reduce `sampling_timesteps` in the Ditto config

#### Audio / microphone not working
- **HTTPS is required** for browser microphone access (the server uses HTTPS by default)
- Check browser permissions: click the **lock icon** in the address bar → Site settings → Microphone → **Allow**
- Try a fresh browser tab (some browsers cache permission denials)

#### Ditto import errors
```bash
# Re-initialize the submodule
cd /mnt/f/Projects/LLM/TalkToMe
git submodule update --init --recursive

# Re-run setup to sync fresh files
./setup_wsl.sh
```

---

### WSL-Specific Issues

#### Everything is very slow
Make sure you're running from the WSL-native filesystem, not `/mnt/`:
```bash
# ✅ Correct (fast)
cd ~/TalkToMe && ./run.sh

# ❌ Wrong (slow — accesses Windows filesystem via 9P)
cd /mnt/f/Projects/LLM/TalkToMe && python -m src.server
```

#### WSL networking breaks after sleep/hibernate
WSL2 networking can break after Windows sleep. Fix:
```powershell
# From PowerShell
wsl --shutdown
# Then start WSL again
wsl
```

#### Port 8765 already in use
```bash
# Find what's using the port
ss -tlnp 'sport = :8765'

# Kill all TalkToMe processes
pkill -f 'python.*src.server'
pkill -f turnserver
```

#### How to fully reset the WSL setup
```bash
# Remove the entire WSL project (keeps Windows files intact)
rm -rf ~/TalkToMe

# Re-run setup from the Windows mount
cd /mnt/f/Projects/LLM/TalkToMe
./setup_wsl.sh
```

---

## 🔄 Updating

When you pull new code from git:

```bash
# 1. Pull latest changes (from Windows path or WSL /mnt/ path)
cd /mnt/f/Projects/LLM/TalkToMe
git pull
git submodule update --recursive

# 2. Re-run setup (syncs changed files, installs new deps, preserves .env)
./setup_wsl.sh

# 3. Restart server
cd ~/TalkToMe
./run.sh
```

---

## 📚 Tech Stack

| Component | Technology | Runs On |
|-----------|-----------|---------|
| **Conversation Orchestration** | [Pipecat](https://github.com/pipecat-ai/pipecat) 0.0.101 | WSL2 |
| **Lip-Sync Animation** | [Ditto TalkingHead](https://github.com/digital-avatar/ditto) | WSL2 (TensorRT + PyTorch hybrid) |
| **LLM** | [Ollama](https://ollama.ai) (Mistral, Llama, Phi, etc.) | Windows |
| **Speech-to-Text** | [Groq](https://groq.com) Whisper | Cloud (free tier) |
| **Text-to-Speech** | [Kokoro TTS](https://github.com/hexgrad/kokoro) (ONNX) | WSL2 (fully local) |
| **Web Server** | [FastAPI](https://fastapi.tiangolo.com) + Uvicorn | WSL2 |
| **Real-Time Streaming** | [WebRTC](https://webrtc.org) via SmallWebRTC + aiortc | Browser ↔ WSL2 |
| **TURN Relay** | [coturn](https://github.com/coturn/coturn) | WSL2 (auto-managed) |
| **GPU Acceleration** | NVIDIA TensorRT 10.x + CUDA 12.x | WSL2 (GPU passthrough) |
| **Voice Activity Detection** | [Silero VAD](https://github.com/snakers4/silero-vad) | WSL2 |
| **Frontend** | Vanilla HTML / CSS / JavaScript | Browser |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Ditto TalkingHead](https://github.com/digital-avatar/ditto) — Audio-driven facial animation
- [Pipecat](https://github.com/pipecat-ai/pipecat) — Real-time conversation pipeline framework
- [Ollama](https://ollama.ai) — Local LLM inference made easy
- [Kokoro TTS](https://github.com/hexgrad/kokoro) — High-quality local text-to-speech
- [coturn](https://github.com/coturn/coturn) — Open-source TURN/STUN server

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
