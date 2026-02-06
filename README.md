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

## 📁 Project Structure

```
TalkToMe/
├── avatars/                    # Avatar images
├── certs/                      # SSL certificates (for HTTPS)
│   ├── cert.pem
│   └── key.pem
├── checkpoints/                # Model weights (gitignored)
│   ├── ditto_pytorch/          # PyTorch model files
│   └── ditto_cfg/              # Configuration files
├── ditto/                      # Ditto TalkingHead submodule
├── src/
│   ├── server.py               # FastAPI server + WebRTC signaling
│   ├── pipeline.py             # Pipecat conversation pipeline
│   ├── config.py               # Configuration management
│   └── services/
│       ├── ditto_realtime.py   # Real-time lip-sync service
│       ├── kokoro_tts.py       # Kokoro TTS service
│       └── avatar_manager.py   # Avatar image management
├── web/                        # Frontend (served as static files)
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── .env                        # Environment variables
├── .env.example                # Example environment config
└── requirements.txt            # Python dependencies
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for STT | **Required** |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | LLM model name | `mistral:7b` |
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
