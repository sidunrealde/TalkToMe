# AI Avatar System: Pipecat + Ditto TalkingHead

## Project Overview

Build a conversational AI avatar web application that:
- Animates a user-provided portrait image with realistic lip-sync
- Uses Pipecat for real-time conversation pipeline (STT, LLM, TTS, VAD, interruption)
- Uses Ditto TalkingHead for audio-driven facial animation
- Supports customizable voices
- Runs on RTX 3090 with CUDA 12.1

## Repository Structure

```
f:\Projects\LLM\TalkToMe\
├── .gitignore
├── .gitmodules
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
│
├── ditto/                      # Git submodule: Ditto TalkingHead
│
├── checkpoints/                # Model weights (gitignored)
│   └── ditto_pytorch/
│       ├── appearance_feature_extractor.safetensors
│       ├── motion_extractor.safetensors
│       ├── stitching_retargeting_module.safetensors
│       ├── warping_module.safetensors
│       └── hubert/
│           └── hubert_base_ls960.pt
│
├── src/
│   ├── __init__.py
│   ├── server.py               # FastAPI + WebRTC signaling
│   ├── pipeline.py             # Pipecat conversation pipeline
│   ├── config.py               # Configuration management
│   │
│   └── services/
│       ├── __init__.py
│       ├── ditto_video.py      # DittoVideoService (Pipecat integration)
│       └── avatar_manager.py   # Avatar image/state management
│
├── web/
│   ├── index.html              # Main UI
│   ├── app.js                  # Frontend logic
│   └── styles.css              # Styling
│
└── tests/
    ├── __init__.py
    ├── test_ditto_service.py
    └── test_pipeline.py
```

---

## Phase 1: Repository & Environment Setup

### 1.1 Create Repository

```powershell
# Create project directory
New-Item -ItemType Directory -Path "f:\Projects\LLM\TalkToMe" -Force
Set-Location "f:\Projects\LLM\TalkToMe"

# Initialize git
git init
```

### 1.2 Create .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
.eggs/

# Virtual environments
.venv/
venv/
ENV/

# Environment variables
.env

# Model checkpoints (large files)
checkpoints/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Test coverage
.coverage
htmlcov/
```

### 1.3 Create Conda Environment

```powershell
# Create environment with Python 3.10 (required by Ditto)
conda create -n TalkToMe python=3.10 -y
conda activate TalkToMe

# Install PyTorch with CUDA 12.1 support
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

### 1.4 Create requirements.txt

```txt
# Core Pipecat
pipecat-ai[groq,silero]>=0.0.102

# WebRTC transport
smallwebrtc>=0.1.0

# Web server
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6

# Audio processing
librosa>=0.10.0
soundfile>=0.12.0
numpy>=1.24.0

# Ditto dependencies
onnxruntime-gpu>=1.16.0
safetensors>=0.4.0
opencv-python>=4.8.0
pillow>=10.0.0
scipy>=1.11.0

# HTTP client for Ollama
httpx>=0.26.0

# Environment management
python-dotenv>=1.0.0

# Development
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

### 1.5 Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## Phase 2: Ditto TalkingHead Setup

### 2.1 Add as Git Submodule

```powershell
git submodule add https://github.com/DigitalAvatar-Developers/ditto-talkinghead.git ditto
git submodule update --init --recursive
```

### 2.2 Download Checkpoints

```powershell
# Install huggingface-cli if needed
pip install huggingface_hub

# Download PyTorch checkpoints only (avoids TensorRT)
huggingface-cli download digital-avatar/ditto-talkinghead --local-dir checkpoints --include "ditto_pytorch/*" "ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
```

### 2.3 Verify Ditto Installation

```python
# test_ditto_setup.py
import sys
sys.path.insert(0, './ditto')

from ditto.sdk.stream_sdk import DittoStreamSDK, DittoStreamSDKConfig

config = DittoStreamSDKConfig(
    config_path="checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
    checkpoint_path="checkpoints/ditto_pytorch",
    device_id=0
)

sdk = DittoStreamSDK(config)
print("Ditto SDK initialized successfully!")
```

---

## Phase 3: Pipecat Pipeline Setup

### 3.1 Verify Pipecat Installation

```python
# test_pipecat_setup.py
from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.groq import GroqSTTService, GroqTTSService
from pipecat.vad.silero import SileroVADAnalyzer

print("Pipecat imported successfully!")
print("GroqSTTService:", GroqSTTService)
print("GroqTTSService:", GroqTTSService)
print("SileroVADAnalyzer:", SileroVADAnalyzer)
```

---

## Phase 4: Core Implementation

### 4.1 Configuration (src/config.py)

```python
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # Groq API
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    
    # Ollama LLM
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "mistral:7b")
    
    # TTS Settings
    tts_model: str = "canopylabs/orpheus-v1-english"
    tts_voice: str = "autumn"  # autumn, breeze, ember, juniper
    
    # Ditto Settings
    ditto_config_path: str = "checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
    ditto_checkpoint_path: str = "checkpoints/ditto_pytorch"
    ditto_device_id: int = 0
    ditto_fps: int = 25
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8765
    
    # System Prompt
    system_prompt: str = """You are a friendly AI assistant with a visual avatar. 
    Keep responses concise and conversational. You can see the user and they can see you."""

config = Config()
```

### 4.2 Ditto Video Service (src/services/ditto_video.py)

```python
import asyncio
import sys
import numpy as np
from typing import Optional
from PIL import Image
import cv2

from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    OutputImageRawFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# Add ditto to path
sys.path.insert(0, './ditto')
from ditto.sdk.stream_sdk import DittoStreamSDK, DittoStreamSDKConfig


class DittoVideoService(FrameProcessor):
    """Pipecat processor that generates talking head video from audio."""
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device_id: int = 0,
        fps: int = 25,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._device_id = device_id
        self._fps = fps
        self._frame_duration = 1.0 / fps
        
        self._sdk: Optional[DittoStreamSDK] = None
        self._source_image: Optional[np.ndarray] = None
        self._is_speaking = False
        self._audio_buffer = []
        self._lock = asyncio.Lock()
        
    async def start(self, frame: Frame):
        """Initialize Ditto SDK."""
        await super().start(frame)
        
        sdk_config = DittoStreamSDKConfig(
            config_path=self._config_path,
            checkpoint_path=self._checkpoint_path,
            device_id=self._device_id
        )
        self._sdk = DittoStreamSDK(sdk_config)
        
    async def stop(self, frame: Frame):
        """Cleanup Ditto SDK."""
        self._sdk = None
        await super().stop(frame)
        
    def set_source_image(self, image: np.ndarray):
        """Set the source portrait image for animation."""
        self._source_image = image
        if self._sdk:
            self._sdk.set_source_image(image)
            
    def set_source_image_from_path(self, path: str):
        """Load and set source image from file path."""
        image = cv2.imread(path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.set_source_image(image)
            
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, StartInterruptionFrame):
            # User interrupted - stop generating frames
            self._is_speaking = False
            self._audio_buffer.clear()
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, StopInterruptionFrame):
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, AudioRawFrame):
            # Collect audio and generate video frames
            if self._sdk and self._source_image is not None:
                await self._process_audio(frame)
            await self.push_frame(frame, direction)
            
        else:
            await self.push_frame(frame, direction)
            
    async def _process_audio(self, audio_frame: AudioRawFrame):
        """Process audio chunk and generate corresponding video frames."""
        async with self._lock:
            # Convert audio bytes to numpy array
            audio_data = np.frombuffer(audio_frame.audio, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Add to buffer
            self._audio_buffer.extend(audio_float.tolist())
            
            # Process when we have enough audio for one video frame
            # At 16kHz audio and 25fps video, need 640 samples per frame
            samples_per_frame = int(audio_frame.sample_rate / self._fps)
            
            while len(self._audio_buffer) >= samples_per_frame:
                # Extract audio chunk
                chunk = np.array(self._audio_buffer[:samples_per_frame], dtype=np.float32)
                self._audio_buffer = self._audio_buffer[samples_per_frame:]
                
                # Generate video frame using Ditto
                try:
                    video_frame = await asyncio.to_thread(
                        self._sdk.run_chunk, chunk
                    )
                    
                    if video_frame is not None:
                        # Convert to OutputImageRawFrame
                        rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                        height, width = rgb_frame.shape[:2]
                        
                        image_frame = OutputImageRawFrame(
                            image=rgb_frame.tobytes(),
                            size=(width, height),
                            format="RGB"
                        )
                        await self.push_frame(image_frame)
                        
                except Exception as e:
                    print(f"Ditto frame generation error: {e}")
```

### 4.3 Avatar Manager (src/services/avatar_manager.py)

```python
import os
import base64
from typing import Optional
import numpy as np
import cv2
from PIL import Image
import io


class AvatarManager:
    """Manages avatar images and state."""
    
    def __init__(self, avatars_dir: str = "avatars"):
        self._avatars_dir = avatars_dir
        self._current_avatar: Optional[np.ndarray] = None
        self._current_voice: str = "autumn"
        
        os.makedirs(avatars_dir, exist_ok=True)
        
    @property
    def current_avatar(self) -> Optional[np.ndarray]:
        return self._current_avatar
    
    @property
    def current_voice(self) -> str:
        return self._current_voice
    
    def set_voice(self, voice: str):
        """Set the TTS voice."""
        valid_voices = ["autumn", "breeze", "ember", "juniper"]
        if voice in valid_voices:
            self._current_voice = voice
            
    def load_avatar_from_path(self, path: str) -> bool:
        """Load avatar image from file path."""
        try:
            image = cv2.imread(path)
            if image is not None:
                self._current_avatar = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return True
        except Exception as e:
            print(f"Error loading avatar: {e}")
        return False
    
    def load_avatar_from_base64(self, data: str) -> bool:
        """Load avatar image from base64 string."""
        try:
            # Remove data URL prefix if present
            if "," in data:
                data = data.split(",")[1]
                
            image_bytes = base64.b64decode(data)
            image = Image.open(io.BytesIO(image_bytes))
            self._current_avatar = np.array(image.convert("RGB"))
            return True
        except Exception as e:
            print(f"Error loading avatar from base64: {e}")
        return False
    
    def get_avatar_base64(self) -> Optional[str]:
        """Get current avatar as base64 string."""
        if self._current_avatar is None:
            return None
            
        try:
            image = Image.fromarray(self._current_avatar)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception:
            return None
```

### 4.4 Pipeline (src/pipeline.py)

```python
import httpx
import asyncio
from typing import Optional

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.groq import GroqSTTService, GroqTTSService
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    Frame, 
    LLMMessagesFrame, 
    TextFrame, 
    LLMFullResponseEndFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)

from .config import config
from .services.ditto_video import DittoVideoService
from .services.avatar_manager import AvatarManager


class OllamaLLMService(FrameProcessor):
    """Simple Ollama LLM integration."""
    
    def __init__(self, base_url: str, model: str, **kwargs):
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(timeout=60.0)
        self._context = []
        
    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMMessagesFrame):
            self._context = frame.messages
            await self._generate_response()
        else:
            await self.push_frame(frame, direction)
            
    async def _generate_response(self):
        try:
            response = await self._client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": self._context,
                    "stream": True
                }
            )
            
            async for line in response.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        text = data["message"]["content"]
                        if text:
                            await self.push_frame(TextFrame(text=text))
                            
            await self.push_frame(LLMFullResponseEndFrame())
            
        except Exception as e:
            print(f"Ollama error: {e}")
            await self.push_frame(TextFrame(text="I'm having trouble responding right now."))
            await self.push_frame(LLMFullResponseEndFrame())


async def create_pipeline(
    transport,
    avatar_manager: AvatarManager
) -> PipelineTask:
    """Create the full conversation pipeline."""
    
    # VAD for detecting user speech
    vad = SileroVADAnalyzer()
    
    # Speech-to-text
    stt = GroqSTTService(
        api_key=config.groq_api_key,
        model="whisper-large-v3"
    )
    
    # LLM
    llm = OllamaLLMService(
        base_url=config.ollama_base_url,
        model=config.ollama_model
    )
    
    # Text-to-speech
    tts = GroqTTSService(
        api_key=config.groq_api_key,
        model=config.tts_model,
        voice=avatar_manager.current_voice
    )
    
    # Ditto video generation
    ditto = DittoVideoService(
        config_path=config.ditto_config_path,
        checkpoint_path=config.ditto_checkpoint_path,
        device_id=config.ditto_device_id,
        fps=config.ditto_fps
    )
    
    # Set avatar image if available
    if avatar_manager.current_avatar is not None:
        ditto.set_source_image(avatar_manager.current_avatar)
    
    # Context aggregator for conversation history
    context = OpenAILLMContext(
        messages=[{"role": "system", "content": config.system_prompt}]
    )
    context_aggregator = llm.create_context_aggregator(context)
    
    # Build pipeline
    pipeline = Pipeline([
        transport.input(),          # Audio from user
        vad,                        # Voice activity detection
        stt,                        # Speech to text
        context_aggregator.user(),  # Add user message to context
        llm,                        # Generate response
        tts,                        # Text to speech
        ditto,                      # Generate video frames
        transport.output()          # Send audio + video to user
    ])
    
    # Create task with interruption support
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True
        )
    )
    
    return task
```

### 4.5 Server (src/server.py)

```python
import asyncio
import os
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from smallwebrtc import SmallWebRTCTransport, SmallWebRTCConnection

from .config import config
from .pipeline import create_pipeline
from .services.avatar_manager import AvatarManager

app = FastAPI(title="AI Avatar")
avatar_manager = AvatarManager()

# Serve static files
app.mount("/static", StaticFiles(directory="web"), name="static")


@app.get("/")
async def index():
    return FileResponse("web/index.html")


@app.post("/api/avatar")
async def upload_avatar(file: UploadFile = File(...)):
    """Upload a new avatar image."""
    contents = await file.read()
    import base64
    b64 = base64.b64encode(contents).decode()
    
    if avatar_manager.load_avatar_from_base64(b64):
        return {"status": "success"}
    return {"status": "error", "message": "Failed to load avatar"}


@app.post("/api/voice")
async def set_voice(voice: str = Form(...)):
    """Set the TTS voice."""
    avatar_manager.set_voice(voice)
    return {"status": "success", "voice": avatar_manager.current_voice}


@app.get("/api/voices")
async def get_voices():
    """Get available TTS voices."""
    return {
        "voices": ["autumn", "breeze", "ember", "juniper"],
        "current": avatar_manager.current_voice
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebRTC signaling endpoint."""
    await websocket.accept()
    
    connection = SmallWebRTCConnection()
    transport = SmallWebRTCTransport(connection)
    
    # Create pipeline
    task = await create_pipeline(transport, avatar_manager)
    
    # Handle signaling
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "offer":
                answer = await connection.handle_offer(data["sdp"])
                await websocket.send_json({"type": "answer", "sdp": answer})
                
            elif data.get("type") == "ice-candidate":
                await connection.add_ice_candidate(data["candidate"])
                
            elif data.get("type") == "start":
                # Start the pipeline
                asyncio.create_task(task.run())
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await task.cancel()


def main():
    uvicorn.run(
        "src.server:app",
        host=config.host,
        port=config.port,
        reload=True
    )


if __name__ == "__main__":
    main()
```

---

## Phase 5: Web Frontend

### 5.1 HTML (web/index.html)

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Avatar</title>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>AI Avatar</h1>
        </header>
        
        <main>
            <!-- Avatar Display -->
            <div class="avatar-container">
                <video id="avatar-video" autoplay playsinline></video>
                <img id="avatar-image" src="" alt="Avatar" style="display: none;">
                <div id="avatar-placeholder">Upload an avatar to begin</div>
            </div>
            
            <!-- Controls -->
            <div class="controls">
                <!-- Avatar Upload -->
                <div class="control-group">
                    <label for="avatar-upload">Avatar Image:</label>
                    <input type="file" id="avatar-upload" accept="image/*">
                </div>
                
                <!-- Voice Selection -->
                <div class="control-group">
                    <label for="voice-select">Voice:</label>
                    <select id="voice-select">
                        <option value="autumn">Autumn</option>
                        <option value="breeze">Breeze</option>
                        <option value="ember">Ember</option>
                        <option value="juniper">Juniper</option>
                    </select>
                </div>
                
                <!-- Connection -->
                <div class="control-group">
                    <button id="connect-btn" class="primary">Connect</button>
                    <button id="disconnect-btn" disabled>Disconnect</button>
                </div>
            </div>
            
            <!-- Status -->
            <div class="status">
                <span id="connection-status">Disconnected</span>
                <span id="speaking-indicator" class="hidden">🎤 Listening...</span>
            </div>
            
            <!-- Transcript -->
            <div class="transcript" id="transcript"></div>
        </main>
    </div>
    
    <script src="/static/app.js"></script>
</body>
</html>
```

### 5.2 JavaScript (web/app.js)

```javascript
class AIAvatarClient {
    constructor() {
        this.ws = null;
        this.pc = null;
        this.localStream = null;
        this.isConnected = false;
        
        this.elements = {
            video: document.getElementById('avatar-video'),
            image: document.getElementById('avatar-image'),
            placeholder: document.getElementById('avatar-placeholder'),
            avatarUpload: document.getElementById('avatar-upload'),
            voiceSelect: document.getElementById('voice-select'),
            connectBtn: document.getElementById('connect-btn'),
            disconnectBtn: document.getElementById('disconnect-btn'),
            status: document.getElementById('connection-status'),
            speaking: document.getElementById('speaking-indicator'),
            transcript: document.getElementById('transcript')
        };
        
        this.bindEvents();
        this.loadVoices();
    }
    
    bindEvents() {
        this.elements.avatarUpload.addEventListener('change', (e) => this.uploadAvatar(e));
        this.elements.voiceSelect.addEventListener('change', (e) => this.setVoice(e.target.value));
        this.elements.connectBtn.addEventListener('click', () => this.connect());
        this.elements.disconnectBtn.addEventListener('click', () => this.disconnect());
    }
    
    async loadVoices() {
        try {
            const response = await fetch('/api/voices');
            const data = await response.json();
            this.elements.voiceSelect.value = data.current;
        } catch (e) {
            console.error('Failed to load voices:', e);
        }
    }
    
    async uploadAvatar(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/avatar', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                // Show preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    this.elements.image.src = e.target.result;
                    this.elements.image.style.display = 'block';
                    this.elements.placeholder.style.display = 'none';
                };
                reader.readAsDataURL(file);
                
                this.log('Avatar uploaded successfully');
            }
        } catch (e) {
            console.error('Failed to upload avatar:', e);
        }
    }
    
    async setVoice(voice) {
        try {
            const formData = new FormData();
            formData.append('voice', voice);
            
            await fetch('/api/voice', {
                method: 'POST',
                body: formData
            });
            
            this.log(`Voice set to: ${voice}`);
        } catch (e) {
            console.error('Failed to set voice:', e);
        }
    }
    
    async connect() {
        try {
            // Get microphone access
            this.localStream = await navigator.mediaDevices.getUserMedia({ 
                audio: true, 
                video: false 
            });
            
            // Connect WebSocket
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            this.ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            this.ws.onopen = () => this.onWebSocketOpen();
            this.ws.onmessage = (e) => this.onWebSocketMessage(e);
            this.ws.onclose = () => this.onWebSocketClose();
            this.ws.onerror = (e) => console.error('WebSocket error:', e);
            
        } catch (e) {
            console.error('Failed to connect:', e);
            this.setStatus('Error: ' + e.message);
        }
    }
    
    async onWebSocketOpen() {
        this.setStatus('Connected - Setting up WebRTC...');
        
        // Create peer connection
        this.pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });
        
        // Add local audio track
        this.localStream.getTracks().forEach(track => {
            this.pc.addTrack(track, this.localStream);
        });
        
        // Handle incoming video
        this.pc.ontrack = (event) => {
            if (event.track.kind === 'video') {
                this.elements.video.srcObject = event.streams[0];
                this.elements.video.style.display = 'block';
                this.elements.image.style.display = 'none';
            }
        };
        
        // Send ICE candidates
        this.pc.onicecandidate = (event) => {
            if (event.candidate) {
                this.ws.send(JSON.stringify({
                    type: 'ice-candidate',
                    candidate: event.candidate
                }));
            }
        };
        
        // Create and send offer
        const offer = await this.pc.createOffer();
        await this.pc.setLocalDescription(offer);
        
        this.ws.send(JSON.stringify({
            type: 'offer',
            sdp: offer.sdp
        }));
    }
    
    async onWebSocketMessage(event) {
        const data = JSON.parse(event.data);
        
        if (data.type === 'answer') {
            await this.pc.setRemoteDescription({
                type: 'answer',
                sdp: data.sdp
            });
            
            // Start the pipeline
            this.ws.send(JSON.stringify({ type: 'start' }));
            
            this.isConnected = true;
            this.setStatus('Connected');
            this.elements.connectBtn.disabled = true;
            this.elements.disconnectBtn.disabled = false;
            
        } else if (data.type === 'ice-candidate') {
            await this.pc.addIceCandidate(data.candidate);
            
        } else if (data.type === 'transcript') {
            this.log(`${data.role}: ${data.text}`);
            
        } else if (data.type === 'speaking') {
            this.elements.speaking.classList.toggle('hidden', !data.value);
        }
    }
    
    onWebSocketClose() {
        this.disconnect();
        this.setStatus('Disconnected');
    }
    
    disconnect() {
        if (this.pc) {
            this.pc.close();
            this.pc = null;
        }
        
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
            this.localStream = null;
        }
        
        this.isConnected = false;
        this.elements.connectBtn.disabled = false;
        this.elements.disconnectBtn.disabled = true;
        this.elements.video.style.display = 'none';
        this.elements.image.style.display = 'block';
        this.setStatus('Disconnected');
    }
    
    setStatus(status) {
        this.elements.status.textContent = status;
    }
    
    log(message) {
        const div = document.createElement('div');
        div.textContent = message;
        this.elements.transcript.appendChild(div);
        this.elements.transcript.scrollTop = this.elements.transcript.scrollHeight;
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.avatarClient = new AIAvatarClient();
});
```

### 5.3 CSS (web/styles.css)

```css
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #eee;
    min-height: 100vh;
}

.container {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem;
}

header {
    text-align: center;
    margin-bottom: 2rem;
}

header h1 {
    font-size: 2.5rem;
    background: linear-gradient(90deg, #00d9ff, #00ff88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.avatar-container {
    position: relative;
    width: 100%;
    aspect-ratio: 1;
    max-width: 500px;
    margin: 0 auto 2rem;
    background: #0a0a0a;
    border-radius: 1rem;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
}

#avatar-video,
#avatar-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

#avatar-placeholder {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #666;
    font-size: 1.2rem;
    text-align: center;
}

.controls {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    justify-content: center;
    margin-bottom: 1.5rem;
}

.control-group {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.control-group label {
    font-size: 0.9rem;
    color: #aaa;
}

input[type="file"] {
    padding: 0.5rem;
    background: #2a2a4a;
    border: 1px solid #444;
    border-radius: 0.5rem;
    color: #fff;
}

select {
    padding: 0.5rem 1rem;
    background: #2a2a4a;
    border: 1px solid #444;
    border-radius: 0.5rem;
    color: #fff;
    cursor: pointer;
}

button {
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 0.5rem;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.2s;
}

button.primary {
    background: linear-gradient(90deg, #00d9ff, #00ff88);
    color: #000;
    font-weight: 600;
}

button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(0, 217, 255, 0.4);
}

button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
}

#disconnect-btn {
    background: #ff4757;
    color: #fff;
}

.status {
    text-align: center;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: center;
    gap: 1rem;
}

#connection-status {
    padding: 0.5rem 1rem;
    background: #2a2a4a;
    border-radius: 2rem;
    font-size: 0.9rem;
}

#speaking-indicator {
    padding: 0.5rem 1rem;
    background: #00ff88;
    color: #000;
    border-radius: 2rem;
    font-size: 0.9rem;
    animation: pulse 1.5s infinite;
}

#speaking-indicator.hidden {
    display: none;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

.transcript {
    background: #0a0a0a;
    border-radius: 1rem;
    padding: 1.5rem;
    max-height: 300px;
    overflow-y: auto;
}

.transcript div {
    padding: 0.5rem 0;
    border-bottom: 1px solid #222;
}

.transcript div:last-child {
    border-bottom: none;
}
```

---

## Phase 6: Environment File

### .env.example

```env
# Groq API Key (required for STT and TTS)
GROQ_API_KEY=your_groq_api_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b

# Server Configuration
HOST=0.0.0.0
PORT=8765
```

---

## Phase 7: Running the Application

### 7.1 Start Ollama

```powershell
# In a separate terminal
ollama serve
```

### 7.2 Start the Server

```powershell
conda activate TalkToMe
cd f:\Projects\LLM\TalkToMe

# Set environment variables
$env:GROQ_API_KEY = "your_api_key"

# Run server
python -m src.server
```

### 7.3 Access the Application

Open browser to: `http://localhost:8765`

1. Upload an avatar image (portrait photo)
2. Select a voice
3. Click "Connect"
4. Start talking!

---

## Phase 8: Testing

### 8.1 Test Ditto Service

```python
# tests/test_ditto_service.py
import pytest
import asyncio
import numpy as np

from src.services.ditto_video import DittoVideoService
from src.config import config


@pytest.fixture
def ditto_service():
    return DittoVideoService(
        config_path=config.ditto_config_path,
        checkpoint_path=config.ditto_checkpoint_path
    )


@pytest.mark.asyncio
async def test_service_initialization(ditto_service):
    """Test that Ditto service initializes correctly."""
    from pipecat.frames.frames import StartFrame
    await ditto_service.start(StartFrame())
    assert ditto_service._sdk is not None


@pytest.mark.asyncio
async def test_source_image_loading(ditto_service):
    """Test loading a source image."""
    # Create a dummy image
    test_image = np.zeros((512, 512, 3), dtype=np.uint8)
    ditto_service.set_source_image(test_image)
    assert ditto_service._source_image is not None
```

### 8.2 Test Pipeline

```python
# tests/test_pipeline.py
import pytest
from src.pipeline import OllamaLLMService


@pytest.mark.asyncio
async def test_ollama_service():
    """Test Ollama LLM service."""
    service = OllamaLLMService(
        base_url="http://localhost:11434",
        model="mistral:7b"
    )
    # Basic initialization test
    assert service._model == "mistral:7b"
```

---

## Phase 9: Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce image resolution in Ditto config
   - Close other GPU applications

2. **Ditto Import Errors**
   - Ensure ditto submodule is initialized
   - Check sys.path includes ditto directory

3. **WebRTC Connection Fails**
   - Check firewall settings
   - Ensure STUN server is accessible
   - Try different browser

4. **Audio/Video Sync Issues**
   - Adjust buffer sizes in DittoVideoService
   - Check audio sample rate matches (16kHz)

5. **Slow Response**
   - Use smaller Ollama model (e.g., `phi3:mini`)
   - Reduce video resolution

---

## Phase 10: Future Enhancements

- [ ] Multiple avatar presets
- [ ] Voice cloning support
- [ ] Emotion detection and expression
- [ ] Background removal/replacement
- [ ] Mobile-optimized UI
- [ ] Recording/export conversations
- [ ] Multi-user sessions
- [ ] Custom LLM system prompts via UI

---

## Quick Start Checklist

- [ ] Create `f:\Projects\LLM\TalkToMe\` directory
- [ ] Initialize git repository
- [ ] Create conda environment (Python 3.10)
- [ ] Install PyTorch with CUDA 12.1
- [ ] Clone Ditto as submodule
- [ ] Download Ditto checkpoints (~3GB)
- [ ] Install requirements.txt
- [ ] Create .env with GROQ_API_KEY
- [ ] Start Ollama server
- [ ] Run the application
- [ ] Upload avatar and test!
