"""
Kokoro TTS Service for Pipecat.

Local text-to-speech using kokoro-onnx - fast, high quality, no API needed.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional
import logging

import numpy as np

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    LLMFullResponseEndFrame,
    StartFrame,
    EndFrame,
    CancelFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)

# Cache directory for model files
KOKORO_CACHE_DIR = Path(os.path.expanduser("~/.cache/kokoro-onnx"))
KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

# Voice mapping from simple names to Kokoro voice IDs
# See: https://github.com/thewh1teagle/kokoro-onnx for full list
VOICE_MAP = {
    # American Female
    "af_heart": "af_heart",
    "af_bella": "af_bella",
    "af_nicole": "af_nicole",
    "af_sarah": "af_sarah",
    "af_sky": "af_sky",
    # American Male
    "am_adam": "am_adam",
    "am_michael": "am_michael",
    # British Female
    "bf_emma": "bf_emma",
    "bf_isabella": "bf_isabella",
    # British Male  
    "bm_george": "bm_george",
    "bm_lewis": "bm_lewis",
    # Simple aliases
    "autumn": "af_heart",  # Map Groq voice names to Kokoro
    "breeze": "af_bella",
    "ember": "am_adam",
    "juniper": "af_sarah",
}


def _download_file(url: str, dest: Path):
    """Download a file from a URL to a destination path."""
    import requests
    logger.info(f"Downloading {url} to {dest}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Downloaded {dest}")


def _ensure_model_files(model_path: Path, voices_path: Path):
    """Download model files if they don't exist."""
    if not model_path.exists():
        _download_file(KOKORO_MODEL_URL, model_path)
    if not voices_path.exists():
        _download_file(KOKORO_VOICES_URL, voices_path)


class KokoroTTSService(FrameProcessor):
    """
    Pipecat TTS service using Kokoro-ONNX (local, fast, high quality).
    
    Automatically downloads ~300MB model on first use.
    """
    
    def __init__(
        self,
        voice: str = "af_heart",
        sample_rate: int = 24000,
        model_path: Optional[str] = None,
        voices_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self._voice = VOICE_MAP.get(voice, voice)
        self._sample_rate = sample_rate
        self._text_buffer = ""
        self._kokoro = None
        
        # Model paths
        self._model_path = Path(model_path) if model_path else KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"
        self._voices_path = Path(voices_path) if voices_path else KOKORO_CACHE_DIR / "voices-v1.0.bin"
        
        logger.info(f"KokoroTTSService created with voice: {self._voice}")
        
    def set_voice(self, voice: str):
        """Change the TTS voice."""
        self._voice = VOICE_MAP.get(voice, voice)
        logger.info(f"Voice changed to: {self._voice}")
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames."""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, StartFrame):
            # Initialize the model on StartFrame
            await self._initialize_model()
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, TextFrame):
            # Buffer text
            self._text_buffer += frame.text
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, LLMFullResponseEndFrame):
            # Generate TTS for buffered text
            if self._text_buffer.strip():
                await self._generate_tts(self._text_buffer.strip())
            self._text_buffer = ""
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, EndFrame):
            self._kokoro = None
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, CancelFrame):
            self._text_buffer = ""
            await self.push_frame(frame, direction)
            
        else:
            await self.push_frame(frame, direction)
    
    async def _initialize_model(self):
        """Load the Kokoro model if not already loaded."""
        if self._kokoro is not None:
            return
            
        # Ensure model files are downloaded
        logger.info("Ensuring Kokoro model files are downloaded...")
        _ensure_model_files(self._model_path, self._voices_path)
        
        # Load the model
        logger.info("Loading Kokoro model...")
        from kokoro_onnx import Kokoro
        self._kokoro = Kokoro(str(self._model_path), str(self._voices_path))
        logger.info("KokoroTTSService initialized - model loaded")
            
    async def _generate_tts(self, text: str):
        """Generate TTS audio from text using Kokoro."""
        logger.info(f"Generating TTS for: {text[:50]}...")
        
        if not self._kokoro:
            logger.error("Kokoro model not loaded!")
            return
        
        try:
            # Signal TTS started
            await self.push_frame(TTSStartedFrame())
            
            # Generate audio using Kokoro's streaming API
            # Run in executor to not block the event loop
            loop = asyncio.get_event_loop()
            
            # Use create_stream for async streaming
            stream = self._kokoro.create_stream(
                text, 
                voice=self._voice, 
                lang="en-us",
                speed=1.0
            )
            
            async for samples, sample_rate in stream:
                # Convert float32 samples to int16 PCM
                audio_int16 = (samples * 32767).astype(np.int16)
                
                # Resample if needed
                if sample_rate != self._sample_rate:
                    # Simple resampling using numpy
                    ratio = self._sample_rate / sample_rate
                    new_length = int(len(audio_int16) * ratio)
                    indices = np.linspace(0, len(audio_int16) - 1, new_length).astype(int)
                    audio_int16 = audio_int16[indices]
                
                audio_bytes = audio_int16.tobytes()
                
                # Output as TTSAudioRawFrame
                frame = TTSAudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=self._sample_rate,
                    num_channels=1
                )
                await self.push_frame(frame)
            
            logger.info(f"TTS generation complete")
            
            # Signal TTS stopped
            await self.push_frame(TTSStoppedFrame())
            
        except Exception as e:
            logger.error(f"TTS generation error: {e}", exc_info=True)
            await self.push_frame(TTSStoppedFrame())
