"""
Edge TTS Service for Pipecat.

Uses Microsoft Edge's free TTS API as a fallback when Groq TTS is rate-limited.
"""

import asyncio
import edge_tts
import io
import logging

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

# Voice mapping from simple names to Edge TTS voices
VOICE_MAP = {
    "autumn": "en-US-JennyNeural",
    "breeze": "en-US-AriaNeural", 
    "ember": "en-US-GuyNeural",
    "juniper": "en-US-SaraNeural",
}


class EdgeTTSService(FrameProcessor):
    """
    Pipecat TTS service using Microsoft Edge TTS (free, no API key needed).
    """
    
    def __init__(
        self,
        voice: str = "autumn",
        sample_rate: int = 24000,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self._voice = VOICE_MAP.get(voice, voice)
        self._sample_rate = sample_rate
        self._text_buffer = ""
        
        logger.info(f"EdgeTTSService created with voice: {self._voice}")
        
    async def start(self, frame: StartFrame):
        """Initialize service."""
        await super().start(frame)
        logger.info("EdgeTTSService started")
        
    async def stop(self, frame: EndFrame):
        """Cleanup."""
        logger.info("EdgeTTSService stopping")
        await super().stop(frame)
        
    async def cancel(self, frame: CancelFrame):
        """Cancel."""
        self._text_buffer = ""
        await super().cancel(frame)
        
    def set_voice(self, voice: str):
        """Change the TTS voice."""
        self._voice = VOICE_MAP.get(voice, voice)
        logger.info(f"Voice changed to: {self._voice}")
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames."""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            # Buffer text
            self._text_buffer += frame.text
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, LLMFullResponseEndFrame):
            # Generate TTS for buffered text
            if self._text_buffer.strip():
                await self._generate_tts(self._text_buffer.strip())
            self._text_buffer = ""
            await self.push_frame(frame, direction)
            
        else:
            await self.push_frame(frame, direction)
            
    async def _generate_tts(self, text: str):
        """Generate TTS audio from text."""
        logger.info(f"Generating TTS for: {text[:50]}...")
        
        try:
            # Signal TTS started
            await self.push_frame(TTSStartedFrame())
            
            # Use Edge TTS to generate audio
            communicate = edge_tts.Communicate(text, self._voice)
            
            # Collect audio data
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            if audio_data:
                # Edge TTS outputs MP3, convert to raw PCM using pydub + ffmpeg
                from pydub import AudioSegment
                
                audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
                audio = audio.set_frame_rate(self._sample_rate)
                audio = audio.set_channels(1)
                audio = audio.set_sample_width(2)  # 16-bit
                
                raw_data = audio.raw_data
                
                # Output as TTSAudioRawFrame
                frame = TTSAudioRawFrame(
                    audio=raw_data,
                    sample_rate=self._sample_rate,
                    num_channels=1
                )
                await self.push_frame(frame)
                logger.info(f"TTS audio generated: {len(raw_data)} bytes")
            
            # Signal TTS stopped
            await self.push_frame(TTSStoppedFrame())
            
        except Exception as e:
            logger.error(f"TTS generation error: {e}", exc_info=True)
            await self.push_frame(TTSStoppedFrame())
