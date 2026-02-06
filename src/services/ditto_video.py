"""
Ditto TalkingHead Video Service for Pipecat.

Generates real-time talking head video frames from TTS audio.
Initially provides static avatar display, with Ditto integration for future enhancement.
"""

import asyncio
import sys
import os
import numpy as np
from typing import Optional
import logging
import cv2
import time

from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    OutputImageRawFrame,
    TTSAudioRawFrame,
    StartFrame,
    EndFrame,
    CancelFrame,
    UserStartedSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)

# Add ditto to path for future use
DITTO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ditto')
if DITTO_PATH not in sys.path:
    sys.path.insert(0, DITTO_PATH)


class DittoVideoService(FrameProcessor):
    """
    Pipecat processor that generates talking head video from TTS audio.
    
    Currently outputs static avatar frames synced to audio.
    Full Ditto integration for lip-sync is available for future enhancement.
    
    Receives TTSAudioRawFrame, generates video frames,
    and outputs OutputImageRawFrame.
    """
    
    def __init__(
        self,
        config_path: str = None,
        checkpoint_path: str = None,
        device_id: int = 0,
        fps: int = 25,
        output_size: tuple = (512, 512),
        enable_lipsync: bool = False,  # Set True to enable full Ditto
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._device_id = device_id
        self._fps = fps
        self._output_size = output_size
        self._enable_lipsync = enable_lipsync
        
        self._sdk = None
        self._source_image: Optional[np.ndarray] = None
        self._is_initialized = False
        self._is_speaking = False
        self._audio_buffer = []
        self._last_frame_time = 0
        self._frame_interval = 1.0 / fps
        self._lock = asyncio.Lock()
        
        logger.info(f"DittoVideoService created (fps={fps}, lipsync={enable_lipsync})")
        
    async def start(self, frame: StartFrame):
        """Initialize service."""
        await super().start(frame)
        logger.info("DittoVideoService starting...")
        
        if self._enable_lipsync:
            try:
                self._initialize_ditto()
            except Exception as e:
                logger.warning(f"Failed to initialize Ditto lipsync: {e}")
                logger.info("Falling back to static avatar mode")
                self._enable_lipsync = False
        
        self._is_initialized = True
        
    async def stop(self, frame: EndFrame):
        """Cleanup."""
        logger.info("DittoVideoService stopping...")
        self._cleanup()
        await super().stop(frame)
        
    async def cancel(self, frame: CancelFrame):
        """Cancel and cleanup."""
        self._cleanup()
        await super().cancel(frame)
        
    def _initialize_ditto(self):
        """Initialize Ditto for lip-sync (future enhancement)."""
        # Note: Full Ditto integration requires modifying their pipeline
        # to output frames in real-time rather than to file.
        # For now, we use static avatar display.
        pass
            
    def _cleanup(self):
        """Cleanup resources."""
        if self._sdk:
            try:
                self._sdk.close()
            except:
                pass
        self._sdk = None
        self._is_initialized = False
        
    def set_source_image(self, image: np.ndarray):
        """Set the source portrait image for animation."""
        logger.info(f"Setting source image: shape={image.shape}")
        self._source_image = image
        
        # Pre-resize to output size
        if image.shape[:2] != self._output_size:
            self._source_image = cv2.resize(image, self._output_size)
            logger.info(f"Resized to {self._output_size}")
            
    def set_source_image_from_path(self, path: str):
        """Load and set source image from file path."""
        image = cv2.imread(path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.set_source_image(image)
        else:
            logger.error(f"Failed to load image from {path}")
            
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames."""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, UserStartedSpeakingFrame):
            # User interrupted
            self._is_speaking = False
            self._audio_buffer.clear()
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._is_speaking = True
            logger.debug("Bot started speaking - enabling video output")
            # Output initial frame
            if self._source_image is not None:
                await self._output_video_frame(self._source_image)
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._is_speaking = False
            logger.debug("Bot stopped speaking - pausing video output")
            # Output final static frame
            if self._source_image is not None:
                await self._output_video_frame(self._source_image)
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, TTSAudioRawFrame):
            # TTS audio - generate video frames synced to audio
            await self._process_tts_audio(frame)
            # Pass audio through
            await self.push_frame(frame, direction)
            
        else:
            await self.push_frame(frame, direction)
            
    async def _process_tts_audio(self, audio_frame: TTSAudioRawFrame):
        """Process TTS audio and generate video frames."""
        if self._source_image is None:
            logger.debug("No source image set, skipping video output")
            return
            
        async with self._lock:
            try:
                # Calculate how many frames to output based on audio duration
                audio_data = np.frombuffer(audio_frame.audio, dtype=np.int16)
                audio_duration = len(audio_data) / audio_frame.sample_rate
                
                # Output video frames at configured FPS
                current_time = time.time()
                elapsed = current_time - self._last_frame_time
                
                # Determine how many frames to output
                num_frames = max(1, int(audio_duration * self._fps))
                
                for _ in range(num_frames):
                    if current_time - self._last_frame_time >= self._frame_interval:
                        # Generate frame (static for now, lip-sync ready)
                        video_frame = await self._generate_frame(audio_data)
                        
                        if video_frame is not None:
                            await self._output_video_frame(video_frame)
                            self._last_frame_time = current_time
                        
            except Exception as e:
                logger.error(f"Error processing TTS audio: {e}", exc_info=True)
                
    async def _generate_frame(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Generate video frame from audio chunk."""
        if self._enable_lipsync and self._sdk:
            # Future: Use Ditto for lip-sync
            try:
                frame = await asyncio.to_thread(
                    self._sdk.run_chunk, audio_chunk
                )
                return frame
            except Exception as e:
                logger.debug(f"Ditto generation error: {e}")
                
        # Return static image (or with slight animation)
        return self._source_image.copy() if self._source_image is not None else None
        
    async def _output_video_frame(self, frame: np.ndarray):
        """Output a video frame."""
        try:
            height, width = frame.shape[:2]
            
            # Ensure RGB format
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
            image_frame = OutputImageRawFrame(
                image=frame.tobytes(),
                size=(width, height),
                format="RGB"
            )
            await self.push_frame(image_frame)
            
        except Exception as e:
            logger.error(f"Error outputting video frame: {e}")
