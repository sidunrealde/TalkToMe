"""
Static Avatar Video Service for Pipecat.

Simple video service that outputs avatar image as video frames.
This ensures the avatar is always visible in the WebRTC stream.
"""

import asyncio
import numpy as np
from typing import Optional
import logging
import cv2
import time

from pipecat.frames.frames import (
    Frame,
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


class StaticAvatarService(FrameProcessor):
    """
    Simple avatar video service that outputs static avatar frames.
    
    This ensures the avatar is always visible in the WebRTC video stream.
    Frames are output at a steady rate when the bot is speaking,
    and periodically when idle to maintain the video track.
    """
    
    def __init__(
        self,
        source_image: np.ndarray = None,
        fps: int = 25,
        output_size: tuple = (512, 512),
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self._fps = fps
        self._output_size = output_size
        self._source_image: Optional[np.ndarray] = None
        self._processed_image: Optional[np.ndarray] = None
        self._is_speaking = False
        self._is_running = False
        self._frame_task = None
        self._last_frame_time = 0
        self._frame_interval = 1.0 / fps
        self._idle_frame_interval = 1.0  # 1 fps when idle
        
        if source_image is not None:
            self.set_source_image(source_image)
            
        logger.info(f"StaticAvatarService created (fps={fps}, size={output_size})")
        
    def set_source_image(self, image: np.ndarray):
        """Set the avatar image."""
        logger.info(f"Setting avatar image: shape={image.shape}")
        self._source_image = image
        
        # Process and resize image
        processed = image.copy()
        
        # Ensure RGB format
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        elif processed.shape[2] == 4:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGBA2RGB)
        elif processed.shape[2] == 3:
            # Assume it's already RGB from PIL/numpy
            pass
            
        # Resize to output size
        if processed.shape[:2] != self._output_size:
            processed = cv2.resize(processed, self._output_size)
            
        self._processed_image = processed
        logger.info(f"Avatar processed: {self._processed_image.shape}")
        
    async def _frame_loop(self):
        """Background loop to output video frames."""
        logger.info("Avatar frame loop started")
        
        while self._is_running:
            try:
                if self._processed_image is not None:
                    current_time = time.time()
                    
                    # Determine frame interval based on state
                    interval = self._frame_interval if self._is_speaking else self._idle_frame_interval
                    
                    if current_time - self._last_frame_time >= interval:
                        await self._output_frame()
                        self._last_frame_time = current_time
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.01)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Frame loop error: {e}")
                await asyncio.sleep(0.1)
                
        logger.info("Avatar frame loop ended")
        
    async def _output_frame(self):
        """Output a single video frame."""
        if self._processed_image is None:
            return
            
        try:
            height, width = self._processed_image.shape[:2]
            
            image_frame = OutputImageRawFrame(
                image=self._processed_image.tobytes(),
                size=(width, height),
                format="RGB"
            )
            await self.push_frame(image_frame)
            
        except Exception as e:
            logger.error(f"Error outputting frame: {e}")
            
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames."""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, StartFrame):
            # Initialize and start frame loop
            logger.info("StaticAvatarService received StartFrame - initializing")
            self._is_running = True
            if self._frame_task is None or self._frame_task.done():
                self._frame_task = asyncio.create_task(self._frame_loop())
            await self.push_frame(frame, direction)
        
        elif isinstance(frame, EndFrame):
            # Stop the service
            logger.info("StaticAvatarService received EndFrame - stopping")
            self._is_running = False
            if self._frame_task:
                self._frame_task.cancel()
                try:
                    await self._frame_task
                except asyncio.CancelledError:
                    pass
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, CancelFrame):
            self._is_running = False
            if self._frame_task:
                self._frame_task.cancel()
            await self.push_frame(frame, direction)
        
        elif isinstance(frame, UserStartedSpeakingFrame):
            # User interrupted - slow down video
            self._is_speaking = False
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, BotStartedSpeakingFrame):
            # Bot speaking - output more frames
            self._is_speaking = True
            logger.debug("Bot started speaking - increasing video frame rate")
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._is_speaking = False
            logger.debug("Bot stopped speaking - reducing video frame rate")
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, TTSAudioRawFrame):
            # Output extra frames synced to audio
            if self._processed_image is not None:
                await self._output_frame()
            # Pass audio through
            await self.push_frame(frame, direction)
            
        else:
            await self.push_frame(frame, direction)
