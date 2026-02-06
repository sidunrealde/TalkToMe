import httpx
import asyncio
import json
import logging
from typing import Optional, Callable

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.whisper.stt import WhisperSTTService, Model
from pipecat.transcriptions.language import Language
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame, 
    LLMMessagesFrame, 
    TextFrame, 
    LLMFullResponseEndFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TranscriptionFrame,
    InterruptionFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)

from .config import config
from .services.ditto_realtime import DittoRealtimeService
from .services.static_avatar import StaticAvatarService
from .services.kokoro_tts import KokoroTTSService
from .services.avatar_manager import AvatarManager

# Setup logging
logger = logging.getLogger(__name__)


class OllamaLLMService(FrameProcessor):
    """Simple Ollama LLM integration for Pipecat."""
    
    def __init__(self, base_url: str, model: str, **kwargs):
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(timeout=60.0)
        self._context = []
        self._generating = False
        self._cancelled = False
        logger.info(f"OllamaLLMService initialized with model: {model}")
        
    def create_context_aggregator(self, context):
        """Create a context aggregator - simplified version."""
        return SimpleContextAggregator(context, self)
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Handle interruption - stop generating
        if isinstance(frame, InterruptionFrame):
            logger.info("🛑 Interruption received, cancelling generation")
            self._cancelled = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStartedSpeakingFrame):
            # User started speaking - prepare to interrupt
            if self._generating:
                logger.info("🛑 User started speaking during generation - interrupting")
                self._cancelled = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMMessagesFrame):
            logger.debug(f"OllamaLLM received LLMMessagesFrame with {len(frame.messages)} messages")
            self._context = frame.messages
            self._cancelled = False
            await self._generate_response()
        else:
            await self.push_frame(frame, direction)
            
    async def _generate_response(self):
        logger.info(f"Generating response from Ollama...")
        self._generating = True
        self._cancelled = False
        
        try:
            response = await self._client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": self._context,
                    "stream": True
                }
            )
            
            full_response = ""
            async for line in response.aiter_lines():
                # Check for cancellation
                if self._cancelled:
                    logger.info("🛑 Generation cancelled due to interruption")
                    break
                    
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        text = data["message"]["content"]
                        if text:
                            full_response += text
                            await self.push_frame(TextFrame(text=text))
            
            if full_response:
                logger.info(f"Ollama response: {full_response[:100]}...")
            await self.push_frame(LLMFullResponseEndFrame())
            
        except Exception as e:
            logger.error(f"Ollama error: {e}", exc_info=True)
            await self.push_frame(TextFrame(text="I'm having trouble responding right now."))
            await self.push_frame(LLMFullResponseEndFrame())
        finally:
            self._generating = False


class EventForwarder(FrameProcessor):
    """Forwards pipeline events to the client via a callback (e.g., WebRTC data channel)."""
    
    def __init__(self, send_message_callback: Callable, **kwargs):
        super().__init__(**kwargs)
        self._send_message = send_message_callback
        self._current_response = ""
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Handle different frame types and forward to client
        if isinstance(frame, UserStartedSpeakingFrame):
            logger.debug("Event: user_started_speaking")
            self._send_message({"type": "user_started_speaking"})
            
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("Event: user_stopped_speaking")
            self._send_message({"type": "user_stopped_speaking"})
            
        elif isinstance(frame, TranscriptionFrame):
            text = frame.text.strip() if frame.text else ""
            if text:
                logger.debug(f"Event: transcription - {text[:50]}...")
                self._send_message({"type": "transcription", "text": text})
                
        elif isinstance(frame, LLMMessagesFrame):
            logger.debug("Event: llm_started")
            self._send_message({"type": "llm_started"})
            self._current_response = ""
            
        elif isinstance(frame, TextFrame):
            # Accumulate response text
            self._current_response += frame.text
            
        elif isinstance(frame, LLMFullResponseEndFrame):
            # Send complete response
            if self._current_response.strip():
                logger.debug(f"Event: response - {self._current_response[:50]}...")
                self._send_message({"type": "response", "text": self._current_response.strip()})
            self._current_response = ""
            
        elif isinstance(frame, BotStartedSpeakingFrame):
            logger.debug("Event: bot_started_speaking")
            self._send_message({"type": "bot_started_speaking"})
            
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.debug("Event: bot_stopped_speaking")
            self._send_message({"type": "bot_stopped_speaking"})
        
        # Always pass frame through
        await self.push_frame(frame, direction)


class SimpleContextAggregator:
    """Simple context aggregator for managing conversation history."""
    
    def __init__(self, context, llm_service):
        self._context = context
        self._llm = llm_service
        
    def user(self):
        return UserContextAggregator(self._context)
        
    def assistant(self):
        return AssistantContextAggregator(self._context)


class UserContextAggregator(FrameProcessor):
    """Aggregates user transcripts into context and triggers LLM."""
    
    def __init__(self, context, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._text_buffer = ""
        self._is_speech_active = False
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Handle TranscriptionFrame from STT service (this is the main output from GroqSTT)
        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip() if frame.text else ""
            if text:
                logger.info(f"🎤 Transcription received: '{text}'")
                self._context.messages.append({
                    "role": "user",
                    "content": text
                })
                # Send LLMMessagesFrame to trigger LLM
                logger.info(f"Sending to LLM with {len(self._context.messages)} messages")
                await self.push_frame(LLMMessagesFrame(messages=self._context.messages))
            else:
                logger.debug("Empty transcription received, skipping")
        # Handle TextFrame - for typed text input, trigger LLM immediately
        elif isinstance(frame, TextFrame):
            text = frame.text.strip() if frame.text else ""
            if text and not self._is_speech_active:
                # This is likely typed text input - process immediately
                logger.info(f"⌨️ Text input received: '{text}'")
                self._context.messages.append({
                    "role": "user",
                    "content": text
                })
                logger.info(f"Sending to LLM with {len(self._context.messages)} messages")
                await self.push_frame(LLMMessagesFrame(messages=self._context.messages))
            else:
                # Buffer for speech-based input
                self._text_buffer += frame.text
                logger.debug(f"User text buffer: {self._text_buffer}")
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._is_speech_active = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._is_speech_active = False
            if self._text_buffer.strip():
                logger.info(f"User said (from buffer): {self._text_buffer.strip()}")
                self._context.messages.append({
                    "role": "user",
                    "content": self._text_buffer.strip()
                })
                # Send LLMMessagesFrame to trigger LLM
                await self.push_frame(LLMMessagesFrame(messages=self._context.messages))
                self._text_buffer = ""
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class AssistantContextAggregator(FrameProcessor):
    """Aggregates assistant responses into context."""
    
    def __init__(self, context, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._text_buffer = ""
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            self._text_buffer += frame.text
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._text_buffer.strip():
                self._context.messages.append({
                    "role": "assistant", 
                    "content": self._text_buffer.strip()
                })
                self._text_buffer = ""
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


async def create_pipeline(
    transport,
    avatar_manager: AvatarManager,
    webrtc_connection=None,
    enable_video: bool = True
) -> PipelineTask:
    """Create the full conversation pipeline."""
    logger.info("Creating conversation pipeline...")
    
    # Speech-to-text (Local Whisper via faster-whisper)
    whisper_model = config.whisper_model
    if hasattr(Model, whisper_model.upper()):
        whisper_model = getattr(Model, whisper_model.upper())
    logger.debug(f"Creating local WhisperSTTService with model: {whisper_model}")
    stt = WhisperSTTService(
        model=whisper_model,
        device=config.whisper_device,
        compute_type=config.whisper_compute_type,
        language=Language.EN,
        no_speech_prob=0.3,
    )
    logger.debug("STT service created (local Whisper)")
    
    # LLM
    logger.debug(f"Creating OllamaLLMService with model: {config.ollama_model}")
    llm = OllamaLLMService(
        base_url=config.ollama_base_url,
        model=config.ollama_model
    )
    logger.debug("LLM service created")
    
    # Text-to-speech - Use Kokoro (local, fast, high quality)
    logger.debug(f"Creating KokoroTTSService with voice: {avatar_manager.current_voice}")
    tts = KokoroTTSService(
        voice=avatar_manager.current_voice,
        sample_rate=24000,
    )
    logger.debug("TTS service created (Kokoro)")
    
    # Video service (avatar display)
    video_service = None
    has_avatar = avatar_manager.current_avatar is not None
    
    if enable_video and has_avatar:
        avatar_path = avatar_manager.get_avatar_path()
        
        # Try Ditto real-time if enabled in config
        if config.ditto_use_realtime and avatar_path:
            try:
                logger.info("Initializing Ditto real-time video service...")
                from src.services.ditto_realtime import DittoRealtimeService
                video_service = DittoRealtimeService(
                    config_path=config.ditto_config_path,
                    checkpoint_path=config.ditto_checkpoint_path,
                    source_image_path=avatar_path,
                    fps=config.ditto_fps,
                    output_size=(512, 512),
                )
                logger.info("Ditto real-time service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Ditto: {e}. Falling back to static avatar.")
                video_service = None
        
        # Fallback to static avatar
        if video_service is None:
            logger.info("Initializing static avatar video service...")
            video_service = StaticAvatarService(
                source_image=avatar_manager.current_avatar,
                fps=config.ditto_fps,
                output_size=(512, 512),
            )
            logger.info("Static avatar service initialized")
    elif enable_video and not has_avatar:
        logger.info("No avatar loaded - video disabled. Upload an avatar to enable.")
    
    # Context aggregator for conversation history
    context = OpenAILLMContext(
        messages=[{"role": "system", "content": config.system_prompt}]
    )
    context_aggregator = llm.create_context_aggregator(context)
    logger.debug("Context aggregator created")
    
    # Event forwarder for sending events to client via data channel
    event_forwarder = None
    if webrtc_connection:
        def send_to_client(message):
            try:
                webrtc_connection.send_app_message(message)
            except Exception as e:
                logger.warning(f"Failed to send message to client: {e}")
        
        event_forwarder = EventForwarder(send_to_client)
        logger.debug("Event forwarder created")
    
    # Build pipeline
    pipeline_processors = [
        transport.input(),          # Audio from user
    ]
    
    # Add event forwarder early to catch all events
    if event_forwarder:
        pipeline_processors.append(event_forwarder)
    
    pipeline_processors.extend([
        stt,                        # Speech to text
        context_aggregator.user(),  # Add user message to context
        llm,                        # Generate response
        tts,                        # Text to speech
    ])
    
    # Add video service if available
    if video_service:
        pipeline_processors.append(video_service)
        logger.info("Video output enabled in pipeline")
    else:
        logger.info("Audio-only pipeline (no video)")
        
    pipeline_processors.append(transport.output())  # Send audio/video to user
    
    logger.info(f"Pipeline processors: {[p.__class__.__name__ for p in pipeline_processors]}")
    
    pipeline = Pipeline(pipeline_processors)
    
    # Create task with interruption support
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True
        )
    )
    
    logger.info("Pipeline task created successfully")
    return task
