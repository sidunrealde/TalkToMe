"""
Real-time Ditto TalkingHead Video Service for Pipecat.

This module provides real-time talking head video generation using Ditto's
streaming pipeline with TensorRT acceleration.
"""

import asyncio
import threading
import queue
import sys
import os
import numpy as np
from typing import Optional, Callable
import logging
import cv2
import time

# Add TensorRT DLLs to PATH before importing anything that uses TRT
TENSORRT_BIN = os.getenv("TENSORRT_BIN_PATH", r"F:\nvidia\TensorRT-10.15.1.29\bin")
if TENSORRT_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = TENSORRT_BIN + os.pathsep + os.environ.get("PATH", "")

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

# Add ditto to path
DITTO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ditto')
if DITTO_PATH not in sys.path:
    sys.path.insert(0, DITTO_PATH)


class RealtimeDittoSDK:
    """
    Modified Ditto StreamSDK that outputs frames to a callback instead of file.
    
    This wraps Ditto's streaming pipeline and replaces the writer with a
    callback function for real-time frame output.
    """
    
    def __init__(self, cfg_pkl: str, data_root: str, frame_callback: Callable[[np.ndarray], None], **kwargs):
        self.frame_callback = frame_callback
        self._initialized = False
        self._setup_complete = False
        
        try:
            from core.atomic_components.avatar_registrar import AvatarRegistrar, smooth_x_s_info_lst
            from core.atomic_components.condition_handler import ConditionHandler, _mirror_index
            from core.atomic_components.audio2motion import Audio2Motion
            from core.atomic_components.motion_stitch import MotionStitch
            from core.atomic_components.warp_f3d import WarpF3D
            from core.atomic_components.decode_f3d import DecodeF3D
            from core.atomic_components.putback import PutBack
            from core.atomic_components.wav2feat import Wav2Feat
            from core.atomic_components.cfg import parse_cfg, print_cfg
            
            self._mirror_index = _mirror_index
            self.smooth_x_s_info_lst = smooth_x_s_info_lst
            
            [
                avatar_registrar_cfg,
                condition_handler_cfg,
                lmdm_cfg,
                stitch_network_cfg,
                warp_network_cfg,
                decoder_cfg,
                wav2feat_cfg,
                default_kwargs,
            ] = parse_cfg(cfg_pkl, data_root, kwargs)
            
            self.default_kwargs = default_kwargs
            
            self.avatar_registrar = AvatarRegistrar(**avatar_registrar_cfg)
            self.condition_handler = ConditionHandler(**condition_handler_cfg)
            self.audio2motion = Audio2Motion(lmdm_cfg)
            self.motion_stitch = MotionStitch(stitch_network_cfg)
            self.warp_f3d = WarpF3D(warp_network_cfg)
            self.decode_f3d = DecodeF3D(decoder_cfg)
            self.putback = PutBack()
            self.wav2feat = Wav2Feat(**wav2feat_cfg)
            
            self._initialized = True
            logger.info("RealtimeDittoSDK initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ditto SDK: {e}", exc_info=True)
            raise
    
    def _merge_kwargs(self, default_kwargs, run_kwargs):
        for k, v in default_kwargs.items():
            if k not in run_kwargs:
                run_kwargs[k] = v
        return run_kwargs
    
    def setup(self, source_path: str, **kwargs):
        """Setup avatar from source image."""
        kwargs = self._merge_kwargs(self.default_kwargs, kwargs)
        
        # Avatar registrar settings
        self.max_size = kwargs.get("max_size", 1920)
        self.template_n_frames = kwargs.get("template_n_frames", -1)
        self.crop_scale = kwargs.get("crop_scale", 2.3)
        self.crop_vx_ratio = kwargs.get("crop_vx_ratio", 0)
        self.crop_vy_ratio = kwargs.get("crop_vy_ratio", -0.125)
        self.crop_flag_do_rot = kwargs.get("crop_flag_do_rot", True)
        self.smo_k_s = kwargs.get('smo_k_s', 13)
        
        # Condition handler settings
        self.emo = kwargs.get("emo", 4)
        self.eye_f0_mode = kwargs.get("eye_f0_mode", False)
        self.ch_info = kwargs.get("ch_info", None)
        
        # Audio2motion settings
        self.overlap_v2 = kwargs.get("overlap_v2", 10)
        self.fix_kp_cond = kwargs.get("fix_kp_cond", 0)
        self.fix_kp_cond_dim = kwargs.get("fix_kp_cond_dim", None)
        self.sampling_timesteps = kwargs.get("sampling_timesteps", 50)
        self.online_mode = kwargs.get("online_mode", True)  # Force online mode
        self.v_min_max_for_clip = kwargs.get('v_min_max_for_clip', None)
        self.smo_k_d = kwargs.get("smo_k_d", 3)
        
        # Motion stitch settings
        self.N_d = kwargs.get("N_d", -1)
        self.use_d_keys = kwargs.get("use_d_keys", None)
        self.relative_d = kwargs.get("relative_d", True)
        self.drive_eye = kwargs.get("drive_eye", None)
        self.delta_eye_arr = kwargs.get("delta_eye_arr", None)
        self.delta_eye_open_n = kwargs.get("delta_eye_open_n", 0)
        self.fade_type = kwargs.get("fade_type", "")
        self.fade_out_keys = kwargs.get("fade_out_keys", ("exp",))
        self.flag_stitching = kwargs.get("flag_stitching", True)
        self.ctrl_info = kwargs.get("ctrl_info", dict())
        self.overall_ctrl_info = kwargs.get("overall_ctrl_info", dict())
        
        # Register avatar
        crop_kwargs = {
            "crop_scale": self.crop_scale,
            "crop_vx_ratio": self.crop_vx_ratio,
            "crop_vy_ratio": self.crop_vy_ratio,
            "crop_flag_do_rot": self.crop_flag_do_rot,
        }
        n_frames = self.template_n_frames if self.template_n_frames > 0 else self.N_d
        
        logger.info(f"Registering avatar from: {source_path}")
        source_info = self.avatar_registrar(
            source_path, 
            max_dim=self.max_size, 
            n_frames=n_frames, 
            **crop_kwargs,
        )
        
        if len(source_info["x_s_info_lst"]) > 1 and self.smo_k_s > 1:
            source_info["x_s_info_lst"] = self.smooth_x_s_info_lst(
                source_info["x_s_info_lst"], smo_k=self.smo_k_s
            )
        
        self.source_info = source_info
        self.source_info_frames = len(source_info["x_s_info_lst"])
        
        # Setup condition handler
        self.condition_handler.setup(
            source_info, self.emo, 
            eye_f0_mode=self.eye_f0_mode, 
            ch_info=self.ch_info
        )
        
        # Setup audio2motion (LMDM)
        x_s_info_0 = self.condition_handler.x_s_info_0
        self.audio2motion.setup(
            x_s_info_0, 
            overlap_v2=self.overlap_v2,
            fix_kp_cond=self.fix_kp_cond,
            fix_kp_cond_dim=self.fix_kp_cond_dim,
            sampling_timesteps=self.sampling_timesteps,
            online_mode=self.online_mode,
            v_min_max_for_clip=self.v_min_max_for_clip,
            smo_k_d=self.smo_k_d,
        )
        
        # Setup motion stitch
        is_image_flag = source_info["is_image_flag"]
        x_s_info = source_info['x_s_info_lst'][0]
        self.motion_stitch.setup(
            N_d=self.N_d,
            use_d_keys=self.use_d_keys,
            relative_d=self.relative_d,
            drive_eye=self.drive_eye,
            delta_eye_arr=self.delta_eye_arr,
            delta_eye_open_n=self.delta_eye_open_n,
            fade_out_keys=self.fade_out_keys,
            fade_type=self.fade_type,
            flag_stitching=self.flag_stitching,
            is_image_flag=is_image_flag,
            x_s_info=x_s_info,
            d0=None,
            ch_info=self.ch_info,
            overall_ctrl_info=self.overall_ctrl_info,
        )
        
        # Initialize audio buffer for online mode
        self.audio_feat = self.wav2feat.wav2feat(
            np.zeros((self.overlap_v2 * 640,), dtype=np.float32), sr=16000
        )
        self.cond_idx_start = 0 - len(self.audio_feat)
        
        # Initialize processing state
        self.res_kp_seq = None
        self.res_kp_seq_valid_start = None
        self.global_idx = 0
        self.local_idx = 0
        self.gen_frame_idx = 0
        self.item_buffer = np.zeros((0, self.wav2feat.feat_dim), dtype=np.float32)
        
        self._setup_complete = True
        logger.info("Avatar setup complete")
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> list:
        """
        Process audio chunk using OFFLINE batch mode for perfect lip sync.
        
        This uses the same approach as Ditto's stream_pipeline_offline.py,
        which processes all audio at once and produces one frame per audio feature.
        
        Args:
            audio_chunk: Audio samples as float32 numpy array (16kHz)
            
        Returns:
            List of generated RGB frames
        """
        if not self._setup_complete:
            logger.warning("process_audio_chunk: setup not complete, returning empty")
            return []
        
        frames = []
        
        # Convert audio to features - one feature per frame at 25fps
        # For 16kHz audio, each feature represents 640 samples (40ms)
        aud_feat = self.wav2feat.wav2feat(audio_chunk, sr=16000)
        num_frames = len(aud_feat)
        logger.warning(f"process_audio_chunk: audio shape={audio_chunk.shape}, features={num_frames}")
        
        if num_frames == 0:
            return frames
        
        # OFFLINE BATCH PROCESSING (from Ditto's stream_pipeline_offline.py)
        # Process ALL audio conditions at once
        aud_cond_all = self.condition_handler(aud_feat, 0)
        
        seq_frames = self.audio2motion.seq_frames
        valid_clip_len = self.audio2motion.valid_clip_len
        
        idx = 0
        res_kp_seq = None
        
        # Process audio in chunks, building up the motion sequence
        while idx < num_frames:
            aud_cond = aud_cond_all[idx:idx + seq_frames][None]
            
            # Pad if needed for final chunk
            if aud_cond.shape[1] < seq_frames:
                pad = np.stack([aud_cond[:, -1]] * (seq_frames - aud_cond.shape[1]), 1)
                aud_cond = np.concatenate([aud_cond, pad], 1)
            
            res_kp_seq = self.audio2motion(aud_cond, res_kp_seq)
            idx += valid_clip_len
        
        # Trim to exact number of frames (one per audio feature)
        res_kp_seq = res_kp_seq[:, :num_frames]
        
        # Apply smoothing
        res_kp_seq = self.audio2motion._smo(res_kp_seq, 0, res_kp_seq.shape[1])
        
        # Convert motion to frame info
        x_d_info_list = self.audio2motion.cvt_fmt(res_kp_seq)
        
        logger.warning(f"Generated {len(x_d_info_list)} motion frames for {num_frames} audio features")
        
        # Generate video frames
        for gen_frame_idx, x_d_info in enumerate(x_d_info_list):
            frame_idx = self._mirror_index(gen_frame_idx, self.source_info_frames)
            ctrl_kwargs = self._get_ctrl_info(gen_frame_idx)
            
            # Generate frame
            frame = self._generate_frame(frame_idx, x_d_info, ctrl_kwargs)
            if frame is not None:
                frames.append(frame)
                if self.frame_callback:
                    self.frame_callback(frame)
        
        logger.warning(f"process_audio_chunk: Generated {len(frames)} video frames")
        return frames
    
    def _generate_frame(self, frame_idx: int, x_d_info, ctrl_kwargs: dict) -> Optional[np.ndarray]:
        """Generate a single video frame."""
        try:
            # Motion stitch
            x_s_info = self.source_info["x_s_info_lst"][frame_idx]
            x_s, x_d = self.motion_stitch(x_s_info, x_d_info, **ctrl_kwargs)
            
            # Warp
            f_s = self.source_info["f_s_lst"][frame_idx]
            f_3d = self.warp_f3d(f_s, x_s, x_d)
            
            # Decode
            render_img = self.decode_f3d(f_3d)
            
            # Putback
            frame_rgb = self.source_info["img_rgb_lst"][frame_idx]
            M_c2o = self.source_info["M_c2o_lst"][frame_idx]
            result = self.putback(frame_rgb, render_img, M_c2o)
            
            return result
            
        except Exception as e:
            logger.error(f"Frame generation error: {e}")
            return None
    
    def _get_ctrl_info(self, fid):
        try:
            if isinstance(self.ctrl_info, dict):
                return self.ctrl_info.get(fid, {})
            elif isinstance(self.ctrl_info, list):
                return self.ctrl_info[fid]
            else:
                return {}
        except Exception:
            return {}
    
    def reset(self):
        """Reset processing state for new audio stream."""
        if self._setup_complete:
            self.audio_feat = self.wav2feat.wav2feat(
                np.zeros((self.overlap_v2 * 640,), dtype=np.float32), sr=16000
            )
            self.res_kp_seq = None
            self.res_kp_seq_valid_start = None
            self.global_idx = 0
            self.local_idx = 0
            self.gen_frame_idx = 0
            self.item_buffer = np.zeros((0, self.wav2feat.feat_dim), dtype=np.float32)
    
    def close(self):
        """Cleanup resources."""
        pass


class DittoRealtimeService(FrameProcessor):
    """
    Pipecat processor that generates real-time talking head video from TTS audio.
    
    Uses Ditto's streaming pipeline with optional TensorRT acceleration for
    real-time lip-synced avatar animation.
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        source_image_path: str = None,
        fps: int = 25,
        output_size: tuple = (512, 512),
        **kwargs
    ):
        super().__init__(**kwargs)
        
        logger.warning("="*60)
        logger.warning("DittoRealtimeService __init__ CALLED")
        logger.warning(f"  config_path: {config_path}")
        logger.warning(f"  checkpoint_path: {checkpoint_path}")
        logger.warning(f"  source_image_path: {source_image_path}")
        logger.warning(f"  fps: {fps}")
        logger.warning(f"  output_size: {output_size}")
        logger.warning("="*60)
        
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._source_image_path = source_image_path
        self._fps = fps
        self._output_size = output_size
        
        self._sdk: Optional[RealtimeDittoSDK] = None
        self._is_initialized = False
        self._is_speaking = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=500)  # Larger queue for more frames
        self._lock = asyncio.Lock()
        self._last_frame_time = 0
        self._frame_interval = 1.0 / fps
        
        logger.warning(f"DittoRealtimeService __init__ COMPLETE")
    
    def _frame_callback(self, frame: np.ndarray):
        """Callback for receiving generated frames from SDK."""
        logger.warning(f"_frame_callback CALLED! Got frame shape={frame.shape}, dtype={frame.dtype}")
        try:
            self._frame_queue.put_nowait(frame)
            logger.warning(f"Frame added to queue, queue size={self._frame_queue.qsize()}")
        except queue.Full:
            # Drop oldest frame
            logger.warning("Frame queue full, dropping oldest")
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(frame)
            except:
                pass
    
    async def _initialize_sdk(self):
        """Initialize Ditto SDK."""
        logger.warning("="*60)
        logger.warning("DittoRealtimeService _initialize_sdk() CALLED")
        logger.warning(f"Config path: {self._config_path}")
        logger.warning(f"Checkpoint path: {self._checkpoint_path}")
        logger.warning(f"Source image: {self._source_image_path}")
        logger.warning("="*60)
        
        try:
            logger.warning("Creating RealtimeDittoSDK instance...")
            self._sdk = RealtimeDittoSDK(
                self._config_path,
                self._checkpoint_path,
                frame_callback=self._frame_callback
            )
            logger.warning("RealtimeDittoSDK instance created successfully")
            
            if self._source_image_path and os.path.exists(self._source_image_path):
                logger.warning(f"Setting up avatar with image: {self._source_image_path}")
                self._sdk.setup(self._source_image_path, online_mode=True, N_d=-1)
                logger.warning("Avatar setup complete")
            else:
                logger.warning(f"Source image not found or not set: {self._source_image_path}")
            
            self._is_initialized = True
            logger.warning("DittoRealtimeService fully initialized!")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ditto: {e}", exc_info=True)
            self._is_initialized = False
            logger.error(f"Ditto initialization FAILED - _is_initialized={self._is_initialized}")
    
    async def _cleanup(self):
        """Cleanup resources."""
        logger.info("DittoRealtimeService stopping...")
        if self._sdk:
            self._sdk.close()
        self._sdk = None
        self._is_initialized = False
    
    def set_source_image(self, path: str):
        """Set source image for avatar."""
        self._source_image_path = path
        if self._sdk and self._is_initialized:
            self._sdk.setup(path, online_mode=True, N_d=-1)
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames."""
        await super().process_frame(frame, direction)
        
        # Log all frame types for debugging
        frame_type = type(frame).__name__
        if frame_type in ['StartFrame', 'EndFrame', 'CancelFrame', 'TTSAudioRawFrame', 
                          'BotStartedSpeakingFrame', 'BotStoppedSpeakingFrame', 'UserStartedSpeakingFrame']:
            logger.warning(f"DittoRealtimeService received: {frame_type}")
        
        if isinstance(frame, StartFrame):
            # Initialize Ditto SDK when pipeline starts
            logger.warning("DittoRealtimeService: StartFrame received, calling _initialize_sdk()")
            await self._initialize_sdk()
            logger.warning("DittoRealtimeService: _initialize_sdk() completed, pushing StartFrame")
            await self.push_frame(frame, direction)
        
        elif isinstance(frame, EndFrame):
            # Cleanup
            await self._cleanup()
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, CancelFrame):
            # Cleanup
            await self._cleanup()
            await self.push_frame(frame, direction)
        
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._is_speaking = False
            if self._sdk:
                self._sdk.reset()
            # Clear frame queue
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except:
                    break
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._is_speaking = True
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._is_speaking = False
            # Output remaining frames
            await self._flush_frames()
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, TTSAudioRawFrame):
            # Calculate audio duration
            audio_samples = len(frame.audio) // 2  # int16 = 2 bytes per sample
            audio_duration = audio_samples / frame.sample_rate
            
            # Generate video frames FIRST (queues them up)
            await self._process_tts_audio(frame)
            
            num_frames = self._frame_queue.qsize()
            logger.warning(f"Audio duration={audio_duration:.2f}s, generated {num_frames} frames")
            
            if num_frames > 0:
                # Frame interval = audio duration / number of frames = perfect sync
                # This ensures video ends exactly when audio ends
                frame_interval = audio_duration / num_frames
                logger.warning(f"Outputting {num_frames} frames at {frame_interval*1000:.1f}ms intervals (video duration={num_frames * frame_interval:.2f}s)")
            else:
                frame_interval = 0.04  # 40ms default
            
            # Push audio and output video frames simultaneously
            await asyncio.gather(
                self.push_frame(frame, direction),
                self._output_queued_frames(frame_interval)
            )
            
        else:
            await self.push_frame(frame, direction)
    
    async def _process_tts_audio(self, audio_frame: TTSAudioRawFrame):
        """Process TTS audio and generate video frames."""
        logger.warning(f"_process_tts_audio CALLED - initialized={self._is_initialized}, sdk={self._sdk is not None}")
        
        if not self._is_initialized or not self._sdk:
            logger.warning(f"DittoRealtimeService: Skipping audio - initialized={self._is_initialized}, sdk={self._sdk is not None}")
            return
        
        logger.warning(f"DittoRealtimeService: Processing TTS audio frame, size={len(audio_frame.audio)}, rate={audio_frame.sample_rate}")
        
        async with self._lock:
            try:
                # Convert audio to float32
                audio_data = np.frombuffer(audio_frame.audio, dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0
                logger.warning(f"Converted audio to float32, shape={audio_float.shape}, min={audio_float.min():.4f}, max={audio_float.max():.4f}")
                
                # Resample if needed (Ditto expects 16kHz)
                if audio_frame.sample_rate != 16000:
                    import resampy
                    audio_float = resampy.resample(
                        audio_float, audio_frame.sample_rate, 16000
                    )
                    logger.warning(f"Resampled audio to 16kHz, new shape={audio_float.shape}")
                
                # Process audio and generate frames (queued via callback)
                # Using offline batch mode - processes all audio at once with perfect sync
                logger.warning("Calling self._sdk.process_audio_chunk (offline batch mode)...")
                queue_size_before = self._frame_queue.qsize()
                await asyncio.to_thread(
                    self._sdk.process_audio_chunk, audio_float
                )
                queue_size_after = self._frame_queue.qsize()
                logger.warning(f"process_audio_chunk complete - queue before={queue_size_before}, after={queue_size_after}")
                
                # Frames are now queued - they will be output by the caller
                
            except Exception as e:
                logger.error(f"Error processing TTS audio: {e}", exc_info=True)

    async def _output_queued_frames(self, frame_interval: float = None):
        """Output frames from the queue with proper timing."""
        if frame_interval is None:
            frame_interval = self._frame_interval
        
        frames_output = 0
        start_time = time.time()
        
        while not self._frame_queue.empty():
            try:
                frame = self._frame_queue.get_nowait()
                
                # Calculate when this frame should be displayed
                target_time = start_time + (frames_output * frame_interval)
                current_time = time.time()
                
                # Wait until it's time to display this frame
                if current_time < target_time:
                    await asyncio.sleep(target_time - current_time)
                
                await self._output_video_frame(frame)
                frames_output += 1
                
            except queue.Empty:
                break
        
        if frames_output > 0:
            elapsed = time.time() - start_time
            logger.warning(f"_output_queued_frames: Output {frames_output} frames in {elapsed:.2f}s (target interval={frame_interval*1000:.1f}ms)")
    
    async def _flush_frames(self):
        """Flush remaining frames."""
        await self._output_queued_frames()
    
    async def _output_video_frame(self, frame: np.ndarray):
        """Output a video frame."""
        logger.warning(f"_output_video_frame called with frame shape={frame.shape}")
        try:
            # Resize if needed
            if frame.shape[:2] != self._output_size:
                frame = cv2.resize(frame, self._output_size)
                logger.warning(f"Resized frame to {self._output_size}")
            
            height, width = frame.shape[:2]
            
            # Ensure RGB format
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            image_frame = OutputImageRawFrame(
                image=frame.tobytes(),
                size=(width, height),
                format="RGB"
            )
            logger.warning(f"DittoRealtimeService: PUSHING video frame {width}x{height}")
            await self.push_frame(image_frame)
            
        except Exception as e:
            logger.error(f"Error outputting video frame: {e}", exc_info=True)
