import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Local Whisper STT (using faster-whisper)
    # Models: TINY, BASE, SMALL, MEDIUM, LARGE, LARGE_V3_TURBO, DISTIL_MEDIUM_EN, DISTIL_LARGE_V2
    # Or use HuggingFace model path like "deepdml/faster-whisper-large-v3-turbo-ct2"
    whisper_model: str = os.getenv("WHISPER_MODEL", "LARGE_V3_TURBO")  # Best quality for RTX 3090
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cuda")  # cuda, cpu, or auto
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")  # float16, int8, int8_float16
    
    # Ollama LLM
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "mistral:7b")
    
    # TTS Settings
    tts_model: str = "playai/playht-tts-v3"
    tts_voice: str = "autumn"  # autumn, breeze, ember, juniper
    
    # Ditto Settings - Use TensorRT for real-time performance
    # For TensorRT hybrid + ONNX hubert: "checkpoints/ditto_cfg/v0.4_hubert_cfg_hybrid_onnx.pkl" (TRT + ONNX warp + ONNX hubert)
    # For TensorRT hybrid: "checkpoints/ditto_cfg/v0.4_hubert_cfg_hybrid.pkl" (TRT + ONNX warp)
    # For TensorRT: "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl" (requires custom plugin)
    # For PyTorch: "checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
    ditto_config_path: str = os.getenv(
        "DITTO_CONFIG_PATH",
        "checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"  # PyTorch backend (GridSample3D works in PyTorch)
    )
    # For TensorRT: "checkpoints/ditto_trt" (after running conversion)
    # For ONNX: "checkpoints/ditto_onnx"
    # For PyTorch: "checkpoints/ditto_pytorch"
    ditto_checkpoint_path: str = os.getenv(
        "DITTO_CHECKPOINT_PATH",
        "checkpoints/ditto_pytorch"  # PyTorch models
    )
    ditto_device_id: int = 0
    ditto_fps: int = 25
    ditto_use_realtime: bool = os.getenv("DITTO_REALTIME", "true").lower() == "true"
    
    # TensorRT Settings
    tensorrt_bin_path: str = os.getenv(
        "TENSORRT_BIN_PATH",
        r"F:\nvidia\TensorRT-10.15.1.29\bin"
    )
    
    # Server Settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8765"))
    
    # System Prompt
    system_prompt: str = """You are a friendly AI assistant with a visual avatar. 
    Keep responses concise and conversational. You can see the user and they can see you."""


config = Config()
