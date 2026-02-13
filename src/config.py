import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Personality loader ────────────────────────────────────────────────────

PERSONALITY_PATH = Path(os.getenv("PERSONALITY_PATH", "personality.yaml"))

def _load_personality(path: Path = PERSONALITY_PATH) -> dict:
    """Load the personality YAML file and return it as a dict."""
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed – falling back to default system prompt.")
        return {}

    if not path.exists():
        logger.warning(f"Personality file not found at {path} – using defaults.")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        logger.info(f"Loaded personality from {path} (name={data.get('name', 'unnamed')})")
        return data
    except Exception as e:
        logger.error(f"Failed to parse personality file: {e}")
        return {}


def _build_system_prompt(persona: dict) -> str:
    """Assemble a system prompt from the personality dict sections.
    
    The prompt is structured to front-load hard constraints (output format,
    forbidden patterns) so the model sees them first, then softer personality.
    """
    if not persona:
        return (
            "You are a friendly AI assistant. Your output will be read aloud by a "
            "text-to-speech engine. Keep responses to 1 or 2 sentences. Use only "
            "plain English with basic punctuation. Never use emojis, markdown, "
            "bullet points, or special characters."
        )

    parts: list[str] = []

    # ── Hard constraints FIRST (output format / TTS rules) ──
    parts.append(
        "IMPORTANT: Your responses will be spoken aloud by a text-to-speech engine. "
        "You MUST follow these output rules strictly:\n"
        "- Respond in 1 to 2 short sentences only.\n"
        "- Use ONLY plain English text with basic punctuation (periods, commas, "
        "question marks, exclamation marks, apostrophes).\n"
        "- NEVER output emojis, emoticons, asterisks, markdown, bullet points, "
        "numbered lists, code blocks, hashtags, or any special characters.\n"
        "- NEVER start with greetings like \"Hey there\" or filler like \"Sure,\" "
        "\"Of course,\" \"Absolutely,\" or \"I can help with that.\"\n"
        "- NEVER repeat, summarize, or reference previous messages. Each reply "
        "must answer ONLY the current message as if it is the first thing said.\n"
        "- Do not introduce yourself or describe what you can do unless explicitly asked."
    )

    # ── Identity ──
    name = persona.get("name", "Assistant")
    tagline = persona.get("tagline")
    if tagline:
        parts.append(f"Your name is {name}. {tagline}")
    else:
        parts.append(f"Your name is {name}.")

    # Personality traits
    traits = persona.get("personality")
    if traits:
        trait_list = ", ".join(traits)
        parts.append(f"Your personality: {trait_list}.")

    # Backstory
    backstory = persona.get("backstory", "").strip()
    if backstory:
        parts.append(backstory)

    # Additional instructions from YAML
    instructions = persona.get("instructions")
    if instructions:
        for instr in instructions:
            parts.append(instr)

    return "\n\n".join(parts)


# Pre-load once at import time so every module sees the same prompt
_persona = _load_personality()
_system_prompt = _build_system_prompt(_persona)


@dataclass
class Config:
    # Groq API
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    
    # Ollama LLM
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    
    # TTS Settings
    tts_model: str = "playai/playht-tts-v3"
    tts_voice: str = "autumn"  # autumn, breeze, ember, juniper
    
    # Ditto Settings - Use TensorRT for real-time performance
    # For TRT+PyTorch hybrid (recommended): "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_hybrid_online.pkl"
    #   - Uses TRT for 10 models, PyTorch for warp_network (GridSample3D not supported in TRT/ONNX)
    # For full TensorRT: "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl" (requires GridSample3D TRT plugin)
    # For PyTorch: "checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
    ditto_config_path: str = os.getenv(
        "DITTO_CONFIG_PATH",
        "checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"  # Default: PyTorch (set via .env for TRT hybrid)
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
    
    # Personality / System Prompt (loaded from personality.yaml)
    system_prompt: str = _system_prompt

    # Raw personality data for anything else that needs it (e.g. voice_style)
    personality: dict = field(default_factory=lambda: _persona)


config = Config()
