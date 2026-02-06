import asyncio
import os
import base64
import json
import logging
from typing import Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.audio.vad.silero import SileroVADAnalyzer

from .config import config
from .pipeline import create_pipeline
from .services.avatar_manager import AvatarManager

# Add TensorRT to PATH if configured
if config.tensorrt_bin_path and os.path.exists(config.tensorrt_bin_path):
    os.environ["PATH"] = config.tensorrt_bin_path + os.pathsep + os.environ.get("PATH", "")
    logging.info(f"Added TensorRT to PATH: {config.tensorrt_bin_path}")

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Store connections by pc_id
pcs_map: Dict[str, SmallWebRTCConnection] = {}
pipeline_tasks: Dict[str, "PipelineTask"] = {}

avatar_manager = AvatarManager()

ice_servers = [
    IceServer(urls="stun:stun.l.google.com:19302")
]


async def run_pipeline(webrtc_connection: SmallWebRTCConnection):
    """Run the Pipecat pipeline with the WebRTC connection."""
    logger.info(f"Starting pipeline for connection {webrtc_connection.pc_id}")
    
    try:
        # Import VADParams
        from pipecat.audio.vad.vad_analyzer import VADParams
        
        # Configure VAD with higher thresholds to ignore background noise
        vad = SileroVADAnalyzer(
            params=VADParams(
                confidence=0.8,      # Higher = more confident speech detection (default 0.7)
                start_secs=0.3,      # Longer speech needed to trigger (default 0.2)
                stop_secs=1.0,       # Wait longer for silence before ending (default 0.8)
                min_volume=0.7,      # Higher = ignore quieter sounds (default 0.6)
            )
        )
        
        transport = SmallWebRTCTransport(
            webrtc_connection=webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                video_out_enabled=True,  # Enable video output for avatar
                vad_enabled=True,
                vad_analyzer=vad,
            ),
        )
        
        logger.debug("Transport created, creating pipeline...")
        task = await create_pipeline(
            transport, 
            avatar_manager, 
            webrtc_connection=webrtc_connection
        )
        
        # Store task reference for text input
        pipeline_tasks[webrtc_connection.pc_id] = task
        
        logger.debug("Pipeline created, starting runner...")
        runner = PipelineRunner()
        await runner.run(task)
        
        logger.info(f"Pipeline finished for connection {webrtc_connection.pc_id}")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
    finally:
        # Cleanup task reference
        pipeline_tasks.pop(webrtc_connection.pc_id, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting up...")
    
    # Load default avatar if available
    default_avatar_paths = [
        "avatars/default.png",
        "avatars/default.jpg",
        "ditto/example/image.png",  # Sample image from Ditto
    ]
    
    for avatar_path in default_avatar_paths:
        if os.path.exists(avatar_path):
            if avatar_manager.load_avatar_from_path(avatar_path):
                logger.info(f"Loaded default avatar from: {avatar_path}")
                break
            else:
                logger.warning(f"Failed to load avatar from: {avatar_path}")
    
    if avatar_manager.current_avatar is None:
        logger.warning("No default avatar found. Upload an avatar to enable video.")
    
    yield
    # Cleanup connections on shutdown
    logger.info("Server shutting down, cleaning up connections...")
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros, return_exceptions=True)
    pcs_map.clear()


app = FastAPI(title="AI Avatar", lifespan=lifespan)

# Serve static files
app.mount("/static", StaticFiles(directory="web"), name="static")


@app.get("/")
async def index():
    return FileResponse("web/index.html")


@app.post("/api/avatar")
async def upload_avatar(file: UploadFile = File(...)):
    """Upload a new avatar image."""
    logger.info(f"Uploading avatar: {file.filename}")
    contents = await file.read()
    b64 = base64.b64encode(contents).decode()
    
    if avatar_manager.load_avatar_from_base64(b64):
        logger.info("Avatar loaded successfully")
        return {"status": "success"}
    logger.error("Failed to load avatar")
    return {"status": "error", "message": "Failed to load avatar"}


@app.post("/api/voice")
async def set_voice(voice: str = Form(...)):
    """Set the TTS voice."""
    logger.info(f"Setting voice to: {voice}")
    avatar_manager.set_voice(voice)
    return {"status": "success", "voice": avatar_manager.current_voice}


@app.get("/api/voices")
async def get_voices():
    """Get available TTS voices."""
    return {
        "voices": ["autumn", "breeze", "ember", "juniper"],
        "current": avatar_manager.current_voice
    }


@app.post("/api/text")
async def send_text(request: dict):
    """Handle text input from user."""
    text = request.get("text", "").strip()
    pc_id = request.get("pc_id")
    
    if not text:
        return {"error": "No text provided"}
    
    if not pc_id or pc_id not in pipeline_tasks:
        return {"error": "No active connection"}
    
    logger.info(f"Received text input: {text}")
    
    # Queue the text as a user message in the pipeline
    task = pipeline_tasks[pc_id]
    
    # Import here to avoid circular imports
    from pipecat.frames.frames import TextFrame, LLMMessagesFrame
    
    # We need to inject this text into the pipeline
    # For now, we'll queue it through the task
    # This will be processed by the context aggregator
    await task.queue_frame(TextFrame(text=text))
    
    return {"status": "success"}


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    """Handle WebRTC offer from client."""
    logger.info(f"Received WebRTC offer request")
    logger.debug(f"Request: {json.dumps(request, indent=2)}")
    
    pc_id = request.get("pc_id")
    sdp = request.get("sdp")
    sdp_type = request.get("type", "offer")
    
    if not sdp:
        logger.error("Missing SDP in offer request")
        return {"error": "Missing sdp"}
    
    try:
        if pc_id and pc_id in pcs_map:
            # Reuse existing connection
            logger.info(f"Reusing existing connection for pc_id: {pc_id}")
            pipecat_connection = pcs_map[pc_id]
            await pipecat_connection.renegotiate(
                sdp=sdp,
                type=sdp_type,
                restart_pc=request.get("restart_pc", False),
            )
        else:
            # Create new connection
            logger.info("Creating new WebRTC connection")
            pipecat_connection = SmallWebRTCConnection(ice_servers=ice_servers)
            
            logger.debug("Initializing connection with SDP offer...")
            await pipecat_connection.initialize(sdp=sdp, type=sdp_type)
            
            # Setup disconnect handler
            @pipecat_connection.event_handler("closed")
            async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
                logger.info(f"Connection closed for pc_id: {webrtc_connection.pc_id}")
                pcs_map.pop(webrtc_connection.pc_id, None)
            
            # Start the pipeline in background
            logger.debug("Starting pipeline in background...")
            background_tasks.add_task(run_pipeline, pipecat_connection)
        
        # Get the answer
        answer = pipecat_connection.get_answer()
        
        if not answer:
            logger.error("No answer generated from connection")
            return {"error": "Failed to generate answer"}
        
        # Store connection
        pcs_map[answer["pc_id"]] = pipecat_connection
        
        logger.info(f"WebRTC answer generated for pc_id: {answer['pc_id']}")
        logger.debug(f"Answer type: {answer['type']}")
        
        return answer
        
    except Exception as e:
        logger.error(f"Error handling offer: {e}", exc_info=True)
        return {"error": str(e)}


def main():
    # Check if SSL certificates exist for HTTPS
    cert_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "certs", "cert.pem")
    key_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "certs", "key.pem")
    
    ssl_config = {}
    if os.path.exists(cert_file) and os.path.exists(key_file):
        ssl_config = {
            "ssl_certfile": cert_file,
            "ssl_keyfile": key_file,
        }
        logger.info(f"HTTPS enabled with certificates from certs/")
        print(f"\n{'='*60}")
        print(f"🔒 HTTPS Server running at: https://{config.host}:{config.port}")
        print(f"   Access from other devices: https://192.168.0.176:{config.port}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"⚠️  HTTP Server (no SSL) running at: http://{config.host}:{config.port}")
        print(f"   Note: Microphone won't work from other devices without HTTPS")
        print(f"{'='*60}\n")
    
    uvicorn.run(
        "src.server:app",
        host=config.host,
        port=config.port,
        reload=True,
        log_level="debug",
        **ssl_config
    )


if __name__ == "__main__":
    main()
