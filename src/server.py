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

# ---------------------------------------------------------------------------
# WebRTC ICE fix for WSL2
#
# aioice discovers ALL host IPs inside WSL, including internal ones
# (e.g. 10.255.255.254 – the DNS proxy) that are unreachable from the
# Windows browser.  We filter those out.
#
# - **Mirrored networking:** WSL shares the LAN IP with Windows.
#   We just remove known-bad IPs; the shared LAN IP works directly.
# - **NAT mode:** WSL gets a 172.x.x.x address unreachable from the
#   host.  We replace it with the Windows gateway IP.
# ---------------------------------------------------------------------------

_wsl_detected = False
_wsl_networking_mode = None  # 'mirrored' or 'nat'


def _patch_ice_for_wsl():
    """Detect WSL2, log diagnostics, and filter aioice host addresses."""
    global _wsl_detected, _wsl_networking_mode

    try:
        with open("/proc/version", "r") as f:
            if "microsoft" not in f.read().lower():
                print("[ICE] Not running in WSL — no ICE patches needed")
                return
    except Exception:
        return  # Not Linux

    _wsl_detected = True
    import subprocess, socket

    gw = my_ip = None
    all_ips: list[str] = []

    try:
        gw = subprocess.check_output(
            ["ip", "route", "show", "default"], text=True, timeout=3
        ).split()[2]
    except Exception as e:
        print(f"[ICE] Could not get default gateway: {e}")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        my_ip = s.getsockname()[0]
        s.close()
    except Exception as e:
        print(f"[ICE] Could not get outbound IP: {e}")

    try:
        all_ips = subprocess.check_output(
            ["hostname", "-I"], text=True, timeout=3
        ).strip().split()
    except Exception:
        pass

    is_nat = bool(my_ip and my_ip.startswith("172."))
    _wsl_networking_mode = "nat" if is_nat else "mirrored"

    print(f"[ICE] ═══════════════════════════════════════════════════")
    print(f"[ICE] WSL2 detected — networking mode: {_wsl_networking_mode}")
    print(f"[ICE]   Outbound IP : {my_ip}")
    print(f"[ICE]   Gateway     : {gw}")
    print(f"[ICE]   All IPs     : {all_ips}")

    try:
        import aioice.ice

        orig_addrs = aioice.ice.get_host_addresses(use_ipv4=True, use_ipv6=False)
        print(f"[ICE]   aioice original host addresses: {orig_addrs}")

        if is_nat and gw:
            # NAT mode: WSL's 172.x.x.x is unreachable → use Windows gateway
            patched = [gw]
            print(f"[ICE]   NAT mode → replacing with gateway: {gw}")
        else:
            # Mirrored mode: keep the real LAN IPs, drop known-bad ones
            bad_ips = {"10.255.255.254"}  # WSL DNS proxy — unreachable from browser
            bad_prefixes = ("172.17.", "172.18.", "172.19.")  # Docker bridges
            patched = [
                ip for ip in orig_addrs
                if ip not in bad_ips and not ip.startswith(bad_prefixes)
            ]
            if not patched:
                patched = [my_ip] if my_ip else orig_addrs
                print(f"[ICE]   Mirrored: all IPs filtered — falling back to {patched}")
            else:
                print(f"[ICE]   Mirrored mode → filtered to: {patched}")

        def _patched_get_host_addresses(use_ipv4=True, use_ipv6=True):
            return patched if use_ipv4 else []

        aioice.ice.get_host_addresses = _patched_get_host_addresses
        print(f"[ICE]   ✓ Patched aioice host addresses → {patched}")
    except ImportError:
        print("[ICE]   ✗ aioice not installed — cannot patch")

    # Quick UDP reachability test to gateway
    if gw:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(1)
            s.sendto(b"ping", (gw, 19999))
            print(f"[ICE]   UDP send to gateway {gw}:19999 — OK (no error)")
            s.close()
        except Exception as e:
            print(f"[ICE]   UDP send to gateway {gw}:19999 — FAILED: {e}")

    print(f"[ICE] ═══════════════════════════════════════════════════")


_patch_ice_for_wsl()

# ---------------------------------------------------------------------------
# TURN relay constants (must match turn/turnserver.conf from setup_wsl.sh)
# ---------------------------------------------------------------------------
TURN_HOST = "127.0.0.1"
TURN_PORT = 3479
TURN_USER = "talktome"
TURN_PASS = "talktome123"


def _ensure_coturn_running():
    """Auto-start coturn if WSL detected and it isn't already running."""
    if not _wsl_detected:
        return

    import subprocess, shutil

    # Check if coturn is already listening on our port
    try:
        result = subprocess.run(
            ["ss", "-tlnH", f"sport = :{TURN_PORT}"],
            capture_output=True, text=True, timeout=3
        )
        if f":{TURN_PORT}" in result.stdout:
            print(f"[TURN] coturn already running on {TURN_HOST}:{TURN_PORT}")
            return
    except Exception:
        pass

    # Find the config file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    conf_path = os.path.join(project_root, "turn", "turnserver.conf")
    if not os.path.exists(conf_path):
        print(f"[TURN] Config not found at {conf_path} — run setup_wsl.sh first")
        print(f"[TURN] WebRTC may not connect without TURN relay in WSL2")
        return

    # Check if turnserver binary exists
    if not shutil.which("turnserver"):
        print(f"[TURN] 'turnserver' not found — install with: sudo apt-get install coturn")
        return

    # Start coturn in background
    print(f"[TURN] Starting coturn on {TURN_HOST}:{TURN_PORT}...")
    try:
        proc = subprocess.Popen(
            ["turnserver", "-c", conf_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import time; time.sleep(1)
        if proc.poll() is None:
            print(f"[TURN] ✓ coturn started (pid {proc.pid})")
        else:
            print(f"[TURN] ✗ coturn exited immediately — check /tmp/coturn.log")
    except Exception as e:
        print(f"[TURN] ✗ Failed to start coturn: {e}")


_ensure_coturn_running()


def _log_sdp_candidates(label: str, sdp: str):
    """Extract and log ICE candidate lines from an SDP for debugging."""
    candidates = [l for l in sdp.split("\r\n") if l.startswith("a=candidate:")]
    c_line = [l for l in sdp.split("\r\n") if l.startswith("c=IN IP4")]
    logger.warning(f"═══ SDP {label} — {len(candidates)} ICE candidates ═══")
    for c in c_line:
        logger.warning(f"  {c}")
    for c in candidates:
        parts = c.split()
        if len(parts) >= 8:
            addr, port, proto, ctype = parts[4], parts[5], parts[2], parts[7]
            prio = parts[3]
            logger.warning(f"  {proto.upper():4s} {addr}:{port}  type={ctype}  prio={prio}")
        else:
            logger.warning(f"  {c}")
    if not candidates:
        logger.warning("  (no candidates in SDP!)")


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
tts_services: Dict[str, "KokoroTTSService"] = {}  # Store TTS services by pc_id

avatar_manager = AvatarManager()

# ICE servers configuration
# In WSL2 mirrored mode, direct UDP between Windows and WSL is broken
# (WSL→192.168.0.176 loops back inside WSL, never reaching Windows).
# We use a coturn TURN relay on 127.0.0.1 (shared loopback) to bridge.
if _wsl_detected:
    ice_servers = [
        IceServer(
            urls=f"turn:{TURN_HOST}:{TURN_PORT}?transport=tcp",
            username=TURN_USER,
            credential=TURN_PASS,
        )
    ]
    logger.info(f"WSL mode: using TURN relay at {TURN_HOST}:{TURN_PORT} (TCP)")
else:
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
        task, tts = await create_pipeline(
            transport, 
            avatar_manager, 
            webrtc_connection=webrtc_connection
        )
        
        # Store task and TTS service references for voice changes
        pipeline_tasks[webrtc_connection.pc_id] = task
        tts_services[webrtc_connection.pc_id] = tts
        
        logger.debug("Pipeline created, starting runner...")
        runner = PipelineRunner()
        await runner.run(task)
        
        logger.info(f"Pipeline finished for connection {webrtc_connection.pc_id}")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
    finally:
        # Cleanup task and TTS references
        pipeline_tasks.pop(webrtc_connection.pc_id, None)
        tts_services.pop(webrtc_connection.pc_id, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting up...")
    
    # Load default avatar if available
    default_avatar_paths = [
        "avatars/current_avatar.png",  # Previously uploaded avatar
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


# Disable caching for static files during development
# (Ensures browser always gets latest JS/CSS after code changes)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheMiddleware)

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
    logger.info(f"=== VOICE CHANGE REQUEST ===")
    logger.info(f"Received voice: {voice}")
    logger.info(f"Active TTS services: {list(tts_services.keys())}")
    
    avatar_manager.set_voice(voice)
    logger.info(f"Avatar manager voice updated to: {avatar_manager.current_voice}")
    
    # Update all active TTS services
    for pc_id, tts in tts_services.items():
        logger.info(f"Updating TTS voice for connection {pc_id}: {tts._voice} -> {voice}")
        tts.set_voice(voice)
        logger.info(f"TTS voice after update: {tts._voice}")
    
    return {"status": "success", "voice": avatar_manager.current_voice}


@app.get("/api/voices")
async def get_voices():
    """Get available TTS voices."""
    from .services.kokoro_tts import VOICE_MAP
    return {
        "voices": list(VOICE_MAP.keys()),
        "current": avatar_manager.current_voice
    }


@app.get("/api/ice-config")
async def get_ice_config():
    """Return ICE server configuration for the browser.

    In WSL2 mirrored mode, direct UDP is broken between Windows and WSL.
    We provide TURN relay credentials so the browser forces relay transport.
    """
    if _wsl_detected:
        return {
            "iceServers": [
                {
                    "urls": [
                        f"turn:{TURN_HOST}:{TURN_PORT}?transport=tcp",
                        f"turn:{TURN_HOST}:{TURN_PORT}",
                    ],
                    "username": TURN_USER,
                    "credential": TURN_PASS,
                }
            ],
            "iceTransportPolicy": "relay",
        }
    else:
        return {
            "iceServers": [{"urls": "stun:stun.l.google.com:19302"}],
            "iceTransportPolicy": "all",
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

    # Log browser's offer candidates for ICE debugging
    if sdp:
        _log_sdp_candidates("OFFER (browser → server)", sdp)

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

        # Log server's answer candidates for ICE debugging
        if answer and "sdp" in answer:
            _log_sdp_candidates("ANSWER (server → browser)", answer["sdp"])

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


def get_local_ip():
    """Get the LAN IP address.
    
    On WSL2, the outbound IP is the virtual NAT address (172.x.x.x), not the
    Windows host's real LAN IP.  We detect WSL and return the Windows gateway
    instead, since that is what other LAN devices need (after port-forwarding).
    """
    import socket
    
    # Detect WSL2 and return the Windows host gateway IP
    try:
        with open("/proc/version", "r") as f:
            if "microsoft" in f.read().lower():
                # WSL2 — the default gateway IS the Windows host
                import subprocess
                gw = subprocess.check_output(
                    ["ip", "route", "show", "default"],
                    text=True, timeout=3
                ).split()[2]
                return gw
    except Exception:
        pass
    
    # Regular OS — use outbound socket trick
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    # Check if SSL certificates exist for HTTPS
    cert_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "certs", "cert.pem")
    key_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "certs", "key.pem")
    
    local_ip = get_local_ip()
    
    ssl_config = {}
    if os.path.exists(cert_file) and os.path.exists(key_file):
        ssl_config = {
            "ssl_certfile": cert_file,
            "ssl_keyfile": key_file,
        }
        logger.info(f"HTTPS enabled with certificates from certs/")
        print(f"\n{'='*60}")
        print(f"🔒 HTTPS Server running at:")
        print(f"   Local:   https://localhost:{config.port}")
        print(f"   Network: https://{local_ip}:{config.port}")
        print(f"")
        print(f"   ℹ️  Other devices: open the Network URL and")
        print(f"      accept the self-signed certificate warning.")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"⚠️  HTTP Server (no SSL) running at:")
        print(f"   Local:   http://localhost:{config.port}")
        print(f"   Network: http://{local_ip}:{config.port}")
        print(f"")
        print(f"   ⚠️  Microphone won't work from other devices without HTTPS!")
        print(f"   Run: python -m src.generate_certs to create SSL certs.")
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
