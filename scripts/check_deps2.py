try:
    import kokoro_onnx
    print("kokoro_onnx OK")
except ImportError as e:
    print(f"kokoro_onnx MISSING: {e}")

try:
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    print("silero VAD OK")
except ImportError as e:
    print(f"silero VAD MISSING: {e}")

try:
    import edge_tts
    print("edge_tts OK")
except ImportError as e:
    print(f"edge_tts MISSING: {e}")
