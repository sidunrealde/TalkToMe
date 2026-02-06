"""Quick check of all major dependencies for the server"""
import sys

deps = [
    ("pipecat", "from pipecat.transports.smallwebrtc.connection import IceServer"),
    ("fastapi", "import fastapi"),
    ("uvicorn", "import uvicorn"),
    ("numpy", "import numpy"),
    ("cv2", "import cv2"),
    ("torch", "import torch"),
    ("tensorrt", "import tensorrt as trt"),
    ("cuda-python", "from cuda import cuda, cudart"),
    ("einops", "import einops"),
    ("PIL", "from PIL import Image"),
    ("dotenv", "from dotenv import load_dotenv"),
]

all_ok = True
for name, imp in deps:
    try:
        exec(imp)
        print(f"  OK  {name}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        all_ok = False

if all_ok:
    print("\nAll dependencies OK!")
else:
    print("\nSome dependencies missing!")
