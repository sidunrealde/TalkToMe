"""
Convert Ditto ONNX models to TensorRT engines.

Usage:
    python scripts/convert_to_tensorrt.py

Requires TensorRT bin folder in PATH:
    $env:PATH = "F:\nvidia\TensorRT-10.15.1.29\bin;" + $env:PATH
"""

import os
import sys
import subprocess

# Paths
ONNX_DIR = "checkpoints/ditto_onnx"
TRT_DIR = "checkpoints/ditto_trt"
TRTEXEC = r"F:\nvidia\TensorRT-10.15.1.29\bin\trtexec.exe"

# Models to convert (model_name, fp16, dynamic_shapes)
# dynamic_shapes format: {"input_name": ("min_shape", "opt_shape", "max_shape")}
MODELS = [
    # Core inference models - convert with FP16 for speed
    ("appearance_extractor.onnx", True, None),
    ("motion_extractor.onnx", True, None),
    ("stitch_network.onnx", True, None),
    # Use warp_network_ori instead - doesn't require GridSample3D custom plugin
    ("warp_network_ori.onnx", True, None),
    ("decoder.onnx", True, None),
    
    # Audio models - hubert needs dynamic shapes for variable-length audio
    ("lmdm_v0.4_hubert.onnx", True, None),
    ("hubert.onnx", False, {"input_values": ("1x3200", "1x16000", "1x48000")}),
    
    # Face detection models
    ("insightface_det.onnx", True, None),
    ("landmark106.onnx", True, None),
    ("landmark203.onnx", True, None),
]

def convert_onnx_to_trt(onnx_path: str, trt_path: str, fp16: bool = True, dynamic_shapes: dict = None):
    """Convert ONNX model to TensorRT engine."""
    
    if os.path.exists(trt_path):
        print(f"  Skipping {os.path.basename(trt_path)} - already exists")
        return True
    
    cmd = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_path}",
        "--skipInference",
    ]
    
    if fp16:
        cmd.append("--fp16")
    
    # Add dynamic shapes if specified
    if dynamic_shapes:
        for name, shapes in dynamic_shapes.items():
            min_shape, opt_shape, max_shape = shapes
            cmd.append(f"--minShapes={name}:{min_shape}")
            cmd.append(f"--optShapes={name}:{opt_shape}")
            cmd.append(f"--maxShapes={name}:{max_shape}")
    
    print(f"  Converting: {os.path.basename(onnx_path)}")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            return False
        print(f"  Success: {os.path.basename(trt_path)}")
        return True
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Conversion timed out")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def main():
    # Create output directory
    os.makedirs(TRT_DIR, exist_ok=True)
    
    # Check trtexec exists
    if not os.path.exists(TRTEXEC):
        print(f"ERROR: trtexec not found at {TRTEXEC}")
        print("Please update TRTEXEC path in this script.")
        sys.exit(1)
    
    print(f"Converting ONNX models to TensorRT engines")
    print(f"ONNX dir: {ONNX_DIR}")
    print(f"TRT dir: {TRT_DIR}")
    print(f"trtexec: {TRTEXEC}")
    print()
    
    success = 0
    failed = 0
    
    for model_name, fp16, dynamic_shapes in MODELS:
        onnx_path = os.path.join(ONNX_DIR, model_name)
        trt_name = model_name.replace(".onnx", ".engine")
        trt_path = os.path.join(TRT_DIR, trt_name)
        
        if not os.path.exists(onnx_path):
            print(f"  Skipping {model_name} - ONNX file not found")
            continue
        
        if convert_onnx_to_trt(onnx_path, trt_path, fp16, dynamic_shapes):
            success += 1
        else:
            failed += 1
    
    print()
    print(f"Conversion complete: {success} success, {failed} failed")
    
    if success > 0:
        print()
        print("To use TensorRT models, update config to use:")
        print(f"  ditto_config_path: checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl")
        print(f"  ditto_checkpoint_path: {TRT_DIR}")

if __name__ == "__main__":
    main()
