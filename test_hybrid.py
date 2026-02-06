#!/usr/bin/env python
"""Test TensorRT Hybrid Ditto initialization."""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up environment for hybrid mode
os.environ["DITTO_CONFIG_PATH"] = "checkpoints/ditto_cfg/v0.4_hubert_cfg_hybrid.pkl"
os.environ["DITTO_CHECKPOINT_PATH"] = "checkpoints/ditto_trt"

# Change to project directory
os.chdir(os.path.expanduser("~/TalkToMe"))

# Add ditto to path
sys.path.insert(0, 'ditto')

def test_import():
    """Test that we can import Ditto modules."""
    logger.info("Testing Ditto imports...")
    
    try:
        from core.atomic_components.cfg import parse_cfg, print_cfg
        logger.info("✓ cfg module imported")
    except Exception as e:
        logger.error(f"✗ Failed to import cfg: {e}")
        return False
    
    return True

def test_tensorrt():
    """Test TensorRT engine loading."""
    logger.info("Testing TensorRT engine loading...")
    
    import tensorrt as trt
    
    logger.info(f"TensorRT version: {trt.__version__}")
    
    # Test loading one of our engines
    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    
    engine_path = "checkpoints/ditto_trt/decoder_fp16.engine"
    if os.path.exists(engine_path):
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine:
                logger.info(f"✓ Successfully loaded decoder engine")
                logger.info(f"  - Num IO tensors: {engine.num_io_tensors}")
            else:
                logger.error("✗ Failed to deserialize decoder engine")
                return False
    else:
        logger.error(f"✗ Engine not found: {engine_path}")
        return False
    
    return True

def test_onnx():
    """Test ONNX warp_network loading."""
    logger.info("Testing ONNX warp_network loading...")
    
    import onnxruntime as ort
    
    logger.info(f"ONNX Runtime version: {ort.__version__}")
    
    warp_path = "checkpoints/ditto_trt/warp_network.onnx"
    if os.path.exists(warp_path):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess = ort.InferenceSession(warp_path, providers=providers)
        logger.info(f"✓ Successfully loaded warp_network.onnx")
        logger.info(f"  - Inputs: {[i.name for i in sess.get_inputs()]}")
        logger.info(f"  - Outputs: {[o.name for o in sess.get_outputs()]}")
    else:
        logger.error(f"✗ ONNX file not found: {warp_path}")
        return False
    
    return True

def test_parse_config():
    """Test parsing the hybrid config."""
    logger.info("Testing hybrid config parsing...")
    
    try:
        from core.atomic_components.cfg import parse_cfg
        
        cfg_path = "checkpoints/ditto_cfg/v0.4_hubert_cfg_hybrid.pkl"
        data_root = "checkpoints/ditto_trt"
        
        result = parse_cfg(cfg_path, data_root, {})
        logger.info(f"✓ Config parsed successfully")
        logger.info(f"  - Got {len(result)} config sections")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to parse config: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TensorRT Hybrid Mode Test")
    print("=" * 60)
    
    all_passed = True
    
    if not test_import():
        all_passed = False
    
    if not test_tensorrt():
        all_passed = False
    
    if not test_onnx():
        all_passed = False
    
    if not test_parse_config():
        all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 60)
