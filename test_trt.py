#!/usr/bin/env python
import tensorrt as trt
print(f"TensorRT version: {trt.__version__}")
print(f"TensorRT Python version: {trt.__version__}")

# Test plugin loading
logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(logger, "")
print("TensorRT plugins initialized successfully")
