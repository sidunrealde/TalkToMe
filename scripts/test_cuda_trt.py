from cuda import cuda, cudart
print("cuda-python OK")
import tensorrt as trt
print(f"tensorrt {trt.__version__} OK")

# Quick test: create a logger and init plugins
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, "")
print("TRT plugins initialized OK")
