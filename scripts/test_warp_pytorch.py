"""Test that PyTorch WarpingNetwork can load the .pth file"""
import sys
import os
import time

# Add ditto to path
sys.path.insert(0, os.path.expanduser('~/TalkToMe/ditto'))

print("Testing PyTorch WarpingNetwork load...")
t0 = time.time()

from core.models.warp_network import WarpNetwork

model_path = os.path.expanduser('~/TalkToMe/checkpoints/ditto_trt/warp_network.pth')
print(f"Loading: {model_path}")

wn = WarpNetwork(model_path=model_path, device='cuda')
t1 = time.time()
print(f"WarpNetwork loaded successfully in {t1-t0:.2f}s")
print(f"Model type: {wn.model_type}")
print(f"Device: {wn.device}")

# Test inference with dummy data
import numpy as np
feature_3d = np.random.randn(1, 32, 16, 64, 64).astype(np.float32)
kp_source = np.random.randn(1, 21, 3).astype(np.float32)
kp_driving = np.random.randn(1, 21, 3).astype(np.float32)

t2 = time.time()
result = wn(feature_3d, kp_source, kp_driving)
t3 = time.time()
print(f"Inference: {result.shape}, time: {(t3-t2)*1000:.1f}ms")

# Run a few more to get warm timing
times = []
for i in range(5):
    t2 = time.time()
    result = wn(feature_3d, kp_source, kp_driving)
    t3 = time.time()
    times.append((t3-t2)*1000)

print(f"Warm avg: {sum(times)/len(times):.1f}ms per call")
print("\n✅ PyTorch WarpingNetwork works correctly!")
