"""
Create a hybrid config: TRT for all models except warp_network which uses PyTorch.
This avoids needing the GridSample3D TensorRT plugin.
"""
import pickle
import copy
import sys
import os

# Load the full TRT config as base
trt_cfg_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'ditto_cfg', 'v0.4_hubert_cfg_trt.pkl')
trt_cfg_path = os.path.abspath(trt_cfg_path)

with open(trt_cfg_path, 'rb') as f:
    cfg = pickle.load(f)

# Also load the TRT online config to get the online_mode flag
trt_online_path = trt_cfg_path.replace('_trt.pkl', '_trt_online.pkl')
if os.path.exists(trt_online_path):
    with open(trt_online_path, 'rb') as f:
        online_cfg = pickle.load(f)
    has_online = True
else:
    has_online = False

# Swap warp_network to use PyTorch .pth file
# The .pth file will be symlinked into the TRT checkpoint dir
cfg['base_cfg']['warp_network_cfg']['model_path'] = 'warp_network.pth'

# Save as new hybrid config
out_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'ditto_cfg', 'v0.4_hubert_cfg_trt_hybrid.pkl')
out_path = os.path.abspath(out_path)

with open(out_path, 'wb') as f:
    pickle.dump(cfg, f)

print(f"Created hybrid config (offline): {out_path}")

# Also create online version
if has_online:
    online_hybrid = copy.deepcopy(online_cfg)
    online_hybrid['base_cfg']['warp_network_cfg']['model_path'] = 'warp_network.pth'
    
    out_online_path = out_path.replace('_trt_hybrid.pkl', '_trt_hybrid_online.pkl')
    with open(out_online_path, 'wb') as f:
        pickle.dump(online_hybrid, f)
    print(f"Created hybrid config (online): {out_online_path}")

# Verify
print("\n--- Verification ---")
with open(out_path, 'rb') as f:
    verify = pickle.load(f)

for k, v in verify['base_cfg'].items():
    if isinstance(v, dict) and 'model_path' in v:
        mp = v['model_path']
        backend = 'TRT' if mp.endswith('.engine') else 'PyTorch' if mp.endswith('.pth') else 'ONNX'
        print(f"  {k}: {mp} [{backend}]")
    elif isinstance(v, dict):
        for kk, vv in v.items():
            if 'model_path' in kk or 'path' in kk:
                backend = 'TRT' if str(vv).endswith('.engine') else 'PyTorch' if str(vv).endswith('.pth') else 'Other'
                print(f"  {k}.{kk}: {vv} [{backend}]")

print(f"\n  audio2motion: {verify['audio2motion_cfg']['model_path']}")
print(f"  online_mode: {verify['default_kwargs'].get('online_mode', 'N/A')}")
