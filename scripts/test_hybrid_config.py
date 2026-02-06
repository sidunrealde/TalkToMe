"""Test loading all models in the hybrid TRT+PyTorch config"""
import sys
import os
import time

sys.path.insert(0, os.path.expanduser('~/TalkToMe/ditto'))

from core.atomic_components.cfg import parse_cfg

cfg_pkl = os.path.expanduser('~/TalkToMe/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_hybrid_online.pkl')
data_root = os.path.expanduser('~/TalkToMe/checkpoints/ditto_trt')

print(f"Config: {cfg_pkl}")
print(f"Data root: {data_root}")
print()

t0 = time.time()
result = parse_cfg(cfg_pkl, data_root)
t1 = time.time()
print(f"Config parsed in {t1-t0:.2f}s")

[
    avatar_registrar_cfg,
    condition_handler_cfg,
    lmdm_cfg,
    stitch_network_cfg,
    warp_network_cfg,
    decoder_cfg,
    wav2feat_cfg,
    default_kwargs,
] = result

print(f"\n--- Model paths resolved ---")
# Check warp_network
print(f"warp_network: {warp_network_cfg['model_path']}")
print(f"  exists: {os.path.exists(warp_network_cfg['model_path'])}")

# Check decoder
print(f"decoder: {decoder_cfg['model_path']}")
print(f"  exists: {os.path.exists(decoder_cfg['model_path'])}")

# Check stitch
print(f"stitch: {stitch_network_cfg['model_path']}")
print(f"  exists: {os.path.exists(stitch_network_cfg['model_path'])}")

# Check lmdm
print(f"lmdm: {lmdm_cfg['model_path']}")
print(f"  exists: {os.path.exists(lmdm_cfg['model_path'])}")

# Check hubert
print(f"hubert: {wav2feat_cfg['w2f_cfg']['model_path']}")

# Check avatar registrar models
for k, v in avatar_registrar_cfg.items():
    if isinstance(v, dict) and 'model_path' in v:
        mp = v['model_path']
        ext = os.path.splitext(mp)[1]
        exists = os.path.exists(mp)
        sz = os.path.getsize(mp) if exists else 0
        print(f"  {k}: {mp} [{ext}] exists={exists} size={sz/(1024*1024):.1f}MB")
    elif isinstance(v, dict):
        for kk, vv in v.items():
            if 'path' in kk and vv:
                exists = os.path.exists(vv)
                sz = os.path.getsize(vv) if exists else 0
                print(f"  {k}.{kk}: {vv} exists={exists} size={sz/(1024*1024):.1f}MB")

print("\n--- Loading TRT engines ---")
# Try loading a TRT engine
from core.utils.load_model import load_model

models_to_test = [
    ("decoder", decoder_cfg['model_path'], decoder_cfg['device']),
    ("stitch", stitch_network_cfg['model_path'], stitch_network_cfg['device']),
    ("warp_network", warp_network_cfg['model_path'], warp_network_cfg['device']),
]

for name, path, device in models_to_test:
    t0 = time.time()
    try:
        if name == "warp_network":
            model, mtype = load_model(path, device, module_name="WarpingNetwork")
        else:
            model, mtype = load_model(path, device)
        t1 = time.time()
        print(f"  ✅ {name}: {mtype} loaded in {t1-t0:.2f}s")
    except Exception as e:
        t1 = time.time()
        print(f"  ❌ {name}: FAILED in {t1-t0:.2f}s - {e}")

print("\n✅ Hybrid config test complete!")
