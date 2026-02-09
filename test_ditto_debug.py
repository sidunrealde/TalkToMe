#!/usr/bin/env python3
"""Debug script to test Ditto SDK initialization with PyTorch."""
import os
import sys

# Change to project directory
os.chdir("/home/siddarthag/TalkToMe")

print("=== Ditto SDK Debug Test (PyTorch Mode) ===\n")

# Check files
print("=== File Checks ===")
avatar_path = "avatars/current_avatar.png"
cfg_path = "checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
ckpt_path = "checkpoints/ditto_pytorch"

print(f"Avatar exists: {os.path.exists(avatar_path)}")
print(f"Config exists: {os.path.exists(cfg_path)}")
print(f"Checkpoint dir exists: {os.path.exists(ckpt_path)}")

if os.path.exists(ckpt_path):
    import glob
    models = glob.glob(os.path.join(ckpt_path, "models", "*.pth"))
    print(f"PyTorch models: {len(models)}")
    for m in models[:3]:
        print(f"  - {os.path.basename(m)}")

# Try importing ditto
print("\n=== Import Test ===")
sys.path.insert(0, "ditto")
try:
    from core.atomic_components.cfg import parse_cfg
    print("✓ Ditto cfg imports OK")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Try parsing config
print("\n=== Config Parse Test ===")
try:
    result = parse_cfg(cfg_path, ckpt_path, {})
    print(f"✓ Config parsed successfully ({len(result)} sections)")
except Exception as e:
    print(f"✗ Config parse error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try creating SDK components
print("\n=== SDK Component Test ===")
try:
    from core.atomic_components.avatar_registrar import AvatarRegistrar
    
    [
        avatar_registrar_cfg,
        condition_handler_cfg,
        lmdm_cfg,
        stitch_network_cfg,
        warp_network_cfg,
        decoder_cfg,
        wav2feat_cfg,
        default_kwargs,
    ] = parse_cfg(cfg_path, ckpt_path, {})
    
    print("Creating AvatarRegistrar...")
    avatar_registrar = AvatarRegistrar(**avatar_registrar_cfg)
    print("✓ AvatarRegistrar created")
    
    print("\n=== Avatar Registration Test ===")
    print(f"Registering avatar from: {avatar_path}")
    source_info = avatar_registrar(
        avatar_path, 
        max_dim=1920, 
        n_frames=-1, 
        crop_scale=2.3,
        crop_vx_ratio=0,
        crop_vy_ratio=-0.125,
        crop_flag_do_rot=True,
    )
    print(f"✓ Avatar registered!")
    print(f"  Frames: {len(source_info['x_s_info_lst'])}")
    print(f"  Is image: {source_info['is_image_flag']}")
    
    print("\n=== ALL TESTS PASSED ===")
    print("Ditto SDK can initialize with PyTorch mode!")
    
except Exception as e:
    print(f"✗ SDK component error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
