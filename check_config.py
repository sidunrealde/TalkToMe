#!/usr/bin/env python
import pickle
import pprint
import os

# Load hybrid config
cfg_path = os.path.expanduser("~/TalkToMe/checkpoints/ditto_cfg/v0.4_hubert_cfg_hybrid.pkl")
with open(cfg_path, "rb") as f:
    cfg = pickle.load(f)

print("=== Hybrid Config Contents ===")
pprint.pprint(cfg)
