import tarfile
import sys

path = sys.argv[1]
try:
    t = tarfile.open(path, "r")
    names = t.getnames()
    print(f"Total entries: {len(names)}")
    for n in names[:20]:
        print(f"  {n}")
except Exception as e:
    print(f"Error: {e}")

# Also try torch.load directly
import torch
try:
    data = torch.load(path, map_location='cpu', weights_only=False)
    print(f"\nLoaded successfully! Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())[:10]}")
except Exception as e:
    print(f"\ntorch.load error: {e}")
