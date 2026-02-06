import pickle
import sys

cfg_path = sys.argv[1]
cfg = pickle.load(open(cfg_path, 'rb'))
for k, v in cfg.items():
    if isinstance(v, dict):
        print(f'--- {k} ---')
        for kk, vv in v.items():
            print(f'  {kk}: {vv}')
    else:
        print(f'--- {k} ---')
        print(f'  {v}')
