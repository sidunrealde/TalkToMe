import pickle
import sys

cfg = pickle.load(open(sys.argv[1], 'rb'))
print('online_mode:', cfg['default_kwargs'].get('online_mode', 'NOT SET'))
