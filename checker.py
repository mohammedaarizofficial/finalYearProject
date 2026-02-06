# inspect_cached_subgraphs.py
import glob, os, pickle
p = 'synthetic_criminal_network/seal_checkpoints'
files = sorted(glob.glob(os.path.join(p, "*_subgraphs.pkl")))
print("Found cached files:", files)
if len(files) == 0:
    print("No cached subgraphs found.")
else:
    # load first file and inspect shapes
    with open(files[0], 'rb') as f:
        data = pickle.load(f)   # list of [pos, neg] pairs
    pair = data[0]
    pos, neg = pair
    print("pos.x.shape:", getattr(pos, "x", None).shape)
    print("neg.x.shape:", getattr(neg, "x", None).shape)
