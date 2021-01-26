import os
import sys
import pickle as pk

feat_path = sys.argv[1]
out_path = sys.argv[2]

print("Processing",feat_path)

os.makedirs(out_path, exist_ok=True)

feat_dict = dict()
for fname in os.listdir(feat_path):
    file_path = os.path.join(feat_path, fname)

    with open(file_path, 'r') as f:
        for line in f:
            continue
        feats = line.split(",")[1:-1]
    
        feat=[float(f) for f in feats]
        with open(os.path.join(out_path, fname+".pk"), 'wb') as f:
            pk.dump(feat, f)
