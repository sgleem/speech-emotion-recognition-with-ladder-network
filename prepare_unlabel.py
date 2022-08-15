import os
import sys
import json
import pickle as pk
from tqdm import tqdm 
data_dir = sys.argv[1]
out_dir = sys.argv[2]

total_feat_dict={
    "Train": dict(),
}

for txt_id in tqdm(os.listdir(data_dir)):
    with open(data_dir+"/"+txt_id, 'rb') as f:
        feat = pk.load(f)
    utt_id = txt_id.split(".")[0]+".wav"
    dtype = "Train"
    total_feat_dict[dtype][utt_id]=feat

for dtype, cur_feats in total_feat_dict.items():
    os.makedirs(out_dir+"/"+dtype, exist_ok=True)
    
    with open(out_dir+"/"+dtype+"/feats.pk", 'wb') as f:
        pk.dump(cur_feats, f)



