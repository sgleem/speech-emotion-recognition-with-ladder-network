import os
import sys
import json
from tqdm import tqdm
import pickle as pk

data_dir = sys.argv[1]
out_dir = sys.argv[2]
label_path = sys.argv[3]


with open(label_path, 'r') as f:
    label_dict = json.load(f)

total_feat_dict={
    "Train": dict(),
    "Validation": dict(),
    "Test1": dict(),
    "Test2": dict()
}
total_label_dict={
    "Train": dict(),
    "Validation": dict(),
    "Test1": dict(),
    "Test2": dict()
}

for txt_id in tqdm(os.listdir(data_dir)):
    with open(data_dir+"/"+txt_id, 'rb') as f:
        feat = pk.load(f)
    utt_id = txt_id.split(".")[0]+".wav"
    label = label_dict[utt_id]
    dtype = label["Split_Set"]
    total_feat_dict[dtype][utt_id]=feat
    total_label_dict[dtype][utt_id]=[label["EmoAct"], label["EmoDom"], label["EmoVal"]]

for dtype, cur_feats in total_feat_dict.items():
    os.makedirs(out_dir+"/"+dtype, exist_ok=True)
    cur_labels = total_label_dict[dtype]

    with open(out_dir+"/"+dtype+"/feats.pk", 'wb') as f:
        pk.dump(cur_feats, f)
    with open(out_dir+"/"+dtype+"/labs.pk", 'wb') as f:
        pk.dump(cur_labels, f)



