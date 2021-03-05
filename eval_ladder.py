import os
import sys
import json
import importlib
import argparse
import pickle as pk
import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sch

from torch.utils.data import DataLoader

import utils.dataset as ds
import utils.loss_manager as loss_manager
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
########### Argument ###########
parser = argparse.ArgumentParser(description='  \
    --task_type    \
    --net_type    \
    --norm_type    \
    --model_path    \
')

parser.add_argument('--task_type', help='STL, MTL')
parser.add_argument('--net_type', help='Ladder')
parser.add_argument('--norm_type', type=int, help='\
    1. All dataset is normalized by their own stats. \
    2. All dataset is normalized by stats from the clean training set. \
    3. All dataset is normalized by stats from the training set matched with each domain. \
')
parser.add_argument('--model_path', help='model/mlp, model/ladder, ... etc')
parser.add_argument('--store_path', default=None)

args = parser.parse_args()
args.net_type=args.net_type.upper()

assert args.task_type in ["STL", "MTL"]
assert args.norm_type in [1, 2, 3]
if args.norm_type == 1:
    norm_description="1. All dataset is normalized by their own stats"
if args.norm_type == 2:
    norm_description="2. All dataset is normalized by stats from the clean training set."
if args.norm_type == 3:
    norm_description="3. All dataset is normalized by stats from the training set matched with each domain."
is_stl = True if args.task_type == "STL" else False
is_mtl = True if args.task_type == "MTL" else False
#######################################

################ Data Load #################
test_path = os.path.join("data", "labeled", "Test1")

feat_path = os.path.join(test_path,"feats.pk")
lab_path = os.path.join(test_path,"labs.pk")
with open(feat_path, 'rb') as f:
    test_feats = pk.load(f)
with open(lab_path, 'rb') as f:
    test_labs = pk.load(f)
############################################

################ Model Load ################
model_path = args.model_path
net = importlib.import_module("net."+args.net_type.lower())
E = getattr(net, "HLD")

E_model = dict()
if args.task_type == "STL":
    for attr in ["aro", "dom", "val"]:
        E_model[attr] = E(6373, 256, 2, 1)
        E_model[attr].load_state_dict(torch.load(os.path.join(model_path, "final_E_"+attr+".pt")))

elif args.task_type == "MTL":
    for attr in ["aro", "dom", "val"]:
        E_model[attr] = E(6373, 256, 2, 3)
        E_model[attr].load_state_dict(torch.load(os.path.join(model_path, "final_E_"+attr+".pt")))
        
############################################

################ Data preprocessing ################

if args.norm_type == 1:
    test_set = ds.HLDset(test_feats, test_labs)
else:
    with open(os.path.join(model_path, "data_stat.pk"), 'rb') as f:
        feat_mean, feat_var, lab_mean, lab_var = pk.load(f)
    test_set = ds.HLDset(test_feats, test_labs,
        feat_mean=feat_mean, feat_var=feat_var, 
        lab_mean=lab_mean, lab_var=lab_var)

model_list = [E_model]
attr_list = ["aro", "dom", "val"]

for model in model_list:
    for attr in attr_list:
        model[attr].eval()
        model[attr].cuda()
test_loader = DataLoader(test_set, batch_size=2048, shuffle=False)

with torch.no_grad():
    total_pred = [] 
    total_y = []
    for x, y in test_loader:
        x=x.float().cuda()
        y=y.float().cuda()
        pred = []
        for i, attr in enumerate(attr_list):
            clean_y, _, _ = E_model[attr](x)
            cur_pred = clean_y
                
            if is_mtl:
                pred.append(cur_pred[:, i].unsqueeze(1))
            else:
                pred.append(cur_pred)

        pred = torch.cat(pred, dim=1)

        total_pred.append(pred)
        total_y.append(y)
    total_pred = torch.cat(total_pred, 0)
    total_y = torch.cat(total_y, 0)

    ccc = loss_manager.CCC_loss(total_pred, total_y)            
    ccc = ccc.cpu().numpy()     
    aro = str(np.round(ccc[0], 4))
    dom = str(np.round(ccc[1], 4))
    val = str(np.round(ccc[2], 4))
    print("Arousal CCC:", aro)
    print("Dominance CCC:", dom)
    print("Valence CCC:", val)
    
if args.store_path is not None:
    lab_mu = test_set.lab_mean
    lab_std = np.sqrt(test_set.lab_var)
    with open(args.store_path, 'w') as f:
        for utt_id, p in zip(test_set.utt_list, total_pred):
            real_p = lab_std*p + lab_mu
            f.write(utt_id+"\t"+str(real_p[0])+"\t"+str(real_p[1])+"\t"+str(real_p[2])+"\n")
