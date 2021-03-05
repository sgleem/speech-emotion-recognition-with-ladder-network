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

import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

########### Argument ###########
parser = argparse.ArgumentParser(description='  \
    --task_type    \
    --net_type    \
    --norm_type    \
    --use_unlabel    \
    --model_path    \
    --seed    \
    --batch_size    \
    --learning_rate    \
')

parser.add_argument('--task_type', help='STL, MTL')
parser.add_argument('--net_type', help='Ladder')
parser.add_argument('--norm_type', type=int, help='\
    1. All dataset is normalized by their own stats. \
    2. All dataset is normalized by stats from the clean training set. \
    3. All dataset is normalized by stats from the training set matched with each domain. \
')
parser.add_argument('--use_unlabel', default=False, action='store_true')
parser.add_argument('--model_path', help='model/mlp, model/ladder, ... etc')
parser.add_argument('--seed', type=int)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--learning_rate', type=float, default=0.00005)

args = parser.parse_args()
args.net_type=args.net_type.upper()

assert args.task_type in ["STL", "MTL"]
assert args.net_type in ["LADDER"]
assert args.norm_type in [1, 2, 3]
if args.norm_type == 1:
    norm_description="1. All dataset is normalized by their own stats"
if args.norm_type == 2:
    norm_description="2. All dataset is normalized by stats from the clean training set."
if args.norm_type == 3:
    norm_description="3. All dataset is normalized by stats from the training set matched with each domain."

is_stl = True if args.task_type == "STL" else False
is_mtl = True if args.task_type == "MTL" else False
is_un = args.use_unlabel
#######################################

torch.set_deterministic(True)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

start_time = time.time()
################ Data Load #################
feat_name="HLD"
data_root = "data"
train_path = os.path.join(data_root, "labeled", "Train")
dev_path = os.path.join(data_root, "labeled", "Validation")

train_feat_path = os.path.join(train_path,"feats.pk")
train_lab_path = os.path.join(train_path,"labs.pk")
with open(train_feat_path, 'rb') as f:
    train_feats = pk.load(f)
with open(train_lab_path, 'rb') as f:
    train_labs = pk.load(f)

dev_feat_path = os.path.join(dev_path,"feats.pk")
dev_lab_path = os.path.join(dev_path,"labs.pk")
with open(dev_feat_path, 'rb') as f:
    dev_feats = pk.load(f)
with open(dev_lab_path, 'rb') as f:
    dev_labs = pk.load(f)

if is_un:
    un_path = os.path.join(data_root, "unlabeled", "Train")
    un_feat_path = os.path.join(un_path,"feats.pk")
    with open(un_feat_path, 'rb') as f:
        un_feats = pk.load(f)
    
############################################
end_time = time.time()
print("Data loading time elapsed:", end_time-start_time)

################ Model Load ################
model_path = args.model_path
os.makedirs(os.path.join(model_path, "param"), exist_ok=True)
net = importlib.import_module("net."+args.net_type.lower())
E = getattr(net, "HLD")
D = getattr(net, "HLD_Decoder")

opt = getattr(optim, "Adam")
lr=args.learning_rate

E_model = dict(); D_model = dict()
E_opt = dict(); D_opt = dict()


attr_list = ["aro", "dom", "val"]

if is_stl:
    for attr in attr_list:
        E_model[attr] = E(6373, 256, 2, 1)
        D_model[attr] = D(6373, 256, 2, 1)

        E_opt[attr] = opt(E_model[attr].parameters(), lr=lr)
        D_opt[attr] = opt(D_model[attr].parameters(), lr=lr)
        
elif is_mtl:
    coef={
        "aro": [0.7, 0.0, 0.3],
        "dom": [0.0, 0.8, 0.2],
        "val": [0.1, 0.1, 0.8]
    }
    for attr in attr_list:
        E_model[attr] = E(6373, 256, 2, 3)
        D_model[attr] = D(6373, 256, 2, 3)
        
        E_opt[attr] = opt(E_model[attr].parameters(), lr=lr)
        D_opt[attr] = opt(D_model[attr].parameters(), lr=lr)

        
with open(model_path+"/str.txt", 'w') as f:
    f.write("model_type: "+args.net_type+"\n")
    f.write("task_type: "+args.task_type+"\n")
    f.write("unlabel_used: "+args.use_unlabel+"\n")
    f.write("norm_type: "+norm_description+"\n")
############################################

start_time = time.time()
################ Data preprocessing ################
train_set = ds.HLDset(train_feats, train_labs)
if args.norm_type == 1:
    dev_set = ds.HLDset(dev_feats, dev_labs)
    if is_un:
        un_set = ds.HLDset(un_feats, None)
else:
    dev_set = ds.HLDset(dev_feats, dev_labs,
        feat_mean=train_set.feat_mean, feat_var=train_set.feat_var, 
        lab_mean=train_set.lab_mean, lab_var=train_set.lab_var)
    if is_un:
        un_set = ds.HLDset(un_feats, None, feat_mean=train_set.feat_mean, feat_var=train_set.feat_var)

if args.norm_type == 2:
    with open(os.path.join(model_path, "data_stat.pk"), 'wb') as f:
        stat = (train_set.feat_mean, train_set.feat_var, train_set.lab_mean, train_set.lab_var)
        pk.dump(stat, f)
end_time = time.time()
print("Data processing time elapsed:", end_time-start_time)

epochs = 100
lm = loss_manager.LogManager()
lm.alloc_stat_type_list(["train_aro", "train_dom", "train_val", "train_recon",
    "dev_aro", "dev_dom", "dev_val", "dev_all"])
if is_un:
    lm.alloc_stat_type("unlabel_recon")

model_list = [E_model, D_model]
opt_list = [E_opt, D_opt]


for model in model_list:
    for attr in attr_list:
        model[attr].cuda()

min_epoch = dict()
min_loss = dict()
for attr in attr_list:
    min_epoch[attr] = 0
    min_loss[attr] = 99999999999


train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
dev_loader = DataLoader(dev_set, batch_size=2048, shuffle=False)

if is_un:
    un_loader = DataLoader(un_set, batch_size=args.batch_size, shuffle=True)

for epoch in range(epochs):
    print("Epoch:",epoch)
    start_time = time.time()
    lm.init_stat()
    

    for model in model_list:
        for attr in attr_list:
            model[attr].train()
    
    for x, y in train_loader:
        x=x.float().cuda()
        y=y.float().cuda()

        pred = []
        recon_loss = torch.zeros(1).float().cuda()
        for i, attr in enumerate(attr_list):
            noise_y, noise_h, noise_list = E_model[attr](x, noisy=True)
            clean_y, clean_h, clean_list, clean_stats = E_model[attr](x, need_stat=True)

            recon_list = D_model[attr](noise_h, noise_list, clean_stats)
            cur_recon = loss_manager.ladder_loss(recon_list, clean_list)
            recon_loss += cur_recon
            
            cur_pred = noise_y
            
            pred.append(cur_pred)

        recon_loss = recon_loss.squeeze(0)
        total_loss = 1.0 * recon_loss

        ccc_stat={"aro":None,"dom":None,"val":None}
        if is_stl:
            pred = torch.cat(pred, dim=1)
            ccc = loss_manager.CCC_loss(pred, y)
            loss = 1.0-ccc
            total_loss += loss[0] + loss[1] + loss[2]

            ccc_stat["aro"] = ccc[0]
            ccc_stat["dom"] = ccc[1]
            ccc_stat["val"] = ccc[2]
        else:
            for i, attr in enumerate(attr_list):
                ccc = loss_manager.CCC_loss(pred[i], y)
                loss = 1.0-ccc
                total_loss += coef[attr][0] * loss[0] + coef[attr][1] * loss[1] + coef[attr][2] * loss[2]
                ccc_stat[attr] = ccc[i]

        for opt in opt_list:
            for attr in attr_list:
                opt[attr].zero_grad()
        total_loss.backward()
        for opt in opt_list:
            for attr in attr_list:
                opt[attr].step()

        lm.add_torch_stat("train_aro", ccc_stat["aro"])
        lm.add_torch_stat("train_dom", ccc_stat["dom"])
        lm.add_torch_stat("train_val", ccc_stat["val"])
        lm.add_torch_stat("train_recon", recon_loss)

        if is_un:
            try:
                ux = next(iter(un_loader))
            except:
                un_loader = DataLoader(un_set, batch_size=args.batch_size, shuffle=True)
                ux = next(iter(un_loader))
            ux=ux.float().cuda()
            recon_loss = torch.zeros(1).float().cuda()
            for attr in attr_list:
                noise_y, noise_list = E_model[attr](ux, noisy=True)
                clean_y, clean_list, clean_stats = E_model[attr](ux, need_stat=True)
                recon_list = D_model[attr](noise_y, noise_list, clean_stats)
                cur_recon = loss_manager.ladder_loss(recon_list, clean_list)
                recon_loss += cur_recon

            recon_loss = 1.0 * recon_loss.squeeze(0)
            for opt in opt_list:
                for attr in attr_list:
                    opt[attr].zero_grad()
            recon_loss.backward()
            for opt in opt_list:
                for attr in attr_list:
                    opt[attr].step()
            lm.add_torch_stat("unlabel_recon", recon_loss)
    
    for model in model_list:
        for attr in attr_list:
            model[attr].eval()

    with torch.no_grad():
        total_pred = [] 
        total_y = []
        for x, y in dev_loader:
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
        lm.add_torch_stat("dev_aro", ccc[0])
        lm.add_torch_stat("dev_dom", ccc[1])
        lm.add_torch_stat("dev_val", ccc[2])
        lm.add_torch_stat("dev_all", ccc[0]+ccc[1]+ccc[2])

    lm.print_stat()
    for attr in attr_list:
        dev_loss = 1.0 - lm.get_stat("dev_"+str(attr))
        if min_loss[attr] > dev_loss:
            min_epoch[attr] = epoch
            min_loss[attr] = dev_loss


    for attr in attr_list:
        torch.save(E_model[attr].state_dict(), os.path.join(model_path, "param", str(epoch)+"_E_"+attr+".pt"))
        torch.save(D_model[attr].state_dict(), os.path.join(model_path, "param", str(epoch)+"_D_"+attr+".pt"))
        
    end_time = time.time()
    print("Train elapsed:", end_time-start_time) 

print("Save",end=" ")
for attr in attr_list:
    print(min_epoch[attr], end=" ")
print("")

print("Loss",end=" ")
for attr in attr_list:
    print(1.0-min_loss[attr], end=" ")
print("")
for attr in attr_list:
    for mtype in ["E", "D"]:
        os.system("cp "+os.path.join(model_path, "param", str(min_epoch[attr])+"_"+mtype+"_"+attr+".pt") + \
            " "+os.path.join(model_path, "final_"+mtype+"_"+attr+".pt"))    