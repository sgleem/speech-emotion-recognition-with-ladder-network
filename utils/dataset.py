import torch
import torch.nn as nn
import numpy as np

def calculate_stat(feat_list, exclude=0.5):
    # feat_mean = np.mean(feat_list, axis=0)
    # feat_var = np.var(feat_list, axis=0, ddof=0)
    # feat_std = np.sqrt(feat_var+1e-5)

    mean_list = []
    var_list = []
    total_list = feat_list.T
    for feat_idx, cur_feats in enumerate(total_list):
        lower_bound = np.percentile(cur_feats, exclude)
        upper_bound = np.percentile(cur_feats, 100-exclude)
        
        cur_list = []
        for elem in cur_feats:
            if elem < lower_bound or elem > upper_bound:
                continue
            cur_list.append(elem)
        mean_list.append(np.mean(cur_list))
        var_list.append(np.var(cur_list, ddof=0))
    
    return np.array(mean_list), np.array(var_list)

class LLDset():
    def __init__(self, *args, **kwargs):
        self.feat_dict = kwargs.get("feat_dict", args[0])
        self.lab_dict = kwargs.get("lab_dict", args[1])
        self.data_num = kwargs.get("data_num", -1)

        if self.data_num == -1:
            self.data_num = len(self.feat_dict)
        
        self.utt_list = list(self.feat_dict.keys())[:self.data_num]

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        utt_id = self.utt_list[idx]
        feat_mat = np.array(self.feat_dict[utt_id]).T
        cur_lab = self.lab_dict[utt_id]
        labs=np.array([cur_lab[1],cur_lab[2],cur_lab[3]]) # Act Dom Val

        return (feat_mat, labs)

class HLDset():
    def __init__(self, *args,   **kwargs):
        feat_dict = kwargs.get("feat_dict", args[0])
        lab_dict = kwargs.get("lab_dict", args[1])

        self.feat_list = []
        self.lab_list = []
        self.utt_list = []

        self.feat_mean = kwargs.get("feat_mean", None)
        self.feat_var = kwargs.get("feat_var", None)
        self.lab_mean = kwargs.get("lab_mean", None)
        self.lab_var = kwargs.get("lab_var", None)
        
        self.data_num = 0
        self.unlabeled = True if lab_dict is None else False
        # print(self.unlabeled)
        # print("Feature Selection")
        for utt_id, feat_vec in feat_dict.items():
            if len(feat_vec) != 0:
                self.utt_list.append(utt_id)
                self.feat_list.append(feat_vec)
                
                if not self.unlabeled:
                    cur_lab = lab_dict[utt_id]
                    lab_vec=np.array(cur_lab)
                    self.lab_list.append(lab_vec)

                self.data_num += 1
        
        del feat_dict
        if not self.unlabeled:
            del lab_dict

        self.feat_list = np.array(self.feat_list)
        if not self.unlabeled:
            self.lab_list = np.array(self.lab_list)
        
        # print("Statistics calculation")
        if self.feat_mean is None and self.feat_var is None:
            self.feat_mean, self.feat_var = calculate_stat(self.feat_list)
            # self.feat_mean = np.mean(self.feat_list, axis=0)
            # self.feat_var = np.var(self.feat_list, axis=0, ddof=0)

        if not self.unlabeled:
            if self.lab_mean is None and self.lab_var is None:
                self.lab_mean = np.mean(self.lab_list, axis=0)
                self.lab_var = np.var(self.lab_list, axis=0, ddof=0)

        # Normalization
        
        for idx, feat_vec in enumerate(self.feat_list):
            feat_vec = (self.feat_list[idx] - self.feat_mean) / np.sqrt(self.feat_var + 0.000001)
            feat_vec = np.clip(feat_vec, -10, 10)
            self.feat_list[idx] = feat_vec
        
        if not self.unlabeled:
            for idx, feat_vec in enumerate(self.lab_list):
                labs = (self.lab_list[idx] - self.lab_mean) / np.sqrt(self.lab_var + 0.000001)
                self.lab_list[idx] = labs

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        # feat_vec = (self.feat_list[idx] - self.feat_mean) / np.sqrt(self.feat_var + 0.000001)
        # feat_vec = np.clip(feat_vec, -10, 10)
        if not self.unlabeled:
            # labs = (self.lab_list[idx] - self.lab_mean) / np.sqrt(self.lab_var + 0.000001)
            # result = (feat_vec, labs)
            result = (self.feat_list[idx], self.lab_list[idx])
        else:
            # result = feat_vec
            result = self.feat_list[idx]
        return result

def get_loader_lld(dataset, batch_size, shuffle):
    data_idxs = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(data_idxs)
    
    batch_count = 0
    tot_x = []; tot_y = []
    for cur_idx in data_idxs:
        x, y = dataset[cur_idx]
        
        x = torch.Tensor(x).float().cuda()

        tot_x.append(x)
        tot_y.append(y)

        batch_count += 1
        # print(batch_count)
        if batch_count == batch_size:
            if batch_size == 1:
                tot_x = tot_x[0]
                tot_x = tot_x.reshape(1, tot_x.shape[0], tot_x.shape[1])
            else:
                tot_x = nn.utils.rnn.pad_sequence(tot_x, batch_first=True)
            tot_y = torch.Tensor(tot_y).float().cuda()

            yield (tot_x, tot_y)

            batch_count = 0
            tot_x = []; tot_y = []

    if batch_count != 0 and batch_size != 1:
        if len(tot_x) == 1:
            tot_x = tot_x[0]
            tot_x = tot_x.reshape(1, tot_x.shape[0], tot_x.shape[1])
        else:
            tot_x = nn.utils.rnn.pad_sequence(tot_x, batch_first=True)
        tot_y = torch.Tensor(tot_y).float().cuda()

        yield tot_x, tot_y

def get_loader_hld(dataset, batch_size, shuffle):
    data_idxs = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(data_idxs)
    
    batch_count = 0
    tot_x = []; tot_y = []
    for cur_idx in data_idxs:
        x, y = dataset[cur_idx]
        
        x = torch.Tensor(x).float().cuda()
        
        tot_x.append(x)
        tot_y.append(y)

        batch_count += 1
        # print(batch_count)
        if batch_count == batch_size:
            if batch_size == 1:
                tot_x = torch.Tensor([tot_x]).float().cuda()
            else:
                tot_x = torch.cat(tot_x, 0).float().cuda()
            
            tot_y = torch.Tensor(tot_y).float().cuda()

            yield (tot_x, tot_y)

            batch_count = 0
            tot_x = []; tot_y = []

    if batch_count != 0 and batch_size != 1:
        if len(tot_x) == 1:
            tot_x = torch.Tensor([tot_x]).float().cuda()
        else:
            tot_x = torch.cat(tot_x, 0).float().cuda()
        
        tot_y = torch.Tensor(tot_y).float().cuda()

        yield tot_x, tot_y

class HLDset_Utt():
    def __init__(self, *args,   **kwargs):
        feat_dict = kwargs.get("feat_dict", args[0])
        lab_dict = kwargs.get("lab_dict", args[1])

        self.feat_list = []
        self.lab_list = []
        self.utt_list = []

        self.feat_mean = kwargs.get("feat_mean", None)
        self.feat_var = kwargs.get("feat_var", None)
        self.lab_mean = kwargs.get("lab_mean", None)
        self.lab_var = kwargs.get("lab_var", None)
        
        self.data_num = 0
        self.unlabeled = True if lab_dict is None else False
        print(self.unlabeled)
        print("Feature Selection")
        for utt_id, feat_vec in feat_dict.items():
            if len(feat_vec) != 0:
                self.feat_list.append(feat_vec)
                self.utt_list.append(utt_id)
                
                if not self.unlabeled:
                    cur_lab = lab_dict[utt_id]
                    lab_vec=np.array(cur_lab)
                    self.lab_list.append(lab_vec)

                self.data_num += 1
        
        del feat_dict
        if not self.unlabeled:
            del lab_dict

        self.feat_list = np.array(self.feat_list)
        if not self.unlabeled:
            self.lab_list = np.array(self.lab_list)

        print("Statistics calculation")
        if self.feat_mean is None and self.feat_var is None:
            self.feat_mean = np.mean(self.feat_list, axis=0)
            self.feat_var = np.var(self.feat_list, axis=0, ddof=0)

        if not self.unlabeled:
            if self.lab_mean is None and self.lab_var is None:
                self.lab_mean = np.mean(self.lab_list, axis=0)
                self.lab_var = np.var(self.lab_list, axis=0, ddof=0)
        
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        feat_vec = (self.feat_list[idx] - self.feat_mean) / np.sqrt(self.feat_var + 0.000001)
        utt_id = self.utt_list[idx]
        if not self.unlabeled:
            labs = (self.lab_list[idx] - self.lab_mean) / np.sqrt(self.lab_var + 0.000001)
            result = (feat_vec, labs, utt_id)
        else:
            result = (feat_vec, utt_id)
        return result
    
    def get_data_by_utt(self, utt_id):
        utt_id = list(utt_id)[0]
        idx = self.utt_list.index(utt_id)
        feat_vec, labs, cur_id = self.__getitem__(idx)
        feat_vec = torch.Tensor(feat_vec).unsqueeze(0)
        labs = torch.Tensor(labs).unsqueeze(0)
        assert cur_id == utt_id

        return (feat_vec, labs)