import torch
import torch.nn.functional as F
import numpy as np

class LogManager:
    def __init__(self):
        self.log_book=dict()
    def alloc_stat_type(self, stat_type):
        self.log_book[stat_type] = []
    def alloc_stat_type_list(self, stat_type_list):
        for stat_type in stat_type_list:
            self.alloc_stat_type(stat_type)
    def init_stat(self):
        for stat_type in self.log_book.keys():
            self.log_book[stat_type] = []
    def add_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat)
    def add_torch_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat.detach().cpu().item())
    def get_stat(self, stat_type):
        result_stat = 0
        stat_list = self.log_book[stat_type]
        if len(stat_list) != 0:
            result_stat = np.mean(stat_list)
            result_stat = np.round(result_stat, 4)
        return result_stat

    def print_stat(self):
        for stat_type in self.log_book.keys():
           stat = self.get_stat(stat_type)
           print(stat_type,":",stat, end=' / ')
        print(" ")

def CCC_loss(pred, lab, m_lab=None, v_lab=None):
    """
    pred: (N, 3)
    lab: (N, 3)
    """
    m_pred = torch.mean(pred, 0, keepdim=True)
    m_lab = torch.mean(lab, 0, keepdim=True)

    d_pred = pred - m_pred
    d_lab = lab - m_lab

    v_pred = torch.var(pred, 0, unbiased=False)
    v_lab = torch.var(lab, 0, unbiased=False)

    corr = torch.sum(d_pred * d_lab, 0) / (torch.sqrt(torch.sum(d_pred ** 2, 0)) * torch.sqrt(torch.sum(d_lab ** 2, 0)))

    s_pred = torch.std(pred, 0, unbiased=False)
    s_lab = torch.std(lab, 0, unbiased=False)

    ccc = (2*corr*s_pred*s_lab) / (v_pred + v_lab + (m_pred[0]-m_lab[0])**2)    
    return ccc

def ladder_loss(decoder_h, encoder_h, layer_wise=False):
    assert len(decoder_h) == len(encoder_h)

    h_num = len(decoder_h)
    if layer_wise:
        total_loss=[torch.zeros(1).float().cuda() for h in range(h_num)]
    else:
        total_loss = torch.zeros(1).float().cuda()
    for h_idx in range(h_num):
        if layer_wise:
            total_loss[h_idx] += F.mse_loss(decoder_h[h_idx], encoder_h[h_num-1-h_idx])
        else:
            total_loss += F.mse_loss(decoder_h[h_idx], encoder_h[h_num-1-h_idx])
    return total_loss

def decoupled_ladder_loss(decoder_he, encoder_he, decoder_hr, encoder_hr):
    assert len(decoder_he) == len(encoder_he) == len(decoder_hr) == len(encoder_hr)

    h_num = len(decoder_he)
    total_loss = torch.zeros(1).float().cuda()
    for h_idx in range(h_num):
        if h_idx == h_num - 1:
            x = encoder_he[h_num-h_idx-1]
            recon_x = decoder_he[h_idx]+decoder_hr[h_idx]
            total_loss += F.mse_loss(recon_x, x)
        else:
            total_loss += F.mse_loss(decoder_he[h_idx], encoder_hr[h_num-h_idx-1])
            total_loss += F.mse_loss(decoder_he[h_idx], encoder_hr[h_num-h_idx-1])
    return total_loss


def orthogonal_loss(eh, rh, eps=0.0):
    batch_size = eh.size(0)
    out = torch.zeros(1).cuda()
    
    for e, r in zip(eh, rh):
        len_e = torch.sqrt(torch.sum(torch.pow(e, 2)))
        len_r = torch.sqrt(torch.sum(torch.pow(r, 2)))
        out += torch.dot(e, r) / ((len_e*len_r)+eps)
    out /= batch_size
    out = torch.abs(out)
    return out

def MSE_emotion(pred, lab):
    aro_loss = F.mse_loss(pred[:][0], lab[:][0])
    dom_loss = F.mse_loss(pred[:][1], lab[:][1])
    val_loss = F.mse_loss(pred[:][2], lab[:][2])

    return [aro_loss, dom_loss, val_loss]