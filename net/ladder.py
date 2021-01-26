import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

class Denoiser_MLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Denoiser_MLP, self).__init__()
        self.feat_num = args[0]

        self.W = nn.Sequential(
            nn.Linear(3, 4), nn.LeakyReLU(0.1),
            nn.Linear(4, 1)
        )
        
    def forward(self, u, z, uz):
        h = torch.stack([u,z,uz],2) # {32, 256, 3}
        result = self.W(h) # {32, 256, 1}
        result = result.squeeze(2)#reshape(-1, self.feat_num)
        return result

class HLD(nn.Module):
    def __init__(self, *args, **kwargs):
        super(HLD, self).__init__()
        input_dim = args[0]
        hidden_dim = args[1]
        num_layers = args[2]
        output_dim = args[3]
        p = kwargs.get("dropout", 0.1)

        self.inp_drop = nn.Dropout(0.2)
        self.inp_bn = nn.BatchNorm1d(input_dim, affine=False)

        self.W=nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.norm = nn.ModuleList([nn.BatchNorm1d(hidden_dim, affine=False)])
        self.norm_scale=nn.ParameterList([nn.Parameter(torch.ones(hidden_dim))])
        self.norm_bias=nn.ParameterList([nn.Parameter(torch.zeros(hidden_dim))])
        
        for lidx in range(num_layers-1):
            self.W.append(nn.Linear(hidden_dim, hidden_dim))
            self.norm.append(nn.BatchNorm1d(hidden_dim, affine=False))
            self.norm_scale.append(nn.Parameter(torch.ones(hidden_dim)))
            self.norm_bias.append(nn.Parameter(torch.zeros(hidden_dim)))

        self.out_W = nn.Linear(hidden_dim, output_dim)
        self.out_norm = nn.BatchNorm1d(output_dim)
        self.out_scale = nn.Parameter(torch.ones(output_dim))
        self.out_bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x, noisy=False, need_stat=False, eval=False):
        """
        (N, 6373) => (N, 256) => (N, 256) => (N, 3)
        """
        z_out = []
        stat_out = []
        
        # h(0) #
        z = x
        if noisy:
            z =  z + (torch.randn_like(z)*math.sqrt(0.3))
        elif need_stat:
            mu_pre = torch.zeros(z.size(1)).cuda()
            var_pre = torch.ones(z.size(1)).cuda()
            stat_out.append((mu_pre, var_pre))
        z_out.append(z)
        
        # h(1) ~ h(L-1) #
        h = self.inp_drop(z)
        # h = z
        
        for i, fc in enumerate(self.W):
            z_pre=fc(h)
            z = self.norm[i](z_pre)
            if noisy:
                z =  z + (torch.randn_like(z)*math.sqrt(0.3))
            elif need_stat:
                if eval:
                    mu_pre = self.norm[i].running_mean
                    var_pre = self.norm[i].running_var
                else:
                    mu_pre = torch.mean(z_pre, 0)
                    var_pre = torch.var(z_pre, 0, unbiased=False)
                stat_out.append((mu_pre, var_pre))
            z_out.append(z)
            h = F.relu(self.norm_scale[i]*z+self.norm_bias[i])

        # h(L) #
        o = self.out_W(h)
        o_norm = self.out_norm(o)
        if noisy:
            o_norm = o_norm + (torch.randn_like(o)*math.sqrt(0.3))
        elif need_stat:
            if eval:
                mu_pre = self.out_norm.running_mean
                var_pre = self.out_norm.running_var
            else:
                mu_pre = torch.mean(o, 0)
                var_pre = torch.var(o, 0, unbiased=False)
            stat_out.append((mu_pre, var_pre))
        z_out.append(o_norm)
        y = self.out_scale*o_norm + self.out_bias

        # y = h
        if need_stat:
            return y, z_out, stat_out
        else:
            return y, z_out 

class HLD_Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(HLD_Decoder, self).__init__()
        input_dim = args[0]
        hidden_dim = args[1]
        num_layers = args[2]
        output_dim = args[3]

        self.decoder=nn.ModuleList([
            nn.BatchNorm1d(output_dim, affine=False),
            nn.Sequential(
                nn.Linear(output_dim, hidden_dim), nn.BatchNorm1d(hidden_dim, affine=False)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim, affine=False)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, input_dim), nn.BatchNorm1d(input_dim, affine=False)
            )
        ])
        self.denoiser=nn.ModuleList([
            Denoiser_MLP(output_dim),
            Denoiser_MLP(hidden_dim),
            Denoiser_MLP(hidden_dim),
            Denoiser_MLP(input_dim)
        ])
    def forward(self, y_tilde, h_list, stat_list):
        h_len = len(h_list)
        zbn_out=[]
        z_hat = y_tilde
        for i, fc in enumerate(self.decoder):   
            u = fc(z_hat)
            z_tilde = h_list[h_len-1-i]
            z_hat = self.denoiser[i](u, z_tilde, u*z_tilde)
            mu, var = stat_list[h_len-1-i]
            z_bn = (z_hat - mu) / torch.sqrt(var+1e-05)
            zbn_out.append(z_bn)

        return zbn_out
