import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

class HLD(nn.Module):
    def __init__(self, *args, **kwargs):
        super(HLD, self).__init__()
        input_dim = args[0]
        hidden_dim = args[1]
        num_layers = args[2]
        output_dim = args[3]
        p = kwargs.get("dropout", 0.5)
        self.fc=nn.ModuleList([
            nn.Sequential(
                nn.Dropout(p),
                nn.Linear(input_dim, hidden_dim), nn.ReLU()
            )
        ])
        for lidx in range(num_layers-1):
            self.fc.append(
                nn.Sequential(
                    nn.Dropout(p),
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
                )
            )
        self.out = nn.Sequential(
                nn.Linear(hidden_dim, output_dim)
            )
    def forward(self, x):
        h = x
        for fc in self.fc:
            h=fc(h)

        result = self.out(h)
        return result
