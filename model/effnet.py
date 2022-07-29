import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils, datasets, models
from efficientnet_pytorch import EfficientNet
from torch.optim import lr_scheduler

class EffnetMLP(nn.Module):
    def __init__(self, enet_type, out_dim, **kwargs):
        super(EffnetMLP, self).__init__()
        self.enet = EfficientNet.from_pretrained(enet_type)
        for m in self.enet.modules():
            m.requires_grad = False
            
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet._fc.in_features
        self.linear_layer = nn.Linear(in_ch, out_dim)
        self.enet._fc = nn.Identity()
        
        self.flatten = nn.Flatten()
        self.linear_img = nn.Linear(1408, 512)
        self.linear_fin = nn.Linear(1024,1408)
        self.linear1 = nn.Linear(160*160, 512)
        self.linear2 = nn.Linear(512, 512)
        self.batch1d = nn.BatchNorm1d(512)
        self.batch1d_n = nn.BatchNorm1d(160*160)
        self.dropout = nn.Dropout(0.2)
        
    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, s):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        x = torch.flatten(x, 1)
        x = self.linear_img(x)

        # slope
        s = self.flatten(s)
        s = self.batch1d_n(s)
        s = self.dropout(s)
        s = self.batch1d(F.leaky_relu(self.linear1(s)))
        s = F.leaky_relu(self.linear2(s))
        
        x = torch.cat((x, s), dim=1)
        x = self.linear_fin(x)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.linear_layer(dropout(x))
            else:
                out += self.linear_layer(dropout(x))
        out /= len(self.dropouts)
        return out

def _effnet(arch, **kwargs):
    model = EffnetMLP(arch, 4, **kwargs)
    return model

def effnetb2(pretrained=False, **kwargs):
    return _effnet('efficientnet-b2', **kwargs)
