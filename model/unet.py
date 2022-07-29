import torch
import torch.nn as nn
import torch.nn.functional as F

def get_train_weights(train_dataset):
    train_weights = [0, 0, 0, 0, 0]
    for i in range(len(train_dataset)):
        image, seg, labels = train_dataset[i]
        segs = torch.mul(seg, labels)
        indx, counts = torch.unique(segs, return_counts=True)
        for ind, count in zip(indx, counts):
            train_weights[ind] += count
            
    weights = 1 / np.asarray(train_weights[1:])
    weights = weights / np.max(weights)
    return weights

def get_train_weights_binary(train_dataset):
    train_weights_bin_seg = [0, 0]
    for i in range(len(train_dataset)):
        image, seg, labels = train_dataset[i]
        indx, counts = torch.unique(seg, return_counts=True)
        for ind, count in zip(indx, counts):
            train_weights_bin_seg[ind] += count
    train_weights_bin_seg = train_weights_bin_seg[0]/train_weights_bin_seg[1]
    return train_weights_bin_seg

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, stage, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.stage = stage

        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        factor = 2 if bilinear else 1
        self.down4 = Down(64, 128 // factor)
        self.up1 = Up(128, 64 // factor, bilinear)
        self.up2 = Up(64, 32 // factor, bilinear)
        self.up3 = Up(32, 16 // factor, bilinear)
        self.up4 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)

        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(6400, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 4)
        
        # slope
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(160*160, 512)
        self.linear2 = nn.Linear(512, 256)
        self.batch1d = nn.BatchNorm1d(512)
        self.batch1d_n = nn.BatchNorm1d(160*160)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, s=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        pred = self.flatten(x5)
        pred = F.relu(self.lin1(pred))
        pred = F.relu(self.lin2(pred))

        if s != None:
            s = self.flatten(s)
            s = self.batch1d_n(s)
            s = self.dropout(s)
            s = self.batch1d(F.relu(self.linear1(s)))
            s = F.relu(self.linear2(s))
            
            pred = torch.concat((pred,s),1)
            pred = self.lin3(self.dropout(self.linear2(pred)))
        
        if self.stage == "classification":
            return logits, pred
        
        if self.stage == "projection":
            return pred