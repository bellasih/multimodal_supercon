import torch
import torch.nn as nn
import torch.nn.functional as F

def get_acc_seg(outputs, segs):
    outputs = torch.max(outputs, dim=1)[1]
    acc = (outputs==segs)
    acc = acc.view(-1)
    return acc.sum()/len(acc)

def get_acc_seg_weighted(outputs, segs):
    outputs = torch.max(outputs, dim=1)[1]
    acc = []
    for i in range(5):
        acc_temp = (outputs==segs)
        acc_temp = acc_temp.view(-1)
        acc.append(acc_temp.sum()/len(acc_temp))
    return torch.mean(torch.stack(acc))

def get_acc_nzero(outputs, segs):
    mask = ~segs.eq(0)
    outputs = torch.max(outputs, dim=1)[1]
    acc = torch.masked_select((outputs==segs), mask)
    return acc.sum()/len(acc)

def get_acc_class(outputs, labels):
    outputs = torch.max(outputs, dim=1)[1]
    acc = (outputs==labels)
    return acc.sum()/len(acc)

def get_acc_binseg(outputs, segs):
    outputs = F.sigmoid(outputs)
    outputs[outputs>=0.5] = 1
    outputs[outputs<0.5] = 0
    acc = (outputs==segs)
    acc_1 = acc[segs==1].view(-1)
    acc_0 = acc[segs==0].view(-1)
    acc_1 = acc_1.sum()/len(acc_1)
    acc_0 = acc_0.sum()/len(acc_0)
    return acc_1, acc_0, (acc_1+acc_0)/2