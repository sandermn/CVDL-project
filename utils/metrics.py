import torchgeometry as tgm
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

def dice_function(predb, yb):
    scores = []
    for pred, y in zip(predb, yb):
        score = calc_dice_coef(pred, y)
        #print('df', score.requires_grad)
        scores.append(score) 
    return torch.mean(torch.stack(scores))
    
        

def calc_dice_coef(pred, y):
    #print(f'calc_dice_coef: {pred.shape}{y.shape}')
    y_classes = list(torch.unique(y)) # [0,1,2,3] or [0,1]
    p_classes = list(range(pred.shape[0]))
    #print(f'y_classes: {y_classes}, p_classes: {p_classes}')
    #assert y_classes == p_classes , "pred and y should have the same channels"
    pred = pred.argmax(dim=0)
    dice = []
    smooth = 1e-6
    for pc in p_classes[1:]:
        y_match = torch.where(y == pc, 1, 0)
        p_match = torch.where(pred == pc, 1, 0)
        TP = torch.sum(torch.multiply(y_match, p_match), dtype=y_match.dtype)
        #print('cdc1.5', TP.requires_grad)
        dice.append((2*TP+smooth)/(torch.sum(y_match)+torch.sum(p_match)+smooth))
    doice = torch.mean(torch.stack(dice))
    doice.requires_grad_(True)
    #print('cdc2', doice.requires_grad)
    return doice

                    
def dice_loss(predb, yb):
    dl = dice_function(predb, yb)
    return torch.add(torch.neg(dl), torch.FloatTensor([1.]).cuda())


def jaccard_distance_loss(predb, yb):
    scores = []
    for pred, y in zip(predb, yb):
        score = calc_jdl(pred, y)
        scores.append(score)

    return torch.mean(torch.stack(scores))

def calc_jdl(pred, y):
    y_classes = list(torch.unique(y))
    p_classes = list(range(pred.shape[0]))
    pred = F.softmax(pred, dim=0)
    jdls = []
    smooth = 1e-6
    for pc in p_classes[1:]:
        y_match = torch.where(y == pc, 1, 0)
        p_match = torch.where(pred == pc, 1, 0)
        TP = torch.sum(torch.multiply(y_match, p_match), dtype=y_match.dtype)
        jac = (2*TP+smooth)/(torch.sum(y_match)+torch.sum(p_match)+TP+smooth)
        jdls.append((1-jac)*100)
    
    jdls = torch.mean(torch.stack(jdls))
    jdls.requires_grad_(True)

    return jdls

class DiceLoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-7):
        super().__init__()
        self.eps, self.reduction = eps, reduction
        
    def forward(self, output, targ):
        """
        output is NCHW, targ is NHW
        """
        eps = 1e-7
        # convert target to onehot # eye(4)[32,384,384] 
        targ_onehot = torch.eye(output.shape[1])[targ].permute(0,3,1,2).float().cuda()
        # convert logits to probs
        pred = self.activation(output)
        # sum over HW
        inter = (pred * targ_onehot).sum(axis=[-1, -2])
        union = (pred + targ_onehot).sum(axis=[-1, -2])
        # mean over C
        loss = 1. - (2. * inter / (union + self.eps)).mean(axis=-1)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()

    def activation(self, output):
        return F.softmax(output, dim=1)