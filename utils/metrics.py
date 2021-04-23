import torchgeometry as tgm
import torch
import numpy as np
import torch.nn.functional as F

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()


def dice_function(predb, yb):
    scores = []
    for pred, y in zip(predb, yb):
        score = calc_dice_coef(pred, y)
        print('df', score.requires_grad)
        scores.append(score) 
    return torch.mean(torch.stack(scores))
    
        

def calc_dice_coef(pred, y):
    #print(f'calc_dice_coef: {pred.shape}{y.shape}')
    y_classes = list(torch.unique(y)) # [0,1,2,3] or [0,1]
    p_classes = list(range(pred.shape[0]))
    #print(f'y_classes: {y_classes}, p_classes: {p_classes}')
    #assert y_classes == p_classes , "pred and y should have the same channels"
    print('cdc1', pred.requires_grad)
    pred = pred.argmax(dim=0)
    dice = []
    smooth = 1e-6
    for pc in p_classes[1:]:
        y_match = torch.where(y == pc, 1, 0)
        p_match = torch.where(pred == pc, 1, 0)
        TP = torch.sum(torch.multiply(y_match, p_match), dtype=y_match.dtype)
        print('cdc1.5', TP.requires_grad)
        dice.append((2*TP+smooth)/(torch.sum(y_match)+torch.sum(p_match)+smooth))
    doice = torch.mean(torch.stack(dice))
    doice.requires_grad(True)
    print('cdc2', doice.requires_grad)
    return doice
                    
def dice_loss(predb, yb):
    dl = dice_function(predb, yb)
    return torch.add(torch.neg(dl), torch.FloatTensor([1.]))