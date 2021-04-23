import torchgeometry as tgm
import torch
import numpy as np

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

"""
def dice_metric(predb, yb):
    #pred = predb[:,1:,:,:].clone().detach()
    
    dice_loss = tgm.losses.dice_loss(predb, yb)
    return 1 - dice_loss


def dice_loss(predb, yb):
    return tgm.losses.dice_loss(predb, yb)
"""



def dice_function(predb, yb):
    #print(f'dice_function: {predb.shape}{yb.shape}')
    scores = []
    for pred, y in zip(predb, yb):
        score = calc_dice_coef(pred, y)
        scores.append(score)

    return np.sum(scores) / len(scores)
    
        

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
        y_match = torch.where(y == pc, 1, 0).detach().cpu().numpy()
        p_match = torch.where(pred == pc, 1, 0).detach().cpu().numpy()
        TP = np.sum(a=np.multiply(y_match, p_match), axis=None, dtype=y_match.dtype)
        dice.append((2*TP+smooth)/(np.sum(y_match)+np.sum(p_match)+smooth))
        
    return np.mean(dice)
                    

def dice_loss(predb, yb):
    return 1 - dice_function(predb, yb)

