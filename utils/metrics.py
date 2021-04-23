import torchgeometry as tgm
import torch
import numpy as np

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

def dice_function(predb, yb):
    scores = []
    for pred, y in zip(predb, yb):
        score = calc_dice_coef(pred, y)
        scores.append(score)

    return np.sum(scores) / len(scores)

def calc_dice_coef(pred, y):
    y_classes = list(torch.unique(y))
    p_classes = list(range(pred.shape[0]))
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


def jaccard_distance_loss(predb, yb):
    scores = []
    for pred, y in zip(predb, yb):
        score = calc_jdl(pred, y)
        scores.append(score)

    return np.sum(scores) / len(scores)

def calc_jdl(pred, y):
    y_classes = list(torch.unique(y))
    p_classes = list(range(pred.shape[0]))
    pred = pred.argmax(dim=0)
    jdls = []
    smooth = 1e-6
    for pc in p_classes[1:]:
        y_match = torch.where(y == pc, 1, 0).detach().cpu().numpy()
        p_match = torch.where(pred == pc, 1, 0).detach().cpu().numpy()
        TP = np.sum(a=np.multiply(y_match, p_match), axis=None, dtype=y_match.dtype)
        jac = (2*TP+smooth)/(np.sum(y_match)+np.sum(p_match)+TP+smooth)
        jdls.append((1-jac)*smooth)

    return np.mean(jdls)



