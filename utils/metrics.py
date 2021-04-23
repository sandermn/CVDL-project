import torchgeometry as tgm
import torch
import numpy as np
import torch.nn.functional as F

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()
"""

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
    
"""

def dice_function(predb, yb):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        yb/true: a tensor of shape [B, 1, H, W].
        predb/logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = predb.shape[1]
    eps= 1e-7
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[yb.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(predb)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[yb.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(predb, dim=1)
    true_1_hot = true_1_hot.type(predb.type())
    dims = (0,) + tuple(range(2, yb.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_score = (2. * intersection / (cardinality + eps)).mean()
    return dice_score

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



