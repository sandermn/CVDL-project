import torchgeometry as tgm

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

def dice_metric(predb, yb):
    pred = predb[:,1:,:,:].clone().detach()
    dice_loss = tgm.losses.dice_loss(pred, yb)
    return 1 - dice_loss