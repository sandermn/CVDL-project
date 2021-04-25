import torch
import matplotlib.pyplot as plt
from pathlib import Path
from Unet2D import Unet2D
from DatasetMedical import DatasetTEE, DatasetCAMUS
from torch.utils.data import DataLoader
from utils.helpers import predb_to_mask, batch_to_img
from utils.metrics import acc_metric
from torchmetrics import F1
import torch.nn.functional as F
def test(model, test_dl, acc_fn, dice_fn, visual_debug):
    model.cuda()
    model.eval()
    running_acc = 0
    running_dice = 0

    for x, y in test_dl:
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            outputs = model(x)
            acc = acc_fn(outputs, y)
            pred = F.softmax(outputs, dim=1)
            dice = dice_fn(pred, y)
        running_acc += acc.detach().item()
        running_dice += dice.detach().item()
    
        if visual_debug:
            bs=4
            fig, ax = plt.subplots(bs, 3, figsize=(15, bs * 5))
            for i in range(bs):
                ax[i, 0].imshow(batch_to_img(x, i))
                ax[i, 1].imshow(y[i].cpu())
                ax[i, 2].imshow(predb_to_mask(outputs, i))
            fig.savefig(Path('inference_outputs') / 'best_128_full.png')

    return running_acc/len(test_dl), running_dice/len(test_dl)


def main(model_path, data_path, dataset):

    # split the training dataset and initialize the data loaders
    bs = 5

    if dataset == 'TEE':
        test_dataset = DatasetTEE(data_path / 'train_gray', data_path / 'train_gt')
    elif dataset == 'CAMUS':
        test_dataset = DatasetCAMUS(data_path, 
                        isotropic=False, 
                        include_es=False, 
                        include_2ch=False, 
                        include_4ch=True, 
                        start=401,
                        stop=450
        )
    
    test_dl = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    
    # build the Unet2D with one channel as input and 4 channels as output
    dice_function = F1(num_classes=4, average='macro', ignore_index=0, mdmc_average='samplewise')

    unet = Unet2D(1, 4)
    # Load best model params
    unet.load_state_dict(torch.load(model_path))

    acc, dice = test(unet, test_dl, acc_metric, dice_function, visual_debug=True)
    print('acc: ', acc,'dice: ', dice)

if __name__ == "__main__":
    model_path = Path('models/full_dataset_novalidation/final.pth')
    data_path = Path('data/')
    dataset = 'CAMUS'
    # cluster_path = Path('/work/datasets/medical_project/TEE')
    # local_path = Path('data_tee/TEE')
    main(model_path, data_path, dataset)