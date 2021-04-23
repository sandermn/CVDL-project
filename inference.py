import torch
import matplotlib.pyplot as plt
from pathlib import Path
from Unet2D import Unet2D
from DatasetMedical import DatasetTEE, DatasetCAMUS
from torch.utils.data import DataLoader
from utils.helpers import predb_to_mask, batch_to_img
from utils.metrics import acc_metric, dice_function

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
            dice = dice_fn(outputs, y)
        running_acc += acc
        running_dice += dice
    
        if visual_debug:
            fig, ax = plt.subplots(x.size(0), 3, figsize=(15, x.size(0) * 5))
            for i in range(x.size(0)):
                ax[i, 0].imshow(batch_to_img(x, i))
                ax[i, 1].imshow(y[i].cpu())
                ax[i, 2].imshow(predb_to_mask(outputs, i))
            plt.show()

    return running_acc/len(test_dl), running_dice/len(test_dl)


def main(model_path, data_path, dataset):

    # split the training dataset and initialize the data loaders
    bs = 4

    if dataset == 'TEE':
        test_dataset = DatasetTEE(local_path / 'train_gray', local_path / 'train_gt')
    elif dataset == 'CAMUS':
        test_dataset = DatasetCAMUS(data_path, 
                        isotropic=False, 
                        include_es=False, 
                        include_2ch=True, 
                        include_4ch=False, 
                        start=401,
                        stop=450
        )
    
    test_dl = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    
    # build the Unet2D with one channel as input and 4 channels as output
    unet = Unet2D(1, 4)
    # Load best model params
    unet.load_state_dict(torch.load(model_path))

    acc, dice = test(unet, test_dl, acc_metric, dice_function, visual_debug=True)
    print('acc: ', acc,'dice: ', dice)

if __name__ == "__main__":
    model_path = Path('models/baseline/final.pth')
    data_path = Path('data/')
    dataset = 'CAMUS'
    # cluster_path = Path('/work/datasets/medical_project/TEE')
    # local_path = Path('data_tee/TEE')
    main(model_path, data_path, dataset)