import matplotlib.pyplot as plt
import time
import torch
import os
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from utils.helpers import get_train_val_set, predb_to_mask, batch_to_img, make_3d
from Unet2D import Unet2D
from utils.metrics import acc_metric, DiceLoss
from torchmetrics import F1



def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, dice_fn, params_path, epochs=1):
    start = time.time()
    model.cuda()

    train_loss, valid_loss = [], []

    best_loss = 10
    es_counter = 0
    
    if params_path:
        params_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
                dataloader = train_dl
            else:
                model.train(False)
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0
            running_dice = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                x = x.cuda()
                y = y.cuda()
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                # stats 
                acc = acc_fn(outputs, y)
                pred = F.softmax(outputs, dim=1)
                dice = dice_fn(pred, y)

                running_acc += acc * x.size(0) 
                running_loss += loss * x.size(0) 
                running_dice += dice * x.size(0)

                if step % 100 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc,
                                                                                          torch.cuda.memory_allocated() / 1024 / 1024))
                    # print(torch.cuda.memory_summary())

            epoch_loss = (running_loss / len(dataloader.dataset))
            epoch_acc = running_acc / len(dataloader.dataset)
            epoch_dice = running_dice / len(dataloader.dataset)
            # epoch_dice = 1
            
            print('Epoch {}/{}'.format(epoch+1, epochs))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {} Dice: {}'.format(phase, epoch_loss, epoch_acc, epoch_dice))
            print('-' * 10)

            train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)

            # for each tenth epoch save predictions of the last batch
            if phase == 'valid' and epoch+1 % 10 == 0 and visual_debug:
                fig, ax = plt.subplots(4, 7, figsize=(4*6, 3 * 6))
                for i in range(4):
                    ax[i, 0].imshow(batch_to_img(x, i))
                    ax[i, 1].imshow(y[i].cpu())
                    for channel in range(4):
                        ax[i, 2+channel].imshow(outputs[i, channel, :, :].cpu())
                    ax[i, 6] .imshow(predb_to_mask(outputs, i))
                #plt.show()
                fig.savefig(params_path/f'predictions_epoch{epoch}.png')


        #torch.save(model.state_dict(), params_path + f'{epoch}.pth')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            es_counter = 0
            best_acc = epoch_acc
            # best_dice = epoch_dice
            best_model = model.state_dict()

        es_counter += 1
        if (es_counter > 10):
            print(f'Early stopped after epoch {epoch}')
            break
    if params_path:
        params_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), params_path/'final.pth')
        f = open(params_path / 'config.txt', 'w')
        f.write(f'bs: {bs},\nepochs: {epochs_val},\nlearn_rate: {learn_rate},\nloss: {epoch_loss},\nacc: {epoch_acc},\ndice:{epoch_dice}')
        f.close()

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_loss, valid_loss


def main(
    visual_debug=False, 
    params_path=None, 
    bs=4,
    epochs_val=10,
    learn_rate=0.01,
    ckpt=None,
    dataset='CAMUS',
    outputs=4,
    pre_process=None,
    transform=None,
    isotropic=False,
    include_es=False,
    is_local=False,
    include_2ch=False,
    include_4ch=False
    ):
    # enable if you want to see some plotting
    
    # load the training data
    train_dataset, valid_dataset, _ = get_train_val_set(dataset, is_local, pre_process, transform, isotropic, include_es, include_2ch, include_4ch)

    train_dl = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=bs, shuffle=False)
    
    print(f'Size of train dataset: {len(train_dataset)}\nSize of validation dataset: {len(valid_dataset)}')

    # if visual_debug:
    #    fig, ax = plt.subplots(1, 2)
    #    ax[0].imshow(train_dataset.open_as_array(150))
    #    mask = train_dataset.open_mask(150)
    #    ax[1].imshow(mask)
    #    plt.show()

    xb, yb = next(iter(train_dl))
    #print('c', xb.shape, yb.shape)

    # build the Unet2D with one channel as input and 4 channels as output
    unet = Unet2D(1, outputs)
    
    if ckpt:
        unet.load_state_dict(torch.load(ckpt))
    # loss function and optimizer
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = DiceLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=learn_rate)
    dice_function = F1(num_classes=4, average='macro', ignore_index=0, mdmc_average='samplewise')

    train_loss, valid_loss = train(unet, train_dl, valid_dl, loss_fn, opt, acc_metric, dice_function, epochs=epochs_val,
                                   params_path=params_path)

    # plot training and validation losses
    if visual_debug:
        plt.figure(figsize=(10, 8))
        plt.plot(train_loss, label='Train loss')
        plt.plot(valid_loss, label='Valid loss')
        plt.legend()
        plt.show()
        plt.savefig(params_path/'loss.png')
    # predict on the next train batch (is this fair?)
    xb, yb = next(iter(train_dl))
    with torch.no_grad():
        predb = unet(xb.cuda())
    # show the predicted segmentations
    if visual_debug:
        fig, ax = plt.subplots(4, 3, figsize=(15, 4 * 5))
        for i in range(4):
            ax[i, 0].imshow(batch_to_img(xb, i))
            ax[i, 1].imshow(yb[i])
            ax[i, 2].imshow(predb_to_mask(predb, i))
        #plt.show()
        fig.savefig(params_path/'predictions_final.png')

if __name__ == "__main__":
    # Visual Debug
    visual_debug = True
    
    # Model Save Path
    # Use models/custom
    params_path = Path('models/diceloss')

    # parameters
    bs = 8
    epochs_val = 50
    learn_rate = 0.01
    dataset = 'CAMUS'
    outputs = 4
    ckpt = None
    isotropic = False
    include_es = False
    is_local = True
    include_2ch = False
    include_4ch = True

    # Preprocessing
    pre_process = transforms.Compose([
        # transforms.GaussianBlur(9, sigma=(0.1,1))
    ])
    augmentation = transforms.Compose([
        #transforms.RandomHorizontalFlip(p=0.3)
    ])

    #ckpt = 'models/4ch_50_epochsfinal.pth'
    main(
        visual_debug=visual_debug,
        bs=bs,
        params_path=params_path,
        epochs_val=epochs_val,
        learn_rate=learn_rate,
        dataset=dataset,
        outputs=outputs,
        pre_process=pre_process,
        transform=augmentation,
        ckpt=ckpt,
        isotropic=isotropic,
        include_es=include_es,
        is_local=is_local,
        include_2ch=include_2ch,
        include_4ch=include_4ch
    )