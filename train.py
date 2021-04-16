import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import time
import os
import random
from pathlib import Path
import torch
import torchgeometry as tgm
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn
from torchvision import transforms
from helpers import get_train_val_set
from DatasetMedical import DatasetCAMUS_r, DatasetCAMUS, DatasetTEE
from Unet2D import Unet2D


def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, params_path, epochs=1):
    start = time.time()
    model.cuda()

    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

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

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                # stats - whatever is the phase
                acc = acc_fn(outputs, y)

                running_acc += acc * dataloader.batch_size
                running_loss += loss * dataloader.batch_size

                if step % 100 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc,
                                                                                          torch.cuda.memory_allocated() / 1024 / 1024))
                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)

            train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)
        #torch.save(model.state_dict(), params_path + f'{epoch}.pth')
    torch.save(model.state_dict(), params_path + 'final.pth')
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_loss, valid_loss


def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

def dice_metric(predb, yb):
    dice_loss = tgm.losses.dice_loss(predb, yb)
    return 1 - dice_loss

def multiclass_dice(predb, yb, num_labels):
    multi_dice = 0
    for i in range(num_labels):
        multi_dice += dice_metric(predb, yb, i)

def batch_to_img(xb, idx):
    img = np.array(xb[idx, 0:3])
    return img.transpose((1, 2, 0))


def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()


def main():
    # enable if you want to see some plotting
    visual_debug = True
    params_path = 'models/model_epoch_'

    # batch size
    bs = 4

    # epochs
    epochs_val = 10

    # learning rate
    learn_rate = 0.01

    # datasets, 0=background+LV, 1=b+LV+M+RV, 2=??
    datasets = ['CAMUS_resized', 'CAMUS', 'TEE']
    curr_dataset = datasets[1]
    outputs = 4

    # sets the matplotlib display backend (most likely not needed)
    # mp.use('TkAgg', force=True)

    # Preprocessing
    pre_process = transforms.Compose([
        transforms.GaussianBlur(3, sigma=(0.1,1))
    ])
    transform = transforms.Compose([
        # transforms.RandomVerticalFlip(p=0.3),
        # transforms.Resize((224,224))
    ])

    # load the training data
    train_dataset, valid_dataset = get_train_val_set(curr_dataset, pre_process, transform)

    train_dl = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=bs, shuffle=False)

    if visual_debug:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(train_dataset.open_as_array(150))
        mask = train_dataset.open_mask(150)
        ax[1].imshow(mask)
        plt.show()

    xb, yb = next(iter(train_dl))
    print(xb.shape, yb.shape)

    # build the Unet2D with one channel as input and 2 channels as output
    unet = Unet2D(1, outputs)

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=learn_rate)

    # do some training
    train_loss, valid_loss = train(unet, train_dl, valid_dl, loss_fn, opt, acc_metric, epochs=epochs_val,
                                   params_path=params_path)

    # plot training and validation losses
    if visual_debug:
        plt.figure(figsize=(10, 8))
        plt.plot(train_loss, label='Train loss')
        plt.plot(valid_loss, label='Valid loss')
        plt.legend()
        plt.show()

    # predict on the next train batch (is this fair?)
    xb, yb = next(iter(train_dl))
    with torch.no_grad():
        predb = unet(xb.cuda())

    # show the predicted segmentations
    if visual_debug:
        fig, ax = plt.subplots(bs, 3, figsize=(15, bs * 5))
        for i in range(bs):
            ax[i, 0].imshow(batch_to_img(xb, i))
            ax[i, 1].imshow(yb[i])
            ax[i, 2].imshow(predb_to_mask(predb, i))

        plt.show()

if __name__ == "__main__":
    main()