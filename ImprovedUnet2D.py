import torch
from torch import nn

class ImprovedUnet2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #num. of channels proportional with int k
        k = 1
        filter = [32*k, 64*k, 128*k, 256*k, 512*k]
        self.moreLayers = False #ads two more layers

        kernelSize = 3
        padding = 1
        if self.moreLayers:

            self.conv1 = self.contract_block(in_channels, filter[0], 7, 3)
            self.conv2 = self.contract_block(filter[0], filter[1], kernelSize, padding)
            self.conv3 = self.contract_block(filter[1], filter[2], kernelSize, padding)
            self.conv4 = self.contract_block(filter[2], filter[3], kernelSize, padding)
            self.conv5 = self.contract_block(filter[3], filter[4], kernelSize, padding)

            self.upconv5 = self.expand_block(filter[4], filter[3], kernelSize, padding)
            self.upconv4 = self.expand_block(filter[3] * 2, filter[2], kernelSize, padding)
            self.upconv3 = self.expand_block(filter[2] * 2, filter[1], kernelSize, padding)
            self.upconv2 = self.expand_block(filter[1] * 2, filter[0], kernelSize, padding)
            self.upconv1 = self.expand_block(filter[0] * 2, out_channels, kernelSize, padding)
        else:
            self.conv1 = self.contract_block(in_channels, filter[0], 7, 3)
            self.conv2 = self.contract_block(filter[0], filter[1], kernelSize, padding)
            self.conv3 = self.contract_block(filter[1], filter[2], kernelSize, padding)

            self.upconv3 = self.expand_block(filter[2], filter[1], kernelSize, padding)
            self.upconv2 = self.expand_block(filter[1] * 2, filter[0], kernelSize, padding)
            self.upconv1 = self.expand_block(filter[0] * 2, out_channels, kernelSize, padding)

    def __call__(self, x):

        if self.moreLayers:
            #downsample
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)

            #upsample
            upconv5 = self.upconv5(conv5)
            upconv4 = self.upconv4(torch.cat([upconv5, conv4], 1))
            upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
            upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
            upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        else:
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)

            # upsample
            upconv3 = self.upconv3(conv3)
            upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
            upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1
    
    #m?? kanskje sj??p?? sequentil ift layers
    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            #torch.nn.ReLU(),
            torch.nn.LeakyReLU(),
            # torch.nn.RReLU(),

            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            #torch.nn.ReLU(),
            torch.nn.LeakyReLU(),
            # torch.nn.RReLU(),

            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            #torch.nn.ReLU(),
                            torch.nn.LeakyReLU(),
                            #torch.nn.RReLU(),

                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            #torch.nn.ReLU(),
                            torch.nn.LeakyReLU(),
                            # torch.nn.RReLU(),

                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                            )
        return expand

