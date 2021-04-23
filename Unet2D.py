import torch
from torch import nn







class Unet2D(nn.Module):

    #added one deeper lvl of conv layers
    def __init__(self, in_channels, out_channels):
        super().__init__()

        filter = [32, 64, 128, 256, 512]

        self.conv1 = self.contract_block(in_channels, filter[0], 7, 3)
        self.conv2 = self.contract_block(filter[0], filter[1], 3, 1)
        self.conv3 = self.contract_block(filter[1], filter[2], 3, 1)
        self.conv4 = self.contract_block(filter[2], filter[3], 3, 1)
        self.conv5 = self.contract_block(filter[3], filter[4], 3, 1)

        self.upconv5 = self.expand_block(filter[4], filter[3], 3, 1)
        self.upconv4 = self.expand_block(filter[3], filter[2], 3, 1)
        self.upconv3 = self.expand_block(filter[2], filter[1], 3, 1)
        self.upconv2 = self.expand_block(filter[1], filter[0], 3, 1)
        self.upconv1 = self.expand_block(filter[0], out_channels, 3, 1)

    def __call__(self, x):
        # downsampling part
        #print(size(x))
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        #upsample
        upconv5 = self.upconv5(conv4)
        upconv4 = self.upconv4(torch.cat([upconv5, conv4]), 1)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3]), 1)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))


        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),

            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            #torch.nn.ReLU(),
            torch.nn.LeakyReLU(),

            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),

                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.LeakyReLU(),

                            #torch.nn.ReLU(),

                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                            )
        return expand
