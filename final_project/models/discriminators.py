import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



##############################
#        Discriminator

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, *args):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat(args, 1)
        return self.model(img_input)


class Discriminator_OT(nn.Module):
    def __init__(self, in_channels=2):
        super(Discriminator_OT, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def discriminator_block_even(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            *discriminator_block_even(512, 128),
            *discriminator_block_even(128, 64),
            *discriminator_block_even(64, 16)
        )

        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.linear_1 = nn.Linear(400, 50)
        self.linear_2 = nn.Linear(50, 1)

    def forward(self, *args):
        x = torch.cat(args,1)
        x = self.model(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.leaky(self.linear_1(x))
        return self.linear_2(x)


class Discriminator_wGAN(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator_wGAN, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def discriminator_block_even(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            *discriminator_block_even(512, 128),
            *discriminator_block_even(128, 64),
            *discriminator_block_even(64, 16)
        )

        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.linear_1 = nn.Linear(400, 50)
        self.linear_2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.leaky(self.linear_1(x))
        return self.linear_2(x)