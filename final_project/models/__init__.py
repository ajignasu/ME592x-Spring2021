import torch
from .generators import *
from .discriminators import *

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


__all__ = ['Encoder', 'Decoder', 'AE', 'VAE',
            'Discriminator_wGAN', 'weights_init_normal']