import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np





#log normalization functions
def log_normalization(x):
	x = torch.clamp(x, min = 1e-22, max = None)
	x = (22 + torch.log10(torch.clamp(x/torch.max(x), 1e-22, 1.0)))/22.0
	return x