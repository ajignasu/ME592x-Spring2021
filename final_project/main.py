import os, sys, argparse
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
from models import *
from trainers import *
import operator
import data
from data import TopoDataset1


def main(args):
	#training parameters
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	#set data path
	data_path = '/data/Aditya/TopologyOptimization/ConvLSTM/topopt/Data/uniform_data_72k/surrogate_data'

	#initialize model architecture
	if args.architecture == 'AE':
		model = AE(in_channels=2, out_channels=1)
	elif args.architecture == 'unet':
		model = UNet(in_channels=2, out_channels=1)
	elif args.architecture == 'psp':
		model = UNetPSP()

	#send model to GPU
	model = model.to(device)

	#initialize model weights
	model.apply(weights_init)

	#initialize datasets
	train_dataset = TopoDatasetSurrogate(data_path=data_path, mode='train')
	val_dataset = TopoDatasetSurrogate(data_path=data_path, mode='validation')

	#initialize val samples for figures
	samples_ids = [1, 10, 200, 131, 150, 431, 472, 900, 700, 881]
	val_samples = [val_dataset[idx] for idx in samples_ids]

	#initialize data loaders
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

	#tensorboard
	board = args.tensorboard

	#output path
	output_path = os.path.join('./surrogate_models/' + board + '/') 

	#create appropriate directory if it DNE
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	tensorboard = SummaryWriter(output_path)

	#training loop
	epochs = args.epoch
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	print("# of params in model: ", sum(a.numel() for a in model.parameters()))
	if args.surrogate_model == 'density':
		for i in range(epochs):
			print('beginning epoch ', i+1)
			print(training_epoch_D(device, model, train_loader, optimizer, tensorboard, epoch=i))
			validation_epoch_D(device, model, val_loader, val_samples, tensorboard, epoch=i)

	elif args.surrogate_model == 'compliance':
		for i in range(epochs):
			print('beginning epoch ', i+1)
			print(training_epoch_SE(device, model, train_loader, optimizer, tensorboard, epoch=i))
			validation_epoch_SE(device, model, val_loader, val_samples, tensorboard, epoch=i)

	print('Training loop completed.')
	print('Saving model...')
	#save training outputs and model checkpoints
	torch.save(model.state_dict(), os.path.join(output_path, args.surrogate_model+".pt"))
	print('Model saved.')



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Compliance Network - topopt invnet')
	parser.add_argument('-arch', '--architecture', default='AE', type=str,
						help='Which architecture would you like to use?')
	# parser.add_argument('-dp', '--data_path', default='scs', type=str,
	#                   help='Which computing cluster are you using?')
	parser.add_argument('-board', '--tensorboard', type=str,
						help='Title of tensorboard run.')
	# parser.add_argument('-exp_id', '--experiment_id', type=str,
	#                   help='Experiment ID for various runs.')
	parser.add_argument('-ep', '--epoch', default=100, type=int,
						help='How many epochs would you like to train for?')
	parser.add_argument('-bs', '--batch_size', default=32, type=int)
	parser.add_argument('-model', '--surrogate_model', default='density', type=str,
						help='which surrogate are you training?')
	hparams = parser.parse_args()
	main(hparams)


