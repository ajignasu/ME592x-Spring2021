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
	data_path = '/data/Aditya/TopologyOptimization/ConvLSTM/topopt/Data/uniform_data_72k'

	#initialize datasets
	train_dataset = TopoDataset1(data_path=data_path, mode='train')
	val_dataset = TopoDataset1(data_path=data_path, mode='validation')

	#initialize val samples for figures
	samples_ids = [1, 10, 200, 131, 150, 431, 472, 900, 700, 881]
	validation_samples = [val_dataset[idx] for idx in samples_ids]

	#initialize data loaders
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
	validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

	#output path
	output_path = os.path.join('./logs/' + args.tensorboard + '/') 

	#create appropriate directory if it DNE
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	#training loop args
	epochs = args.epoch
	tensorboard = SummaryWriter(output_path)

	#initialize model framework
	if args.framework == 'GAN':
		GAN.train(device=device, train_loader=train_loader, validation_loader=validation_loader,
					validation_samples=validation_samples, epochs=epochs, tensorboard=tensorboard)

	elif args.framework == 'wGAN':
		GAN.train(device=device, train_loader=train_loader, validation_loader=validation_loader,
					validation_samples=validation_samples, epochs=epochs, tensorboard=tensorboard)

	elif args.framework == 'pix2pix':
		pix2pix.train(device=device, train_loader=train_loader, validation_loader=validation_loader,
					validation_samples=validation_samples, epochs=epochs, tensorboard=tensorboard)

	elif args.framework == 'patch':
		patchGAN.train(device=device, train_loader=train_loader, validation_loader=validation_loader,
					validation_samples=validation_samples, epochs=epochs, tensorboard=tensorboard)


	print('Training loop completed.')
	print('Saving model...')
	#save training outputs and model checkpoints
	torch.save(generator.state_dict(), os.path.join(output_path, generator+".pt"))
	torch.save(discriminator.state_dict(), os.path.join(output_path, discriminator+".pt"))
	print('Model saved.')



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Compliance Network - topopt invnet')
	parser.add_argument('-frame', '--framework', default='wGAN', type=str,
						help='Which framework would you like to use?')
	parser.add_argument('-board', '--tensorboard', type=str,
						help='Title of tensorboard run.')
	parser.add_argument('-ep', '--epoch', default=100, type=int,
						help='How many epochs would you like to train for?')
	parser.add_argument('-bs', '--batch_size', default=32, type=int)
	hparams = parser.parse_args()
	main(hparams)


