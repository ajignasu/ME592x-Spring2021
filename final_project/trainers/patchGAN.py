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
from models import *
import operator
import data
from data import TopoDataset1
import utils
from utils import *
from tensorboardX import SummaryWriter


# for reference:
# https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/e9c8374ecc202120dc94db26bf08a00f/dcgan_faces_tutorial.ipynb


def train(device, train_loader, validation_loader, validation_samples, epochs, tensorboard):
	print('Beginning training.')

	#initialize models => call nn.Module -> initialize weights -> send to device for training
	generator = # call and initialize appropriate model here
	generator.apply(weights_init)
	generator = generator.to(device)
	discriminator = # call and initialize appropriate model here
	discriminator.apply(weights_init)
	discriminator = discriminator.to(device)

	#initialize optimizers
	opt_generator = # select optimizer type of optimizer matters especially for different GANs
	opt_discriminator = # select optimizer type of optimizer matters especially for different GANs

	#initialize tensorboard
	writer = SummaryWriter(tensorboard)

	#iterate through epochs
	for epoch in range(epochs):

		#initialize losses
		running_gen_loss, running_dis_loss = 0.0, 0.0

		print('Beginning epoch ', epoch+1)

		for idx, batch in enumerate(train_loader):

			#load minibatch
			initial_SE = batch['initial_SE']
			initial_D = batch['initial_D']
			final_SE = batch['final_SE']
			final_D = batch['final_D']

			#send minibatch to GPU for computation
			initial_SE = initial_SE.to(device)
			initial_D = initial_D.to(device)
			final_SE = final_SE.to(device)
			final_D = final_D.to(device)

			#freeze discriminator
			for p in discriminator.parameters():
				p.requires_grad_(False)

			#zero gradient (generator)
			#insert code here

			# generator prediction
			pred_D = model(torch.cat((initial_SE, initial_D), 1))

			#calculate generator loss
			#insert code here

			#call backward pass
			#insert code here

			#take generator's optimization step
			#insert code here



			#unfreeze discriminator


			#zero gradient (discriminator)


			#discriminator forward pass over appropriate inputs


			# calculate discriminator losses


			# call backward pass


			# take discriminator's optimization step 



			#log losses to tensorboard 
			tensorboard.add_scalar('training/generator_loss', generator_loss, epoch)
			tensorboard.add_scalar('training/discriminator_loss', discriminator_loss, epoch)


		# evaluate validation set
		# disable autograd engine
		with torch.no_grad():
			#iterate through validation set
			for idx, batch in enumerate(validation_loader):

				#load minibatch
				initial_SE = batch['initial_SE']
				initial_D = batch['initial_D']
				final_SE = batch['final_SE']
				final_D = batch['final_D']

				#send minibatch to GPU for computation
				initial_SE = initial_SE.to(device)
				initial_D = initial_D.to(device)
				final_SE = final_SE.to(device)
				final_D = final_D.to(device)


				# generator prediction
				pred_D = model(torch.cat((initial_SE, initial_D), 1))

				#calculate generator loss
				#insert code here


				#discriminator forward pass over appropriate inputs


				# calculate discriminator losses


				#log losses to tensorboard 
				tensorboard.add_scalar('validation/generator_loss', generator_loss, epoch)
				tensorboard.add_scalar('validation/discriminator_loss', discriminator_loss, epoch)


			# plot out some samples from validation
			fig, axs = plt.subplots(len(val_samples), 4, figsize=(1*4,1*len(val_samples)),
							subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
			fig.suptitle('Generated Topology Optimization SE predictions')
			for ax_row in axs:
				for ax in ax_row:
					ax.set_xticks([])
					ax.set_yticks([])

			for idx, sample in enumerate(validation_samples):
				initial_SE = sample['initial_SE'].type_as(next(model.parameters()))
				final_SE = sample['final_SE'].type_as(next(model.parameters()))
				final_D = sample['final_D'].type_as(next(model.parameters()))
				prediction_D = model(torch.cat((initial_SE, initial_D), 0).unsqueeze(0))
				if isinstance(prediction_SE, tuple):
					prediction_D = prediction_D[1]
				axs[idx][0].imshow(log_normalization(initial_SE).cpu().detach().squeeze().numpy(), cmap=plt.cm.jet, interpolation='nearest')
				axs[idx][1].imshow((1-initial_D.cpu().detach().squeeze().numpy()), vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')
				axs[idx][2].imshow((1-final_D.cpu().detach().squeeze().numpy()), vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')
				axs[idx][3].imshow((1-prediction_D.cpu().detach().squeeze().numpy()), vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')
			tensorboard.add_figure('Predicted Density', fig, epoch)







			





