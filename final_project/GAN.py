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
import models
from models import *
import operator
import data
from data import TopoDataset1
from tensorboardX import SummaryWriter
import tqdm
from tqdm import tqdm


# for reference:
# https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/e9c8374ecc202120dc94db26bf08a00f/dcgan_faces_tutorial.ipynb


def train(device, train_loader, validation_loader, validation_samples, epochs, tensorboard):
	print('Beginning training.')

	#initialize models => call nn.Module -> initialize weights -> send to device for training
	generator = AE(in_channels=2, out_channel=1)
	generator.apply(weights_init_normal)
	generator = generator.to(device)
	discriminator = Discriminator()
	discriminator.apply(weights_init_normal)
	discriminator = discriminator.to(device)

	#initialize optimizers
	opt_generator = torch.optim.Adam(generator.parameters())
	opt_discriminator = torch.optim.Adam(discriminator.parameters())

	#initialize loss function
	criterion = nn.BCELoss()


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

			# ground truth values
			real_disc = discriminator(final_D)
			real = Variable(torch.ones_like(real_disc), requires_grad_=False)
			fake = Variable(torch.zeros_like(real_disc), requires_grad_=False)

			#freeze discriminator
			for p in discriminator.parameters():
				p.requires_grad_(False)

			#zero gradient (generator)
			generator.zero_grad()

			# generator prediction
			pred_D = model(torch.cat((initial_SE, initial_D), 1))

			#calculate generator loss
			generator_loss = criterion(discriminator(pred_D), real)
			# generator_loss = discriminator(pred_D)
			# generator_loss = generator_loss.mean()
			# generator_loss = -generator_loss

			#call backward pass
			generator_loss.backward()

			#take generator's optimization step
			opt_generator.step


			#unfreeze discriminator
			for p in discriminator.parameters():
				p.requires_grad_(True)


			#zero gradient (discriminator)
			discriminator.zero_grad()


			#discriminator forward pass over appropriate inputs
			discriminator_loss_real = criterion(discriminator(final_D), real)
			discriminator_loss_fake = criterion(discriminator(pred_D.detach()), fake) 
			
			# train with fake data
			# disc_fake = discriminator(pred_D)
			# disc_fake = disc_fake.mean()

			# calculate discriminator losses
			discriminator_loss = (discriminator_loss_real + discriminator_loss_fake) / 2

			# call backward pass
			opt_discriminator.backward()

			# take discriminator's optimization step 
			opt_discriminator.step()


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

				# ground truth values
				real_disc = discriminator(final_D)
				real = Variable(torch.ones_like(real_disc), requires_grad_=False)
				fake = Variable(torch.zeros_like(real_disc), requires_grad_=False)


				# generator prediction
				pred_D = model(torch.cat((initial_SE, initial_D), 1))

				#calculate generator loss
				generator_loss = criterion(discriminator(pred_D), real)
				#generator_loss = discriminator(pred_D)
				#generator_loss = generator_loss.mean()
				#generator_loss = -generator_loss


				#discriminator forward pass over appropriate inputs
				discriminator_loss_real = criterion(discriminator(final_D), real)
				discriminator_loss_fake = criterion(discriminator(pred_D.detach()), fake)
				#disc_real = discriminator(final_D)
				#disc_real = disc_real.mean()

				# train with fake data
				#disc_fake = discriminator(pred_D)
				#disc_fake = disc_fake.mean()

				# calculate discriminator losses
				discriminator_loss = (discriminator_loss_real + discriminator_loss_fake) / 2

				#log losses to tensorboard 
				tensorboard.add_scalar('validation/generator_loss', generator_loss, epoch)
				tensorboard.add_scalar('validation/discriminator_loss', discriminator_loss, epoch)


			# plot out some samples from validation
			fig, axs = plt.subplots(len(validation_samples), 4, figsize=(1*4,1*len(validation_samples)),
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
				prediction_D = generator(torch.cat((initial_SE, initial_D), 0).unsqueeze(0))
				if isinstance(prediction_SE, tuple):
					prediction_D = prediction_D[1]
				axs[idx][0].imshow(log_normalization(initial_SE).cpu().detach().squeeze().numpy(), cmap=plt.cm.jet, interpolation='nearest')
				axs[idx][1].imshow((1-initial_D.cpu().detach().squeeze().numpy()), vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')
				axs[idx][2].imshow((1-final_D.cpu().detach().squeeze().numpy()), vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')
				axs[idx][3].imshow((1-prediction_D.cpu().detach().squeeze().numpy()), vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')
			tensorboard.add_figure('Predicted Density', fig, epoch)
		#save training outputs and model checkpoints
			torch.save(generator.state_dict(), os.path.join(output_path, 'generator.pt'))
			torch.save(discriminator.state_dict(), os.path.join(output_path, "discriminator.pt"))








if __name__ == '__main__':

	# set parameters
	epochs = 50
	tensorboard = 'dbug'
	batch_size = 32
	#training parameters
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	#set data path
	data_path = '/data/Aditya/TopologyOptimization/ConvLSTM/topopt/Data/uniform_data_72k'

	#initialize datasets
	train_dataset = TopoDataset1(data_path=data_path, model_type = 'generator', mode='train')
	validation_dataset = TopoDataset1(data_path=data_path, model_type = 'generator', mode='validation')

	#initialize val samples for figures
	samples_ids = [1, 10, 200, 131, 150, 431, 472, 900, 700, 881]
	validation_samples = [validation_dataset[idx] for idx in samples_ids]

	#initialize data loaders
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
	validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

	#output path
	output_path = os.path.join('./logs/' + tensorboard + '/') 

	#create appropriate directory if it DNE
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	#training loop args
	tensorboard = SummaryWriter(output_path)

	print('Training loop is starting')

	# train model
	train(device, train_loader, validation_loader, validation_samples, epochs, tensorboard)

	print('Training loop completed.')
	print('Saving model...')
	print('Model saved.')





