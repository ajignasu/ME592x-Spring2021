import os
import json
import numpy as np
import torch
from torch.utils import data
import torchvision
from tqdm import tqdm


class TopoDataset1(data.Dataset):
	'PyTorch dataset for Topology Optimization'
	
	def __init__(self, data_path, model_type='regressor', mode='train'):
		'Initialization'
		### Identify unique model dataset path
		if model_type == 'regressor':
			self.model_type = os.path.join(data_path,'regressor_data')
		elif model_type == 'generator':
			self.model_type = os.path.join(data_path,'generator_data')

		### Identify dataset subset
		if mode == 'train':
			self.data_path = os.path.join(self.model_type,'train')
		elif mode == 'test':
			self.data_path = os.path.join(self.model_type,'test')
		elif mode == 'validation':
			self.data_path = os.path.join(self.model_type,'validation')


		self.initial_SE = np.load(os.path.join(self.data_path, 'initial_SE.npz'))['arr_0']
		self.initial_D = np.load(os.path.join(self.data_path, 'initial_D.npz'))['arr_0']
		self.final_SE = np.load(os.path.join(self.data_path, 'final_SE.npz'))['arr_0']
		self.final_D = np.load(os.path.join(self.data_path, 'final_D.npz'))['arr_0']      

	def __len__(self):
		'Denotes the total number of samples'
		return self.initial_SE.shape[0]

	def __getitem__(self, index):
		'Generates one sample of data'
		data_dict = {}
		data_dict['initial_SE'] = torch.FloatTensor(self.initial_SE[index])
		data_dict['initial_D'] = torch.FloatTensor(self.initial_D[index])
		data_dict['final_SE'] = torch.FloatTensor(self.final_SE[index])
		data_dict['final_D'] = torch.FloatTensor(self.final_D[index])
		return data_dict