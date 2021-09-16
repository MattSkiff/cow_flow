# This file starts the training
from torch.cuda import empty_cache
from torch.utils.data import DataLoader # Dataset
from torch.utils.data.sampler import SubsetRandomSampler # RandomSampling
from torch import randn
# from torchvision import transforms

import config as c
from train import train

#from utils import load_datasets, make_dataloaders
from data_loader import CowObjectsDataset
from data_loader import ToTensor
from data_loader import train_valid_split # Balanced split

empty_cache() # free up memory for cuda

# instantiate class
transformed_dataset = CowObjectsDataset(root_dir=c.proj_dir,
                                        transform = ToTensor(),convert_to_points=True,generate_density=True)

# create test train split
train_indices, valid_indices = train_valid_split(dataset = transformed_dataset, train_percent = 70,annotations_only = c.annotations_only)

# TODO: code to save this file (train and valid indices)

# Creating data samplers and loaders:
# only train part for dev purposes 

if not c.annotations_only:
    train_sampler = SubsetRandomSampler(train_indices[:round(c.data_prop*len(train_indices))])
    valid_sampler = SubsetRandomSampler(valid_indices[:round(c.data_prop*len(valid_indices))])

if c.annotations_only:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

if c.verbose:
    print("Training using {} train samples and {} validation samples...".format(len(train_sampler)*c.batch_size,len(valid_sampler)*c.batch_size))

train_loader = DataLoader(transformed_dataset, batch_size=c.batch_size,shuffle=False, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_density,
                        pin_memory=True,sampler=train_sampler)

valid_loader = DataLoader(transformed_dataset, batch_size=c.batch_size,shuffle=False, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_density,
                        pin_memory=True,sampler=valid_sampler)

model = train(train_loader,valid_loader) 