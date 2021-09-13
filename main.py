# This file starts the training
from torch.cuda import empty_cache
from torch.utils.data import DataLoader # Dataset
from torch.utils.data.sampler import SubsetRandomSampler # RandomSampling
# from torchvision import transforms

import config as c
from train import train
#from utils import load_datasets, make_dataloaders
from data_loader import CowObjectsDataset
from data_loader import ToTensor
from data_loader import train_valid_split # Balanced split


# free up memory for cuda
empty_cache()

# instantiate class
transformed_dataset = CowObjectsDataset(root_dir=c.proj_dir,
                                        transform = ToTensor(),convert_to_points=True,generate_density=True)

train_indices, valid_indices = train_valid_split(dataset = transformed_dataset, train_percent = 70)

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

train_loader = DataLoader(transformed_dataset, batch_size=c.batch_size,shuffle=True, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_density,
                        pin_memory=True,sampler=train_sampler)

valid_loader = DataLoader(transformed_dataset, batch_size=c.batch_size,shuffle=True, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_density,
                        pin_memory=True,sampler=valid_sampler)

model = train(train_loader,valid_loader) 