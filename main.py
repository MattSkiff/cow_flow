# This file starts the training
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config as c
from train import train
#from utils import load_datasets, make_dataloaders
from data_loader import CowObjectsDataset
from data_loader import ToTensor

# instantiate class
transformed_dataset = CowObjectsDataset(root_dir=c.proj_dir,
                                        transform = ToTensor())

#train_set, test_set = load_datasets(c.dataset_path, c.class_name)
dataloader = DataLoader(transformed_dataset, batch_size=c.batch_size,shuffle=True, num_workers=0,collate_fn=transformed_dataset.custom_collate_fn)

#train_loader, test_loader = make_dataloaders(train_set, test_set)

#model = train(train_loader, test_loader)
model = train(dataloader)