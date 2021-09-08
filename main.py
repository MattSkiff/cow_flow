# This file starts the training
from torch.cuda import empty_cache
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config as c
from train import train
#from utils import load_datasets, make_dataloaders
from data_loader import CowObjectsDataset
from data_loader import ToTensor

# free up memory for cuda
empty_cache()

# instantiate class
transformed_dataset = CowObjectsDataset(root_dir=c.proj_dir,
                                        transform = ToTensor(),convert_to_points=True,generate_density=True)

#train_set, test_set = load_datasets(c.dataset_path, c.class_name)
dataloader = DataLoader(transformed_dataset, batch_size=1,shuffle=True, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_density)

#train_loader, test_loader = make_dataloaders(train_set, test_set)

#model = train(train_loader, test_loader)
model = train(dataloader) 