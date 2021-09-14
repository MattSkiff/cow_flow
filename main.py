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

from torch import onnx

# free up memory for cuda
empty_cache()

# instantiate class
transformed_dataset = CowObjectsDataset(root_dir=c.proj_dir,
                                        transform = ToTensor(),convert_to_points=True,generate_density=True)

train_indices, valid_indices = train_valid_split(dataset = transformed_dataset, train_percent = 70)

# Creating data samplers and loaders:
# only train part for dev purposes 
train_sampler = SubsetRandomSampler(train_indices[:round(c.data_prop*len(train_indices))])
valid_sampler = SubsetRandomSampler(valid_indices[:round(c.data_prop*len(valid_indices))])

if c.verbose:
    print("Training using {} train samples and {} validation samples...".format(len(train_sampler),len(valid_sampler)))

train_loader = DataLoader(transformed_dataset, batch_size=c.batch_size,shuffle=False, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_density,
                        pin_memory=True,sampler=train_sampler)

valid_loader = DataLoader(transformed_dataset, batch_size=c.batch_size,shuffle=False, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_density,
                        pin_memory=True,sampler=valid_sampler)

model = train(train_loader,valid_loader) 

if False:
    # Input to the model
    dummy_image = randn(c.batch_size,3,c.density_map_h ,c.density_map_w, requires_grad=True).to(c.device) # (RGB = 3)
    dummy_dmaps = randn(c.batch_size,c.density_map_h,c.density_map_w, requires_grad=True).to(c.device) 
    torch_out = model(dummy_image,dummy_dmaps)
    
    torch_out[0].size() # z
    torch_out[1].size() # log_det_jac
    
    # export the model
    onnx.export(model, dummy_image, c.modelname, export_params=True, opset_version=10,do_constant_folding=True)
    
    # load the onnx model
    model = onnx.load(c.modelname)
    onnx.helper.printable_graph(model.graph)
    
    z = randn(c.batch_size,4,c.density_map_h // 2,c.density_map_w // 2)
    
    samples, _ = model(z,transformed_dataset[5809]['image'],rev=True)