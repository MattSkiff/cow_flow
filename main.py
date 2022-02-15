# This file starts the training on the cows dataset
from torchvision.transforms import Compose
from torch.cuda import empty_cache
from torch.utils.data import DataLoader # Dataset                                                                                                                                                                    
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler # RandomSampling

# from torchvision import transforms
import config as c
import arguments as a
import model
import os
from train import train, train_battery

from ipywidgets import FloatProgress # fix iprogress error in tqdm

#from utils import load_datasets, make_dataloaders
from data_loader import CowObjectsDataset, CustToTensor, AerialNormalize, DmapAddUniformNoise, CustCrop, CustResize, train_val_split

from utils import plot_preds, plot_peaks

empty_cache() # free up memory for cuda

# torchivsion inputs are 3x227x227, mnist_resnet 1x227...
# 0.1307, 0.3081 = mean, std dev mnist

dmaps_pre = Compose([
            CustToTensor(),
            AerialNormalize(),
            CustResize(),
            CustCrop(),
            DmapAddUniformNoise(),
        ])
                    
# instantiate class
transformed_dataset = CowObjectsDataset(root_dir=c.proj_dir,transform = dmaps_pre,
                                        convert_to_points=True,generate_density=True,
                                        count = c.counts, 
                                        classification = True,ram=c.ram)

# check dataloader if running interactively
if any('SPYDER' in name for name in os.environ):
    transformed_dataset.show_annotations(5895)

# create test train split
t_indices, t_weights, v_indices, v_weights  = train_val_split(dataset = transformed_dataset,
                                                  train_percent = c.test_train_split,
                                                  annotations_only = c.annotations_only,
                                                  balanced = True,seed = c.seed)

f_t_indices, f_t_weights, f_v_indices, f_v_weights  = train_val_split(dataset = transformed_dataset,
                                                  train_percent = c.test_train_split,
                                                  annotations_only = False,
                                                  balanced = True,seed = c.seed)

# Creating data samplers and loaders:
# only train part for dev purposes 

if not c.annotations_only:
    train_sampler = SubsetRandomSampler(t_indices)
    val_sampler = SubsetRandomSampler(v_indices)

if c.annotations_only:
    train_sampler = SubsetRandomSampler(t_indices)
    val_sampler = SubsetRandomSampler(v_indices)    

if c.weighted:
    # the weight sizes correspond to whether each indices 0...5900 is null-annotated or not
    # the weights correspond to the probability that that indice is sampled, they don't have to sum to one
    train_sampler = WeightedRandomSampler(weights=t_weights,
                                          num_samples=len(t_weights),
                                          replacement=True)
    val_sampler = WeightedRandomSampler(weights=v_weights,
                                        num_samples=len(v_weights),
                                        replacement=True)

if len(c.batch_size) != 1 or len(c.lr_init) != 1 and a.args.feat_extract_only:
    ValueError('Training batteries not available for Feature Extractor only runs')
    
if len(c.batch_size) == 1:
    # CPU tensors can't be pinned; leave false
    
    full_train_sampler = SubsetRandomSampler(f_t_indices)
    full_val_sampler = SubsetRandomSampler(f_v_indices) 
    # TODO - fix random sampling !
    # full_train_sampler = WeightedRandomSampler(weights=f_t_weights,
    #                           num_samples=len(f_t_weights),
    #                           replacement=True)
                
    # full_val_sampler = WeightedRandomSampler(weights=f_v_weights,
    #                                     num_samples=len(f_v_weights),
    #                                     replacement=True)
    
    full_train_loader = DataLoader(transformed_dataset, batch_size=c.batch_size[0],shuffle=False, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_aerial,
                        pin_memory=False,sampler=full_train_sampler)

    full_val_loader = DataLoader(transformed_dataset, batch_size=c.batch_size[0],shuffle=False, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_aerial,
                        pin_memory=False,sampler=full_val_sampler)
    
    train_loader = DataLoader(transformed_dataset, batch_size=c.batch_size[0],shuffle=False, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_aerial,
                        pin_memory=False,sampler=train_sampler)

    val_loader = DataLoader(transformed_dataset, batch_size=c.batch_size[0],shuffle=False, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_aerial,
                        pin_memory=False,sampler=val_sampler)
    
    if len(c.lr_init) == 1:
        if a.args.feat_extract_only:
            feat_extractor = model.select_feat_extractor(c.feat_extractor,train_loader,val_loader)
        else:
            mdl = train(train_loader,val_loader,full_train_loader,full_val_loader,lr_i=c.lr_init)
    else:
        mdl = train_battery([train_loader],[val_loader],lr_i=c.lr_init)
            
else:
    tls,vls = [],[]
    
    for bs in c.batch_size:
        tls.append(DataLoader(transformed_dataset, batch_size=bs,shuffle=False, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_aerial,
                        pin_memory=True,sampler=train_sampler))
        
        vls.append(DataLoader(transformed_dataset, batch_size=bs,shuffle=False, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_density,
                        pin_memory=True,sampler=val_sampler))
        
        mdl = train_battery(tls,vls,lr_i=c.lr_init)
        
plot_preds(mdl,train_loader)
plot_peaks(mdl,train_loader)