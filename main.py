# external
from torch.cuda import empty_cache
from torch.utils.data import DataLoader # Dataset                                                                                                                                                                    
from torch.utils.data.sampler import SubsetRandomSampler # RandomSampling
import torch.nn as nn
import os

# internal
import config as c
import arguments as a
import model
from data_loader import CustToTensor, AerialNormalize, DmapAddUniformNoise, train_val_split, Resize, RotateFlip, CustResize, prep_transformed_dataset
from train import train, train_baselines, train_feat_extractor, train_classification_head
from utils import load_model
from eval import dmap_metrics, eval_baselines
from dlr_acd import dlr_acd
from mnist import mnist



empty_cache() # free up memory for cuda

if a.args.data == 'dlr':
    mdl, train_loader, val_loader = dlr_acd()

if a.args.data == 'mnist':
    mdl, train_loader, val_loader = mnist()

if a.args.data == 'cows':
    # torchivsion inputs are 3x227x227, mnist_resnet 1x227...
    # 0.1307, 0.3081 = mean, std dev mnist
    
    transforms = [CustToTensor()]
    
    if a.args.normalise:
        transforms.append(AerialNormalize())
    
    if a.args.resize:
        transforms.append(Resize())
    
    if not a.args.resize:
        transforms.append(CustResize())
    
    #if a.args.rrc:
    transforms.extend([DmapAddUniformNoise(),RotateFlip(),])

    transformed_dataset = prep_transformed_dataset(transforms)
    
    # check dataloader if running interactively
    if any('SPYDER' in name for name in os.environ):
        transformed_dataset.show_annotations(5895) #
        #transformed_dataset.show_annotations(0) #
    
    # create test train split
    t_indices, t_weights, v_indices, v_weights  = train_val_split(dataset = transformed_dataset,
                                                      train_percent = c.test_train_split,
                                                      annotations_only = (a.args.sampler == 'anno'),
                                                      seed = c.seed,
                                                      oversample= (a.args.sampler == 'weighted'))
    
    f_t_indices, f_t_weights, f_v_indices, f_v_weights  = train_val_split(dataset = transformed_dataset,
                                                      train_percent = c.test_train_split,
                                                      annotations_only = False,
                                                      seed = c.seed,
                                                      oversample=False)
    
    train_sampler = SubsetRandomSampler(t_indices)
    val_sampler = SubsetRandomSampler(v_indices)
  
    full_train_sampler = SubsetRandomSampler(f_t_indices)
    full_val_sampler = SubsetRandomSampler(f_v_indices) 
    
    # leave shuffle off for use of any samplers
    full_train_loader = DataLoader(transformed_dataset, batch_size=a.args.batch_size,shuffle=False, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_aerial,
                        pin_memory=False,sampler=full_train_sampler)

    full_val_loader = DataLoader(transformed_dataset, batch_size=a.args.batch_size,shuffle=False, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_aerial,
                        pin_memory=False,sampler=full_val_sampler)
    
    train_loader = DataLoader(transformed_dataset, batch_size=a.args.batch_size,shuffle=False, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_aerial,
                        pin_memory=False,sampler=train_sampler)

    val_loader = DataLoader(transformed_dataset, batch_size=a.args.batch_size,shuffle=False, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_aerial,
                        pin_memory=False,sampler=val_sampler)
    
    if a.args.mode == 'eval':
        
        mdl = load_model(a.args.mdl_path)
        
        if a.args.model_name == 'NF':
            dmap_metrics(mdl,val_loader,mode='val',n=50)
        else:
            eval_baselines(mdl,val_loader,mode='val',is_unet_seg=(a.args.model_name=='UNet_seg'))
        
    if a.args.mode == 'train':
        
        if a.args.bc_only:
            train_classification_head(None,train_loader,val_loader,criterion = nn.CrossEntropyLoss())
        
        if a.args.fe_only:
            feat_extractor = model.select_feat_extractor(c.feat_extractor,train_loader,val_loader)
            train_feat_extractor(feat_extractor,train_loader,val_loader)
        else:
            if a.args.model_name == 'NF':
                mdl = train(train_loader,val_loader,full_train_loader,full_val_loader)
            else:
                mdl = train_baselines(a.args.model_name,train_loader,val_loader)