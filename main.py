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
from train import train, train_baselines, train_feat_extractor

from dlr_acd import train_dlr_acd
from mnist import train_mnist

from ipywidgets import FloatProgress # fix iprogress error in tqdm

#from utils import load_datasets, make_dataloaders
from data_loader import CowObjectsDataset, CustToTensor, AerialNormalize, DmapAddUniformNoise, CustCrop, CustResize, train_val_split, CropRotateFlipScaling

from utils import plot_preds, plot_peaks

empty_cache() # free up memory for cuda

if a.args.dlr_acd:
    mdl, train_loader, val_loader = train_dlr_acd()

if a.args.mnist:
    mdl, train_loader, val_loader = train_mnist()

if a.args.cows:
    # torchivsion inputs are 3x227x227, mnist_resnet 1x227...
    # 0.1307, 0.3081 = mean, std dev mnist
    
    dmaps_pre = Compose([
                CustToTensor(),
                AerialNormalize(),
                CropRotateFlipScaling(),
                #CustResize(),
                #CustCrop(),
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
                                                      annotations_only = a.args.annotations_only,
                                                      balanced = a.args.balance,seed = c.seed,
                                                      oversample=a.args.weighted_sampler)
    
    f_t_indices, f_t_weights, f_v_indices, f_v_weights  = train_val_split(dataset = transformed_dataset,
                                                      train_percent = c.test_train_split,
                                                      annotations_only = False,
                                                      balanced = a.args.balance,seed = c.seed,
                                                      oversample=a.args.weighted_sampler)
    
    if True:#a.args.annotations_only or a.args.balance:
        train_sampler = SubsetRandomSampler(t_indices)
        val_sampler = SubsetRandomSampler(v_indices)
    else:
        ValueError
        train_sampler = WeightedRandomSampler(weights=t_weights,
                                              num_samples=len(t_weights),
                                              replacement=True)
        
        val_sampler = WeightedRandomSampler(weights=v_weights,
                                            num_samples=len(v_weights),
                                            replacement=True)    
    
    if False:#a.args.weighted_sampler:
        # the weight sizes correspond to whether each indices 0...5900 is null-annotated or not
        # the weights correspond to the probability that that indice is sampled, they don't have to sum to one
        ValueError # remove when balanced working correctly
        full_train_sampler = WeightedRandomSampler(weights=t_weights,
                                              num_samples=len(t_weights),
                                              replacement=True)
        
        full_val_sampler = WeightedRandomSampler(weights=v_weights,
                                            num_samples=len(v_weights),
                                            replacement=True)                    
    
    else:
        # using balanced samplers cuts dataset from 6k to 800 patches
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
    
    if a.args.feat_extract_only:
        feat_extractor = model.select_feat_extractor(c.feat_extractor,train_loader,val_loader)
        train_feat_extractor(feat_extractor,train_loader,val_loader)
    else:
        if a.args.model_name in ['CSRNet', 'UNet', 'FCRN']:
            mdl = train_baselines(a.args.model_name,train_loader,val_loader)
        else:
            mdl = train(train_loader,val_loader,full_train_loader,full_val_loader)
                