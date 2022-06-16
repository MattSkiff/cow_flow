# external
from torch.cuda import empty_cache
from torch.utils.data import DataLoader # Dataset

import torch.nn as nn
import os

# internal
import config as c
import arguments as a
import gvars as g
import model
from data_loader import prep_transformed_dataset, make_loaders
from train import train, train_baselines, train_feat_extractor, train_classification_head
from utils import load_model,plot_preds,plot_preds_baselines,plot_preds_multi
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

    transformed_dataset = prep_transformed_dataset()
    
    # check dataloader if running interactively
    if any('SPYDER' in name for name in os.environ):
        transformed_dataset.show_annotations(66) # 5895
    
    if not a.args.holdout:
        # create test train split
        full_train_loader, full_val_loader, train_loader, val_loader = make_loaders(transformed_dataset)
    else:
        val_loader = DataLoader(transformed_dataset, batch_size=a.args.batch_size,shuffle=True, 
                            num_workers=4,collate_fn=transformed_dataset.custom_collate_aerial,
                            pin_memory=True)
    
    if a.args.mode in ['plot','eval']:
        
        mdl = load_model(a.args.mdl_path)
    
        if a.args.mode == 'eval':
            
            if a.args.model_name == 'NF':
                dmap_metrics(mdl,val_loader,mode='val',n=50)
            else:
                eval_baselines(mdl,val_loader,mode='val',is_unet_seg=(a.args.model_name=='UNet_seg'))
                #eval_baselines(mdl,train_loader,mode='train',is_unet_seg=(a.args.model_name=='UNet_seg')
    
        if a.args.mode == 'plot':  
    
            if a.args.model_name == 'NF':
                plot_preds(mdl,val_loader)
            elif a.args.model_name == 'ALL':
                plot_preds_multi(mode='val',loader=val_loader)
            elif a.args.model_name in g.BASELINE_MODEL_NAMES:
                plot_preds_baselines(mdl, val_loader,mode="val",mdl_type=a.args.model_name)
            
    if a.args.mode == 'train' and not a.args.holdout:
        
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