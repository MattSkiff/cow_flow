from tqdm import tqdm
import numpy as np
from skimage.feature import peak_local_max # 
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

import random
import os
import time 
import torch
from torch import randn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utils import ft_dims_select, np_split,is_baseline, UnNormalize, create_point_map,loader_check
import config as c
import gvars as g
import arguments as a

from data_loader import CustToTensor, AerialNormalize, DmapAddUniformNoise, train_val_split, Resize, RotateFlip, CustResize, prep_transformed_dataset
from torch.utils.data import DataLoader # Dataset   

from lcfcn import lcfcn_loss

MAX_DISTANCE = 100
STEP = 1

def eval_dataloaders(mdl):
    
    transforms = [CustToTensor()]
    transforms.append(AerialNormalize())
    
    if mdl.density_map_h == 256:
        transforms.append(Resize())
    else:
        transforms.append(CustResize())
    
    transformed_dataset = prep_transformed_dataset(transforms)
    
    dataloader = DataLoader(transformed_dataset, batch_size=a.args.batch_size,shuffle=False, 
                        num_workers=0,collate_fn=transformed_dataset.custom_collate_aerial,
                        pin_memory=False)
    
    return dataloader

def gen_metrics(dm_mae,dm_mse,dm_ssim,dm_psnr,y_n,y_hat_n,game,gampe,localisation_dict,mode):
    
    # dmap metrics (average across images)
    dm_mae = np.mean(np.vstack(dm_mae))
    dm_mse = np.mean(np.vstack(dm_mse))
    dm_ssim = np.mean(np.vstack(dm_ssim))
    dm_psnr = np.mean(np.vstack(dm_psnr))
    
    # R2 
    n = len(y_n)
    y_hat_n = np.array(y_hat_n)
    y_n = np.array(y_n)
    
    y_bar = y_n.mean() 
    
    ss_res = np.sum((y_n-y_hat_n)**2)
    ss_tot = np.sum((y_n-y_bar)**2) 
    r2 = round(1 - (ss_res / (ss_tot + 1e-8)),4)
    
    # RMSE
    rmse = round(sum(((y_n-y_hat_n)/n)**2)**0.5,4)
    
    # MAE
    mae = round(sum(np.abs(y_n-y_hat_n))/n,4)
    
    # MAPE
    mape = round(sum(np.abs(y_n-y_hat_n)/np.maximum(np.ones(len(y_n)),y_n))/n,4)
    
   # GAME
    game = np.mean(np.vstack(game))
    
   # GAMPE
    gampe = np.mean(np.vstack(gampe))   
    
    # localisation metrics: PR, RC, F1
    tp = localisation_dict['tp']
    fp = localisation_dict['fp']
    fn = localisation_dict['fn']
    
    if tp+fp == 0:
        pr = -99
    else:
        pr = tp/(tp+fp)
    
    if (tp+fn) == 0:
        rc = -99
    else:
        rc = tp/(tp+fn)
    
    if (tp+0.5*(fp+fn)) == 0:
        fscore = -99
    else:
        fscore = tp/(tp+0.5*(fp+fn))
    
    metric_dict = {
        
        # Counts
        '{}_r2'.format(mode):r2,
        '{}_rmse'.format(mode):rmse,
        '{}_mae'.format(mode):mae,
        '{}_mape'.format(mode):mape,
        # density maps
        '{}_dm_mae'.format(mode):dm_mae,
        '{}_dm_mse'.format(mode):dm_mse,
        '{}_dm_psnr'.format(mode):dm_psnr,
        '{}_dm_ssim'.format(mode):dm_ssim,
        
        # localised counting performance
        '{}_game'.format(mode):game,
        '{}_gampe'.format(mode):gampe,
        
        # localisation
        '{}_fscore'.format(mode):fscore,
        '{}_precision'.format(mode):pr,
        '{}_recall'.format(mode):rc
        
        }
    
    if a.args.mode == 'eval':
        print(metric_dict)
    
    return metric_dict
    
@torch.no_grad()
def eval_baselines(mdl,loader,mode,is_unet_seg=False):
    
    thres=4*2 #mdl.sigma*2
    
    loader_check(mdl=mdl,loader=loader)
    
    assert not mdl.count
    assert c.data_prop == 1
    assert mode in ['train','val']
    assert is_baseline(mdl)
   
    print("Dmap Evaluation....")
    t1 = time.perf_counter()
    
    localisation_dict = {
        'tp':0,
        'fp':0,
        'fn':0
        }
    
    y_n = []; y_coords = []
    y_hat_n = []; y_hat_coords = []
    dm_mae = []; dm_mse = []; dm_psnr = []; dm_ssim = []
    game = []; gampe = []
    
    mdl = mdl.to(c.device)
    
    for i, data in enumerate(tqdm(loader, disable=False)):
        
        images,dmaps,labels, binary_labels , annotations, point_maps = data
        images = images.float().to(c.device)
        
        x = mdl(images)
        
        if str(type(mdl)) == "<class 'baselines.LCFCN'>":# or is_unet_seg:
            x = x.sigmoid().cpu().numpy() # logits -> probs
            blobs = lcfcn_loss.get_blobs(probs=x)
            blob_counts = (np.unique(blobs)!=0).sum()
            pred_points = lcfcn_loss.blobs2points(blobs).squeeze()
            
        for idx in range(images.size()[0]):               
            
            if str(type(mdl)) == "<class 'baselines.LCFCN'>":# or is_unet_seg:
                dmap_np = x[idx].squeeze()
            else:
                dmap_np = x[idx].squeeze().cpu().detach().numpy()
                
            dmap_np = dmap_np/mdl.dmap_scaling
            pred_count = dmap_np.sum() # sum across channel, spatial dims for counts
            
            # We 'DONT' evaluate using the GT dmap
            ground_truth_dmap = dmaps[idx].squeeze().cpu().detach().numpy()
            
            # if pred_count > 0:
            #     thres = int(pred_count)
                       
            # add points onto basemap
            
            # TODO- retrieve points from point annotated masks
            # anno = annotations[idx].cpu().detach().numpy()
            
            gt_coords = annotations[idx]
            
            if gt_coords.nelement() != 0:
                gt_coords = torch.stack([gt_coords[:,2],gt_coords[:,1]]).cpu().detach().numpy()
            else:
                gt_coords = None
                
            ground_truth_point_map, _ = create_point_map(mdl=mdl,annotations=annotations[idx].cpu().detach().numpy()) 
            gt_count = ground_truth_dmap.sum() #.round()
            
            # subtract constrant for uniform noise
            constant = ((mdl.noise)/2)*ground_truth_dmap.shape[0]*ground_truth_dmap.shape[1] 
            loader_noise = ((a.args.noise)/2)*ground_truth_dmap.shape[0]*ground_truth_dmap.shape[1] 
            
            pred_count -= constant
            gt_count -= loader_noise
		
            if is_unet_seg:
                n_peaks=np.inf
            else:
                n_peaks=max(1,int(pred_count))
            
            if str(type(mdl)) == "<class 'baselines.LCFCN'>":
                coordinates = np.argwhere(pred_points != 0)
            elif is_unet_seg:
                coordinates = peak_local_max(dmap_np.squeeze(),min_distance=1,
                                             num_peaks=np.inf,threshold_abs=0,threshold_rel=0.5)
                pred_count = len(coordinates)
            elif str(type(mdl))  in ["<class 'baselines.CSRNet'>"]:
                coordinates = np.array([[0,0]])
            else:
                coordinates = peak_local_max(dmap_np,min_distance=int(mdl.sigma*2),num_peaks=n_peaks)
                
            #y.append(labels[idx].cpu().detach().numpy())
            y_n.append(len(labels[idx]))
            
            if gt_coords is not None:
                y_coords.append(gt_coords)
                
            if str(type(mdl)) == "<class 'baselines.LCFCN'>":# or is_unet_seg:
                y_hat_n.append(blob_counts)
            else:
                y_hat_n.append(pred_count)
            
            y_hat_coords.append(coordinates) 
            
            # local-count metrics # TODO
            l = 1 # cell size param - number of cells to split images into: 1 = 16 
            
            # this splits the density maps into cells for counting per cell
            gt_dmap_split_counts = np_split(ground_truth_point_map,nrows=mdl.density_map_w//4**l,ncols=mdl.density_map_h//4**l).sum(axis=(1,2))
            pred_dmap_split_counts = np_split(dmap_np,nrows=mdl.density_map_w//4**l,ncols=mdl.density_map_h//4**l).sum(axis=(1,2))
            
            game.append(sum(abs(pred_dmap_split_counts-gt_dmap_split_counts)))
            gampe.append(sum(abs(pred_dmap_split_counts-gt_dmap_split_counts)/np.maximum(np.ones(len(gt_dmap_split_counts)),gt_dmap_split_counts)))  
            
            # dmap metrics (kernalised dmap   
            dm_mae.append(sum(abs(dmap_np-ground_truth_dmap)))
            dm_mse.append(sum(np.square(dmap_np-ground_truth_dmap)))
            
            #pdmap_divisor = (np.max(dmap_np)-np.min(dmap_np))
            #if pdmap_divisor == 0:
            #    normalised_pdmap = (dmap_np - np.min(dmap_np)) / pdmap_divisor
            #normalised_gdmap = (ground_truth_dmap - np.min(ground_truth_dmap)) / (np.max(ground_truth_dmap)-np.min(ground_truth_dmap))
            dm_psnr.append(peak_signal_noise_ratio(ground_truth_dmap,dmap_np))
            dm_ssim.append(structural_similarity(ground_truth_dmap,dmap_np))
    
    # localisation metrics (using kernalised dmaps)
    for gt_dmap, pred_dmap in zip(y_coords, y_hat_coords):
        if not mdl.dlr_acd:
            gt_dmap = np.swapaxes(gt_dmap,1,0)
        else:
            gt_dmap = np.array(gt_dmap)
        
        if not len(gt_dmap) == 0:
            dist_matrix = distance.cdist(gt_dmap, pred_dmap, 'euclidean')
            
            # hungarian algorithm from scipy (yes, optimality is critical, even for eval)
            optim = linear_sum_assignment(dist_matrix)
            
            dists = [] # matched distances per density map
            
            # match all pred points (if pred < gt) or match up to predicted number of points
            for i in range(min(int(gt_dmap.sum()),len(optim[0]))):
                # delete entry from distance matrix when match is found
                dists.append(dist_matrix[optim[0][i],optim[1][i]]) 
            
            dists = np.array(dists)
            
            # DEBUG VIZ
            # unnorm = UnNormalize(mean=tuple(c.norm_mean),
            #                       std=tuple(c.norm_std))
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1,1, figsize=(10, 10))
            # ax.scatter(pred_dmap[:,0], pred_dmap[:,1],label='Predicted coordinates')
            # ax.scatter(gt_dmap[:,0], gt_dmap[:,1],c='red',marker='1',label='Ground truth coordinates')
            # 1/0
    
            # check me!
            tp = np.count_nonzero(dists<=thres)

        else:
            tp = 0 # set tp to zero for null annotations
        
        localisation_dict['tp'] += tp
        localisation_dict['fp'] += pred_dmap.shape[0]-tp
        localisation_dict['fn'] += gt_dmap.shape[0]-tp
            
    metric_dict = gen_metrics(dm_mae,dm_mse,dm_ssim,dm_psnr,y_n,y_hat_n,game,gampe,localisation_dict,mode)
    
    t2 = time.perf_counter()
    print("Dmap evaluation finished. Wall time: {}".format(round(t2-t1,2)))
    
    return  metric_dict # localisation_dict
    
def dlr_acd_whole_image_eval(mdl,loader):
    
    metric_dict = {'mae':None,
                   'mnae':None,
                   'rmse':None,
                   'precision':None,
                   'recall':None,
                   'f1 score':None}
    
    # create dataloaders without overlap
    
    # loop over images and sum preds per image
    
    # calculate overall metrics
    
    # MAE
    # MNAE
    # RMSE
    
    # Precision
    # Recall 
    # F1 Score
    
    return metric_dict
    
# TODO extra args: plot = True, save=True,hist=True
# TODO: don't shift computation over to cpu after sampling from model
@torch.no_grad()
def eval_mnist(mdl, valloader, trainloader,samples = 1,confusion = False, preds = False): 
    ''' currently the confusion matrix is calculated across both train and val splits'''
    
    accuracies = []
    p = []
    l = []
    
    for loader in [valloader,trainloader]:
        
        tally = 0

        for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            
            images,labels = data
                
            # Z shape: torch.Size([2, 4, 300, 400]) (batch size = 2)
            # channels * 4 is a result of haar downsampling
            if c.subnet_type == 'conv':
                dummy_z = (randn(images.size()[0], c.channels*4 , c.density_map_h // 2,c.density_map_w  // 2, requires_grad=True)).to(c.device)
            else:
                dummy_z = (randn(images.size()[0], c.channels, requires_grad=True)).to(c.device)
            
            images = images.float().to(c.device)
            dummy_z = dummy_z.float().to(c.device)
            
            mdl = mdl.to(c.device)
            
            mean_array = []
            
            for i in range(samples):
                x, _ = mdl(images,dummy_z,rev=True) # TODO: investigate eval use for log det j?
                
                if c.one_hot:
                    if c.subnet_type == 'conv':
                        x = x.argmax(-3).to(torch.float)
                    else:
                        x = x.argmax(-1).to(torch.float)
                    
#                dims = (1,2,3)
#                 
#                if c.one_hot:
#                    dims = (1,2)
                
                if c.subnet_type == 'conv':
                    x = torch.reshape(x,(images.size()[0],c.density_map_h * c.density_map_w)) # torch.Size([200, 12, 12])
                
                raw_preds = x  # torch.Size([200, 144])
                
                if c.subnet_type == 'conv':
                    x = torch.mode(x,dim = 1).values.cpu().detach().numpy() #print(x.shape) (200,)
                else:
                    x = x.cpu().detach().numpy()
                
                #mean_preds = np.round(x.mean(dim = dims).cpu().detach().numpy()) # calculate using mean of preds, not mode
                mean_array.append(x) # mean_preds
            
            # https://stackoverflow.com/questions/48199077/elementwise-aggregation-average-of-values-in-a-list-of-numpy-arrays-with-same
            #mean_preds = np.mean(mean_array,axis=0)
                
            labels = labels.cpu().detach().numpy()
            tally += (labels == x).sum() # mean_preds
            #print(tally)
            
            if confusion:
                p.append(x)
                l.append(labels)
        
        accuracies.append(round(tally/(len(loader.dataset))*100,2))
  
    out = accuracies[0], accuracies[1]
        
    if confusion:
        l = torch.IntTensor([item for batch in l for item in batch]) # flatten lists
        p = torch.IntTensor([item for batch in p for item in batch]) 
        stacked = torch.stack((l,p),dim = 1)
        cmt = torch.zeros(10,10, dtype=torch.int64) 
        
        for p in stacked:
            tl, pl = p.tolist()
            cmt[tl, pl] = cmt[tl, pl] + 1
        
        out = out + (cmt,) 
    
    if preds:
        out = out + (raw_preds,)

    return  out # train, val

@torch.no_grad()
def dmap_metrics(mdl, loader,n=10,mode='',null_filter=(a.args.sampler == 'weighted')):
    '''DMAP,COUNT,LOCALIZATION metrics'''
    
    thres=mdl.sigma*2
    
    loader_check(mdl=mdl,loader=loader)
    
    assert not mdl.count
    assert c.data_prop == 1
    assert mode in ['train','val']
    assert mdl.subnet_type == 'conv'
   
    print("Dmap Evaluation....")
    t1 = time.perf_counter()
    
    localisation_dict = {
        'tp':0,
        'fp':0,
        'fn':0
        }
    
    y_n = []; y_coords = []
    y_hat_n = [];  y_hat_n_dists = []; y_hat_coords = []
    dm_mae = []; dm_mse = []; dm_psnr = []; dm_ssim = [] #;dm_kl = []
    game = []; gampe = []; correct = 0; total = 0
    
    mdl = mdl.to(c.device)
    
    for i, data in enumerate(tqdm(loader, disable=False)):
        
        if not mdl.dlr_acd and not loader.dataset.classification:
            images,dmaps,labels,annotations, point_maps  = data
        elif not mdl.dlr_acd:
            images,dmaps,labels,binary_labels , annotations, point_maps  = data 
        else:
            images,dmaps,counts,point_maps = data

        images = images.float().to(c.device)
         
        if mdl.scale == 1:
            n_ds = 5
        elif mdl.scale == 2:
            n_ds = 4
        elif mdl.scale == 4:
            n_ds = 3
                            
        in_channels = c.channels*4**n_ds
        ft_dims = ft_dims_select(mdl) 

        x_list = []
                    
        ## sample from model per aerial image ---
        for i in range(n):
            dummy_z = (randn(images.size()[0], in_channels,ft_dims[0],ft_dims[1])).to(c.device)
        
            # feature extractor pre filtering
            # currently only on cow dataset
                
            # TODO - classification head for DLR ACD (if null filtering needed)
            # else: 
            #     outputs = mdl.feat_extractor(images)
            #     _, preds = torch.max(outputs, 1)   
        
            x, _ = mdl(images,dummy_z,rev=True)
            x_list.append(x)
            x_agg = torch.stack(x_list,dim=1)
            x = x_agg.mean(dim=1) # take average of samples from models for mean reconstruction
            
        if not mdl.dlr_acd:
            #features = mdl.feat_extractor(images)
            #outputs = mdl.classification_head(features)
            if null_filter:
                outputs = mdl.classification_head(images)  
                _, preds = torch.max(outputs, 1)  
        
        # replace predicted densities with null predictions if not +ve pred from feature extractor
        if null_filter:
            if not mdl.dlr_acd:
                correct += sum(preds == binary_labels)
                total += len(binary_labels)
                x[(preds == 1).bool(),:,:,:] = torch.zeros(1,mdl.density_map_h,mdl.density_map_w).to(c.device)
            
        for idx in range(images.size()[0]):               
                
            dmap_rev_np = x[idx].squeeze().cpu().detach().numpy()
            pred_count = dmap_rev_np.sum() # sum across channel, spatial dims for counts
            dist_counts  = x_agg[idx].sum((1,2,3)) 
            
            # We 'DONT' evaluate using the GT dmap
            ground_truth_dmap = dmaps[idx].squeeze().cpu().detach().numpy()
            
            # if pred_count > 0:
            #     thres = int(pred_count)

            # TODO- retrieve points from point annotated masks
            if mdl.dlr_acd:
                pm = point_maps[idx].squeeze().cpu().detach().numpy()
                gt_coords = np.argwhere(pm != 0)
                
            # else:
            #     anno = annotations[idx].cpu().detach().numpy()
            
            if mdl.dlr_acd:
                ground_truth_point_map = point_maps[idx].cpu().detach().numpy()
                gt_count = ground_truth_point_map.sum() / 3
            else:
                ground_truth_point_map, _ = create_point_map(mdl=mdl,annotations=annotations[idx].cpu().detach().numpy()) 
                gt_count = ground_truth_point_map.sum().round()
                gt_coords = annotations[idx]
                
                if gt_coords.nelement() != 0:
                    gt_coords = torch.stack([gt_coords[:,2]*mdl.density_map_h,gt_coords[:,1]*mdl.density_map_w]).cpu().detach().numpy()
                else:
                    gt_coords = None
                
            # subtract constrant for uniform noise
            constant = ((mdl.noise)/2)*ground_truth_dmap.shape[0]*ground_truth_dmap.shape[1] 
            loader_noise = ((a.args.noise)/2)*ground_truth_dmap.shape[0]*ground_truth_dmap.shape[1] 
            
            pred_count -= constant
            dist_counts -= constant
            gt_count -= loader_noise
            
            coordinates = peak_local_max(dmap_rev_np,min_distance=int(mdl.sigma*2),num_peaks=max(1,int(pred_count)))
            
            if mdl.dlr_acd:
                #y.append()
                y_n.append(gt_count)
            else:
                #y.append(labels[idx].cpu().detach().numpy())
                y_n.append(len(labels[idx]))
            
            if gt_coords is not None:
                y_coords.append(gt_coords)
                
            y_hat_n.append(pred_count)
            y_hat_n_dists.append(dist_counts)
            y_hat_coords.append(coordinates) 
            
            # dmap metrics (we do use the kernalised dmap for this)
            dm_mae.append(sum(abs(dmap_rev_np-ground_truth_dmap)))
            dm_mse.append(sum(np.square(dmap_rev_np-ground_truth_dmap)))
            
            dm_psnr.append(peak_signal_noise_ratio(ground_truth_dmap,dmap_rev_np)) #data_range=ground_truth_dmap.max()-ground_truth_dmap.min()))
            dm_ssim.append(structural_similarity(ground_truth_dmap,dmap_rev_np))
                        
            # local-count metrics # TODO
            l = 1 # cell size param - number of cells to split images into: 0 = 1, 1 = 4, 2 = 16, etc
            
            # this splits the density maps into cells for counting per cell
            gt_dmap_split_counts = np_split(ground_truth_point_map,nrows=mdl.density_map_w//4**l,ncols=mdl.density_map_h//4**l).sum(axis=(1,2))
            
            #zambingo
            
            pred_dmap_split_counts = np_split(dmap_rev_np,nrows=mdl.density_map_w//4**l,ncols=mdl.density_map_h//4**l).sum(axis=(1,2))

            game.append(sum(abs(pred_dmap_split_counts-gt_dmap_split_counts)))
            gampe.append(sum(abs(pred_dmap_split_counts-gt_dmap_split_counts)/np.maximum(np.ones(len(gt_dmap_split_counts)),gt_dmap_split_counts)))     
    
    # localisation metrics (using kernalised dmaps)
    for gt_dmap, pred_dmap in zip(y_coords, y_hat_coords):
        if not mdl.dlr_acd:
            gt_dmap = np.swapaxes(gt_dmap,1,0)
        else:
            gt_dmap = np.array(gt_dmap)
        
        if not len(gt_dmap) == 0:
            dist_matrix = distance.cdist(gt_dmap, pred_dmap, 'euclidean')
            
            # hungarian algorithm from scipy (yes, optimality is critical, even for eval)
            optim = linear_sum_assignment(dist_matrix)
            
            dists = [] # matched distances per density map
            
            # match all pred points (if pred < gt) or match up to predicted number of points
            for i in range(min(int(gt_dmap.sum()),len(optim[0]))):
                # delete entry from distance matrix when match is found
                dists.append(dist_matrix[optim[0][i],optim[1][i]]) 
            
            dists = np.array(dists)
            
            # DEBUG VIZ
            # unnorm = UnNormalize(mean=tuple(c.norm_mean),
            #                       std=tuple(c.norm_std))
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1,1, figsize=(10, 10))
            # ax.scatter(pred_dmap[:,0], pred_dmap[:,1],label='Predicted coordinates')
            # ax.scatter(gt_dmap[:,0], gt_dmap[:,1],c='red',marker='1',label='Ground truth coordinates')
    
            tp = np.count_nonzero(dists<=thres)
        else:
            tp = 0 # set tp to zero for null annotations
        
        localisation_dict['tp'] += tp
        localisation_dict['fp'] += pred_dmap.shape[0]-tp
        localisation_dict['fn'] += gt_dmap.shape[0]-tp
            
    metric_dict = gen_metrics(dm_mae,dm_mse,dm_ssim,dm_psnr,y_n,y_hat_n,game,gampe,localisation_dict,mode)
    
    t2 = time.perf_counter()
    print("Dmap evaluation finished. Wall time: {}".format(round(t2-t1,2)))
    
    print('correct binary classifier preds: {}/{}'.format(correct,total))
    
    return  metric_dict # localisation_dict

@torch.no_grad()
def dmap_pr_curve(mdl, loader,n = 10,mode = ''):
    '''DMAP,COUNT,LOCALIZATION metrics'''
    
    assert not mdl.count
    assert mode in ['train','val']
    assert mdl.subnet_type == 'conv'
    
    print("Plotting {} PR curve".format(mode))
    
    localisation_dists = []
    thresholds = np.arange(0, MAX_DISTANCE, STEP)
    
    y = []; y_n = []; y_coords = []
    y_hat_n = [];  y_hat_coords = {'div2':[],'same':[],'mult2':[],'mult4':[]}
    
    mdl = mdl.to(c.device)
    
    for i, data in enumerate(tqdm(loader, disable=False)):
        
        if not mdl.dlr_acd and not loader.dataset.classification:
            images,dmaps,labels,annotations, point_maps  = data
        elif not mdl.dlr_acd:
            images,dmaps,labels, _ , annotations, point_maps  = data # binary labels
        else:
            images,dmaps,counts,point_maps = data
            
        images = images.float().to(c.device)
         
        if mdl.scale == 1:
            n_ds = 5
        elif mdl.scale == 2:
            n_ds = 4
        elif mdl.scale == 4:
            n_ds = 3
                            
        in_channels = c.channels*4**n_ds
        ft_dims = ft_dims_select(mdl) 

        x_list = []
                    
        ## sample from model ---
        for i in range(n):
            dummy_z = (randn(images.size()[0], in_channels,ft_dims[0],ft_dims[1])).to(c.device)
            x, _ = mdl(images,dummy_z,rev=True)
            x_list.append(x)
        
        x_agg = torch.stack(x_list,dim=1)
        x = x_agg.mean(dim=1)
        
        # get results for each item in batch
        for idx in range(images.size()[0]):               
        
            idx = random.randint(0,images.size()[0]-1)
                
            dmap_rev_np = x[idx].squeeze().cpu().detach().numpy()
            pred_count = dmap_rev_np.sum()
            
            ground_truth_dmap = dmaps[idx].squeeze().cpu().detach().numpy()
            gt_count = ground_truth_dmap.sum().round()
            
            if mdl.dlr_acd:
                pm = point_maps[idx].squeeze().cpu().detach().numpy()
                gt_coords = np.argwhere(pm != 0)#.tolist()
            else:
                gt_coords = annotations[idx]
                gt_coords = torch.stack([gt_coords[:,2]*mdl.density_map_h,gt_coords[:,1]*mdl.density_map_w]).cpu().detach().numpy()
                    

            # subtract constrant for uniform noise
            constant = ((mdl.noise)/2)*ground_truth_dmap.shape[0]*ground_truth_dmap.shape[1] 
            loader_noise = ((a.args.noise)/2)*ground_truth_dmap.shape[0]*ground_truth_dmap.shape[1] 
            
            gt_count -= loader_noise
            pred_count -= constant
            
            
            y_hat_coords['div2'].append(peak_local_max(dmap_rev_np,min_distance=int(mdl.sigma//2),num_peaks=max(1,int(pred_count))))
            y_hat_coords['same'].append(peak_local_max(dmap_rev_np,min_distance=int(mdl.sigma),num_peaks=max(1,int(pred_count))))
            y_hat_coords['mult2'].append(peak_local_max(dmap_rev_np,min_distance=int(mdl.sigma*2),num_peaks=max(1,int(pred_count))))
            y_hat_coords['mult4'].append(peak_local_max(dmap_rev_np,min_distance=int(mdl.sigma*4),num_peaks=max(1,int(pred_count))))
            
            #y.append(labels[idx].cpu().detach().numpy()) # TODO - rm?
            
            if mdl.dlr_acd:
                y_n.append(counts[idx])
            else:    
                y_n.append(len(labels[idx]))
                
            y_coords.append(gt_coords)
            y_hat_n.append(pred_count)    
    
    # localisation metrics 
    # it turns out this is actually an instance of the assignment problem
            
    # Approach:
    # 1) Calculate distance matrix between ground truth and prediction 2d point sets
    # 2) Find optimal assignment using hungarian algo from scipy
    # 3) match upto the gt no of pts
    
    pr_dict = {'div2':[],'same':[],'mult2':[],'mult4':[]}    
    rc_dict = {'div2':[],'same':[],'mult2':[],'mult4':[]}   
     
    for mid_d in y_hat_coords.keys():
        
        localisation_dists = []
        for i in range(0,len(thresholds)):
            localisation_dists.append({'tp':0,'fp':0,'fn':0})  

        for gt_dmap, pred_dmap in zip(y_coords, y_hat_coords[mid_d]):
            
            if not mdl.dlr_acd:
                gt_dmap = np.swapaxes(gt_dmap,1,0)
                
            if len(gt_dmap) == 0:   
                dist_matrix = distance.cdist(gt_dmap, pred_dmap, 'euclidean')
                
                # hungarian algorithm from scipy (yes, optimality is critical, even for eval)
                optim = linear_sum_assignment(dist_matrix)
                
                dists = [] # optimal matched distances per density map
                
                # match all pred points (if pred < gt) or vice versa
                for i in range(min(int(gt_dmap.sum()),len(optim[0]))):
                    # append distances to distance vector from dist matrix in optimal order
                    dists.append(dist_matrix[optim[0][i],optim[1][i]]) 
        
                if c.debug:
                    fig, ax = plt.subplots(1,1, figsize=(10, 10))
                    ax.scatter(pred_dmap[:,0], pred_dmap[:,1],label='Predicted coordinates')
                    ax.scatter(gt_dmap[:,0], gt_dmap[:,1],c='red',marker='1',label='Ground truth coordinates')
                
                dists = np.array(dists)
            else:
                dists = np.array([])
            
            # loop over threshold values
            i = 0
            
            for num in thresholds:
                
                tp = np.count_nonzero(dists<=num)
                
                localisation_dists[i]['tp'] += tp
                localisation_dists[i]['fp'] += pred_dmap.shape[0]-tp
                localisation_dists[i]['fn'] += gt_dmap.shape[0]-tp
                i += 1
               
            # print('shapes')
            # print(pred_dmap.shape[0])
            # print(gt_dmap.shape[0])
            # print('metrics')
            # print(tp)
            # print(pred_dmap.shape[0]-tp)
            # print(gt_dmap.shape[0]-tp)
                
        pr = []; rc = []
        
        for i in range(0,len(thresholds)):
            pr.append(localisation_dists[i]['tp']/(localisation_dists[i]['tp']+localisation_dists[i]['fp']))
            rc.append(localisation_dists[i]['tp']/(localisation_dists[i]['tp']+localisation_dists[i]['fn']))
            
        pr_dict[mid_d].append(np.array(pr).squeeze())
        rc_dict[mid_d].append(np.array(rc).squeeze())
        
        del localisation_dists
     
    fig, ax = plt.subplots(2,1, figsize=(8, 8))

    ax[0].plot(thresholds,rc_dict['div2'][0], '-o', c='blue',label='Min D: Half kernel')
    ax[0].plot(thresholds,rc_dict['same'][0], '-o', c='red',label='Min D: Kernel width')
    ax[0].plot(thresholds,rc_dict['mult2'][0], '-o', c='orange',label='Min D: Kernel x2')
    ax[0].plot(thresholds,rc_dict['mult4'][0], '-o', c='green',label='Min D: Kernel x4')
    ax[0].title.set_text('Recall Curve')
    ax[0].set(xlabel="", ylabel="Recall")
    ax[0].legend()
    
    ax[1].plot(thresholds,pr_dict['div2'][0], '-o', c='blue',label='Min D: Half kernel')
    ax[1].plot(thresholds,pr_dict['same'][0], '-o', c='red',label='Min D: Kernel width')
    ax[1].plot(thresholds,pr_dict['mult2'][0], '-o', c='orange',label='Min D: Kernel x2')
    ax[1].plot(thresholds,pr_dict['mult4'][0], '-o', c='green',label='Min D: Kernel x4')
    ax[1].title.set_text('Precision Curve')
    ax[1].set(xlabel="Threshold: Euclidean Distance (pixels)", ylabel="Precision") 
    ax[1].legend()
    
    fig.show()
    
    if not os.path.exists(g.VIZ_DIR):
        os.makedirs(g.VIZ_DIR)
    
    plt.savefig("{}/PR_{}.jpg".format(g.VIZ_DIR,mdl.modelname), bbox_inches='tight', pad_inches = 0)
    
    return 
