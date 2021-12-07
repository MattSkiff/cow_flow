from tqdm import tqdm
import numpy as np
from skimage.feature import peak_local_max # 
from scipy.spatial import distance
import random
import torch
from torch import randn

import matplotlib.pyplot as plt
from utils import ft_dims_select, UnNormalize
import config as c

MAX_DISTANCE = 100
MIN_D = int(c.sigma)
STEP = 1

# TODO extra args: plot = True, save=True,hist=True
# TODO: don't shift computation over to cpu after sampling from model
@torch.no_grad()
def eval_mnist(mdl, valloader, trainloader,samples = 1,confusion = False, preds = False): 
    ''' currently the confusion matrix is calculated across both train and val splits'''
    
    mdl.eval()
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
                dummy_z = (randn(loader.batch_size, c.channels*4 , c.density_map_h // 2,c.density_map_w  // 2, requires_grad=True)).to(c.device)
            else:
                dummy_z = (randn(loader.batch_size, c.channels, requires_grad=True)).to(c.device)
            
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
                    x = torch.reshape(x,(loader.batch_size,c.density_map_h * c.density_map_w)) # torch.Size([200, 12, 12])
                
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
def dmap_metrics(mdl, loader,n=1,mode='',thres=c.sigma*2):
    '''DMAP,COUNT,LOCALIZATION metrics'''
    
    assert not mdl.count
    assert mode in ['train','val']
    assert mdl.subnet_type == 'conv'
    
    localisation_dict = {
        'tp':0,
        'fp':0,
        'fn':0
        }
    
    y = []; y_n = []; y_coords = []
    y_hat_n = [];  y_hat_n_dists = [];  y_hat_coords = []
    dm_mae = []; dm_mse = []
    game = []; gampe = []
    
    mdl = mdl.to(c.device)
    
    for i, data in enumerate(tqdm(loader, disable=False)):
        
        images,dmaps,labels,annotations = data
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
            dummy_z = (randn(loader.batch_size, in_channels,ft_dims[0],ft_dims[1])).to(c.device)
            x, _ = mdl(images,dummy_z,rev=True)
            x_list.append(x)
        
        x_agg = torch.stack(x_list,dim=1)
        x = x_agg.mean(dim=1) # sum across channel, spatial dims for counts

        for idx in range(loader.batch_size):               
        
            idx = random.randint(0,loader.batch_size-1)
                
            dmap_rev_np = x[idx].squeeze().cpu().detach().numpy()
            sum_count = dmap_rev_np.sum()
            dist_counts  = x_agg[idx].sum((1,2,3)) 
            
            ground_truth_dmap = dmaps[idx].squeeze().cpu().detach().numpy()
            gt_count = ground_truth_dmap.sum().round()
            
            # if sum_count > 0:
            #     thres = int(sum_count)
            
            gt_coords = annotations[idx]
            gt_coords = torch.stack([gt_coords[:,2]*mdl.density_map_h,gt_coords[:,1]*mdl.density_map_w]).cpu().detach().numpy()
            
            # subtract constrant for uniform noise
            constant = ((c.noise)/2)*ground_truth_dmap.shape[0]*ground_truth_dmap.shape[1] # TODO mdl.noise
            sum_count -= constant
            dist_counts -= constant
            gt_count -= constant
            
            coordinates = peak_local_max(dmap_rev_np,min_distance=MIN_D,num_peaks=max(1,int(sum_count)))
            
            y.append(labels[idx].cpu().detach().numpy())
            y_n.append(len(labels[idx]))
            y_coords.append(gt_coords)
            y_hat_n.append(sum_count)
            y_hat_n_dists.append(dist_counts)
            y_hat_coords.append(coordinates) 
            
            # dmap metrics
            dm_mae.append(sum(abs(dmap_rev_np-ground_truth_dmap)))
            dm_mse.append(sum(np.square(dmap_rev_np-ground_truth_dmap)))
            
    # local-count metrics # TODO
    game.append(0)
    gampe.append(0)
    
    
    # localisation metrics 
    for gt_dmap, pred_dmap in zip(y_coords, y_hat_coords):
        gt_dmap = np.swapaxes(gt_dmap,1,0)
        
        dists = [] # matched distances per density map
        dist_matrix = distance.cdist(gt_dmap, pred_dmap, 'euclidean')
        
        # match all pred points (if pred < gt) or vice versa
        for i in range(0,min(gt_dmap.shape[0],pred_dmap.shape[0])):
            md = np.amin(dist_matrix)
            # delete entry from distance matrix when match is found
            dist_matrix = np.delete(dist_matrix,np.where(dist_matrix == md))
            dists.append(md) 
        
        dists = np.array(dists)
        
        # DEBUG VIZ
        # unnorm = UnNormalize(mean=tuple(c.norm_mean),
        #                      std=tuple(c.norm_std))
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1,1, figsize=(10, 10))
        # ax.scatter(pred_dmap[:,0], pred_dmap[:,1],label='Predicted coordinates')
        # ax.scatter(gt_dmap[:,0], gt_dmap[:,1],c='red',marker='1',label='Ground truth coordinates')

        tp = np.count_nonzero(dists<=thres)
        
        localisation_dict['tp'] += tp
        localisation_dict['fp'] += pred_dmap.shape[0]-tp
        localisation_dict['fn'] += gt_dmap.shape[0]-tp
            
    # dmap metrics
    dm_mae = np.mean(np.vstack(dm_mae))
    dm_mse = np.mean(np.vstack(dm_mse))
    
    ## Counts
    
    # R2 
    n = len(y_n)
    y_hat_n = np.array(y_hat_n)
    y_n = np.array(y_n)
    
    y_bar = y_n.mean() 
    ss_res = np.sum((y_n-y_hat_n)**2)
    ss_tot = np.sum((y_n-y_bar)**2) 
    r2 = round(1 - (ss_res / ss_tot),4)
    
    # RMSE
    rmse = round(sum(((y_n-y_hat_n)/n)**2)**0.5,4)
    
    # MAE
    mae = round(sum(np.abs(y_n-y_hat_n))/n,4)
    
    # MAPE
    mape = round(sum(np.abs(y_n-y_hat_n)/np.maximum(np.ones(len(y_n)),y_n))/n,4)
    
   # GAME = 0
   # GAMPE = 0
    fscore = localisation_dict['tp']/(localisation_dict['tp']+0.5*(localisation_dict['fp']+localisation_dict['fn']))
    
    metric_dict = {
        '{}_r2'.format(mode):r2,
        '{}_rmse'.format(mode):rmse,
        '{}_mae'.format(mode):mae,
        '{}_mape'.format(mode):mape,
        '{}_dm_mae'.format(mode):dm_mae,
        '{}_dm_mse'.format(mode):dm_mse,
       # '{}_game'.format(mode):game,
       # '{}_gampe'.format(mode):gampe,
        '{}_fscore'.format(mode):fscore
        }
    
    return  metric_dict # localisation_dict

@torch.no_grad()
def dmap_pr_curve(mdl, loader,n = 10,mode = ''):
    '''DMAP,COUNT,LOCALIZATION metrics'''
    
    assert not mdl.count
    assert mode in ['train','val']
    assert mdl.subnet_type == 'conv'
    
    localisation_dists = []
    thresholds = np.arange(0, MAX_DISTANCE, STEP)
    
    for i in range(0,len(thresholds)):
      localisation_dists.append({
    'tp':0,
    'fp':0,
    'fn':0
    })  

    y = []; y_n = []; y_coords = []
    y_hat_n = [];  y_hat_coords = []
    
    mdl = mdl.to(c.device)
    
    for i, data in enumerate(tqdm(loader, disable=False)):
        
        images,dmaps,labels,annotations = data
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
            dummy_z = (randn(loader.batch_size, in_channels,ft_dims[0],ft_dims[1])).to(c.device)
            x, _ = mdl(images,dummy_z,rev=True)
            x_list.append(x)
        
        x_agg = torch.stack(x_list,dim=1)
        x = x_agg.mean(dim=1)
        
        # get results for each item in batch
        for idx in range(loader.batch_size):               
        
            idx = random.randint(0,loader.batch_size-1)
                
            dmap_rev_np = x[idx].squeeze().cpu().detach().numpy()
            sum_count = dmap_rev_np.sum()
            
            ground_truth_dmap = dmaps[idx].squeeze().cpu().detach().numpy()
            gt_count = ground_truth_dmap.sum().round()
            
            gt_coords = annotations[idx]
            gt_coords = torch.stack([gt_coords[:,2]*mdl.density_map_h,gt_coords[:,1]*mdl.density_map_w]).cpu().detach().numpy()
            
            # subtract constrant for uniform noise
            constant = ((c.noise)/2)*ground_truth_dmap.shape[0]*ground_truth_dmap.shape[1] # TODO mdl.noise
            gt_count -= constant
            sum_count -= constant
            
            coordinates = peak_local_max(dmap_rev_np,min_distance=MIN_D,num_peaks=max(1,int(sum_count)))
            
            y.append(labels[idx].cpu().detach().numpy())
            y_n.append(len(labels[idx]))
            y_coords.append(gt_coords)
            y_hat_n.append(sum_count)
            y_hat_coords.append(coordinates)     
    
    # localisation metrics 
    
    # it turns out this is actually an instance of the assignment problem and quite complex
    # however n should always be less than 1k-2k, and it's just for evaluation, so optimality isn't critical
            
    # 1) distance matrix
    # 2) iter over each ground truth point
    # 3) find nearest two points (min in dist matrix))
    # 4) add to dist vector and delete item in dist matrix
    # 5) filt dist vector according to distance threshold
            
    for gt_dmap, pred_dmap in zip(y_coords, y_hat_coords):
        gt_dmap = np.swapaxes(gt_dmap,1,0)
        
        dist_matrix = distance.cdist(gt_dmap, pred_dmap, 'euclidean')
        dists = [] # matched distances per density map
        
        # match all pred points (if pred < gt) or vice versa
        for i in range(0,min(gt_dmap.shape[0],pred_dmap.shape[0])):
            md = np.amin(dist_matrix)
            # delete entry from distance matrix when match is found
            dist_matrix = np.delete(dist_matrix,np.where(dist_matrix == md))
            dists.append(md) 

        # DEBUG VIZ
        # fig, ax = plt.subplots(1,1, figsize=(10, 10))
        # ax.scatter(pred_dmap[:,0], pred_dmap[:,1],label='Predicted coordinates')
        # ax.scatter(gt_dmap[:,0], gt_dmap[:,1],c='red',marker='1',label='Ground truth coordinates')
        # 1/0
        
        dists = np.array(dists)
        
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
     
    fig, ax = plt.subplots(2,1, figsize=(8, 8))
    
    ax[0].plot(thresholds,rc, '-o')
    ax[0].title.set_text('Recall Curve')
    ax[0].set(xlabel="", ylabel="Recall")
    
    ax[1].plot(thresholds,pr, '-o')
    ax[1].title.set_text('Precision Curve')
    ax[1].set(xlabel="Threshold: Euclidean Distance (pixels)", ylabel="Precision") 
    
    fig.show()
    
    return localisation_dists
