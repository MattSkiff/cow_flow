from torch import randn
from tqdm import tqdm
import numpy as np
import config as c
import torch

# TODO extra args: plot = True, save=True,hist=True
# TODO: don't shift computation over to cpu after sampling from model
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

def dmap_metrics(mdl, loader,n=1,mode=''):
    '''DMAP,COUNT,LOCALIZATION metrics'''
    
    assert not mdl.count
    assert mode in ['train','val']
    assert mdl.subnet_type == 'conv'

    y = []; y_n = []; y_coords = []
    
    y_hat_n = [];  y_hat_n_dists = [];  y_hat_coords = []
    
    dm_mae = []; dm_mse = []
    
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
        
        # get back to correct image data
        unnorm = UnNormalize(mean=tuple(c.norm_mean),
                             std=tuple(c.norm_std))
        
        x_agg = torch.stack(x_list,dim=1)
        
        # sum across channel, spatial dims for counts
        x = x_agg.mean(dim=1)
        x_std = x_agg.std(dim=1)
        x_std_norm = x_std #x_std-x #x_std.div(x)

        for idx in range(loader.batch_size):               
        
            if not full:
                import random
                idx = random.randint(0,loader.batch_size-1)
                
            dmap_rev_np = x[idx].squeeze().cpu().detach().numpy()
            #dmap_uncertainty = x_std[idx].squeeze().cpu().detach().numpy()
            sum_count = dmap_rev_np.sum()
            dist_counts  = x_agg[idx].sum((1,2,3)) 
            
            ground_truth_dmap = dmaps[idx].squeeze().cpu().detach().numpy()
            gt_count = ground_truth_dmap.sum().round()
            
            # if sum_count > 0:
            #     thres = int(sum_count)
              
            dmap_uncertainty = dmap_rev_np-ground_truth_dmap #x_std_norm[idx].squeeze().cpu().detach().numpy()
            
            coordinates = peak_local_max(dmap_rev_np, min_distance=int(loader.dataset.sigma)//2,threshold_rel=0.4)#,num_peaks=inf)
            
            # subtract constrant for uniform noise
            constant = ((1e-3)/2)*ground_truth_dmap.shape[0]*ground_truth_dmap.shape[1]
            sum_count -= constant
            dist_counts -= constant
            gt_count -= constant
            
            y.append(labels[idx].cpu().detach().numpy())
            y_n.append(len(labels[idx]))
            y_coords.append(annotations[idx])
            y_hat_n.append(sum_count)
            y_hat_n_dists.append(dist_counts)
            y_hat_coords.append(coordinates) 
            
            # dmap metrics
            dm_mae.append(sum(abs(dmap_rev_np-ground_truth_dmap)))
            dm_mse.append(sum(np.square(dmap_rev_np-ground_truth_dmap)))
            
    # dmap metrics
    dm_mae = np.mean(np.vstack(dm_mae))
    dm_mse = np.mean(np.vstack(dm_mse))
    
    ## Counts
    r2,rmse,mae,mape = -99,-99,-99,-99
    
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

    metric_dict = {
        '{}_r2':r2.format(mode),
        '{}_rmse':rmse.format(mode),
        '{}_mae':mae.format(mode),
        '{}_mape':mape.format(mode),
        '{}_dm_mae':dm_mae.format(mode),
        '{}_dm_mse':dm_mse.format(mode),
        }
    return  metric_dict

def dmap_local_metrics(y_coords,y_hat_coords):
    
    
    return 0
