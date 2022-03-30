# MIT License Marco Rudolph 2021
from torch import randn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch
import numpy as np
import random
from datetime import datetime 
import cv2
import re # lukas
from prettytable import PrettyTable

import config as c
import gvars as g
import arguments as a

#from scipy import ndimage as ndi
from skimage.feature import peak_local_max # 
#from skimage import data, img_as_float

if any('SPYDER' in name for name in os.environ):
    import matplotlib
    #matplotlib.use('TkAgg') # get around agg non-GUI backend error

def ft_dims_select(mdl=None):
    
    if mdl == None:
        fe = c.feat_extractor
    else:
        fe = mdl.feat_extractor.__class__.__name__
    
    if c.downsampling:
        
        if a.args.dlr_acd:
            ft_dims = (a.args.image_size // 2**5,a.args.image_size // 2**5) # 10, 10
        elif fe in ['resnet18','ResNetPyramid'] or fe == 'Sequential':
            ft_dims = (a.args.image_size // 2**5,a.args.image_size // 2**5) # 19, 25
        elif fe == 'vgg16_bn' or fe == 'VGG':
            ft_dims = (18,25)
        elif fe == 'alexnet' or fe == 'AlexNet':
            ft_dims = (17,24)
        elif fe == 'none' or fe == 'NothingNet':
            ft_dims = (600,800)
            
    else:
        ft_dims = (c.density_map_h//2,c.density_map_w//2)
        
    return ft_dims

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None

# split numpy matrix into submatrices
# credit SO user Dat
# link: https://stackoverflow.com/questions/11105375/how-to-split-a-matrix-into-4-blocks-using-numpy/11105569
def np_split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

def get_loss(z, jac,dims):
    # in differnet, exponentiate over 'channel' dim (n_feat)  
    # here, we exponentiate over channel, height, width to produce single norm val per density map
    
    return torch.mean(0.5 * torch.sum(z ** 2, dim=dims) - jac) / z.shape[1]

# change default initialisation
# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor      

@torch.no_grad()
def plot_preds_baselines(mdl, loader,mode="",mdl_type=''):
    
        assert mode in ['train','val']
        assert mdl_type in ['UNet','CSRNet','FCRN','LCFCN']
        
        lb_idx = random.randint(0,loader.batch_size-1)
    
        for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
                
                mdl.to(c.device)
                images,dmaps,labels,binary_labels,annotations = data
                preds = mdl(images)
                unnorm = UnNormalize(mean=tuple(c.norm_mean),
                                     std=tuple(c.norm_std))
                
                im = unnorm(images[lb_idx])
                im = im.permute(1,2,0).cpu().numpy()
                preds = preds[lb_idx].permute(1,2,0).cpu().numpy()

                fig, ax = plt.subplots(1,3)
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
                fig.set_size_inches(24*1,7*1)
                fig.set_dpi(100)
                fig.suptitle('Baseline ({}) Test - {}'.format(mdl_type,mode),y=1.0,fontsize=16) # :.2f
                
                [axi.set_axis_off() for axi in ax.ravel()] # turn off subplot axes

                ax[0].title.set_text('Density Map Prediction')
                ax[0].imshow(preds)
                ax[1].title.set_text('Conditioning Aerial Image')
                ax[1].imshow((im * 255).astype(np.uint8))
                ax[2].title.set_text('Ground Truth Density Map')
                ax[2].imshow(dmaps[lb_idx].cpu().numpy())
                
                return preds, preds.sum(),dmaps[lb_idx].sum(),len(labels[lb_idx])
                
# TODO split into minst and non mnist funcs
@torch.no_grad()
def plot_preds(mdl, loader, plot = True, save=False,title = "",digit=None,
               hist=False,sampling="randn",plot_n=None,writer=None,writer_epoch=None,
               writer_mode=None,include_empty=False,sample_n=10):
    
    assert type(loader) == torch.utils.data.dataloader.DataLoader
    
    """

    Parameters
    ----------
    model
        A saved model, either MNISTFlow or CowFlow
    loader
        A dataloader of samples.
    plot : Bool, optional
        Whether to return a plot or not. If not MNISt, will cycle through data loader until an item with an annotation is found.
    save : TYPE, optional
        DESCRIPTION. The default is True.
    title : TYPE, optional
        Plot title. The default is "".
    digit : TYPE, optional
        Predict a specific digit from MNIST. The default is None.
    hist : TYPE, optional
        Plot histogram of predictions. The default is True.
    sampling : TYPE, optional
        Three options: zeros, ones or randn (i.e. ~N(0,1) ). The default is "randn".
    digit: 
        If 'None' and plot_n == 'None', a random digits will be plotted, else the digit specified will be plotted
    plot_n:
        Plots n plots from the loader sequentially. If not MNIST, plots null-labelled data.
    include_empty:
        (True) Skip check for images with no annotations. Default: False.
    sample_n:
        Number of samples to average across 

    Returns
    -------
    dmap_rev_np:
        the samples used to generate the mean reconstruction.
    
    mean_pred:
        The mean reconstruction

    """
      
    if not mdl.mnist and digit != None:
        print('Digit argument ignored for non-MNIST models')
        
    if not mdl.mnist :
        if mdl.count != loader.dataset.count:
            raise ValueError("model and loader count properties do not match!")
    
    if mdl.mnist:
        if plot_n != None and digit != None:
            print("Warning: plot_n argument will be ignored")
        elif plot_n == None and digit == None:
            print("Plotting a random digit")
        elif plot_n != None:
            print("Plotting first {} digits".format(plot_n))
    
    def inner_func(mdl=mdl,hist=hist):
        
        plt.switch_backend('agg') # for tensorboardx
        idx = random.randint(0,len(loader)-1)
        
        k = 0
        # TODO: better var name
        z = 0 
        
        for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            
            # TODO - broken - filename doesn't match image
    #            if not mdl.mnist:
    #                sample_fname = loader.dataset.train_im_paths[i]
            
            if mdl.mnist:
                images,labels = data
            elif mdl.dlr_acd:
                images,dmaps,counts,point_maps = data
            elif loader.dataset.count: # model.dataset.count
                images,dmaps,labels,counts = data
            elif loader.dataset.classification:
                images,dmaps,labels,binary_labels,annotations = data
            else:
                images,dmaps,labels,_ = data
            
            # TODO - intermittent bug here
            dmaps = dmaps.cpu()
            
            lb_idx = random.randint(0,loader.batch_size-1)
                
            if mdl.count: #  or # TODO
    
                # check annotations in batch aren't empty
                lb_idx = None
                 
                for count in counts:
                    j = 0
                    z = z+1
                    if z >= len(loader)*loader.batch_size:
                        print("Loader has no labelled data!")
                    if count != 0:
                        lb_idx = j
                        break
                    else:
                        j = j + 1
                    
                if lb_idx == None:
                    continue  
            
            elif not mdl.mnist and not mdl.dlr_acd and not include_empty:
                
                 # check annotations in batch aren't empty
                lb_idx = None
                 
                j = 0
                for label in labels:
                    
                    z = z+1
                    if z >= len(loader)*loader.batch_size:
                        print("Loader has no labelled data!")
                    if len(label) != 0:
                        lb_idx = j
                        break
                    else:
                        j = j + 1
    
                if lb_idx == None:
                    continue  
            
            if i != idx and plot_n == None and digit == None and mdl.mnist:
                continue
           
            images = images.float().to(c.device)
            mdl = mdl.to(c.device)
            x_list = []
            
            for i in range(sample_n):
           
                ## create noise vector ---
                if (mdl.mnist and labels.size) or not mdl.mnist: # triggers only if there is at least one annotation
                    # Z shape: torch.Size([2, 4, 300, 400]) (batch size = 2)
                    
                    if not mdl.mnist:
                        # TODO - remove hard coding dims here
                        if (loader.dataset.count and not mdl.gap) or (not mdl.downsampling and mdl.subnet_type == 'conv'):
                            dummy_z = (randn(images.size()[0],mdl.channels*4,(mdl.density_map_h) // 2,(mdl.density_map_w) // 2)).to(c.device) 
                        elif loader.dataset.count and mdl.gap:
                            dummy_z = (randn(images.size()[0],1)).to(c.device) 
                        elif mdl.downsampling: # self.downsampling
                            
                            if mdl.scale == 1:
                                n_ds = 5
                            elif mdl.scale == 2:
                                n_ds = 4
                            elif mdl.scale == 4:
                                n_ds = 3
                                
                            in_channels = c.channels*4**n_ds
                            ft_dims = ft_dims_select(mdl)
                            
                            if mdl.feat_extractor.__class__.__name__ == 'NothingNet':
                                in_channels = 2                      
                            
                            dummy_z = (randn(images.size()[0], in_channels,ft_dims[0],ft_dims[1])).to(c.device)
                    else:
                            if mdl.subnet_type == 'conv':
                               
                                if sampling == 'ones':
                                    dummy_z = (torch.ones(images.size()[0], mdl.channels*4, mdl.density_map_h // 2,mdl.density_map_w  // 2).to(c.device))
                                elif sampling == "zeros":
                                    dummy_z = torch.zeros(images.size()[0], mdl.channels*4, mdl.density_map_h // 2,mdl.density_map_w  // 2).to(c.device)
                                elif sampling == "randn":
                                    dummy_z = (randn(images.size()[0], mdl.channels*4, mdl.density_map_h // 2,mdl.density_map_w  // 2)).to(c.device)
                                else:
                                    ValueError("Invalid function arg (sampling). Try 'randn', 'zeros' or 'ones'.")
                            else:
                                dummy_z = (randn(images.size()[0], c.channels)).to(c.device)
                        
                ## sample from model ---
                dummy_z = dummy_z.float().to(c.device)
                x, log_det_jac = mdl(images,dummy_z,rev=True)
                x_list.append(x)
                    
            x_agg = torch.stack(x_list,dim=1)
            x = x_agg.mean(dim=1) # take average of samples from models for mean reconstruction
                
            # replace predicted densities with null predictions if not +ve pred from feature extractor
            if not mdl.mnist and not mdl.count:
                # subtract constrant for uniform noise
                constant = ((mdl.noise)/2)*dmaps[lb_idx].shape[0]*dmaps[lb_idx].shape[1]
                if not mdl.dlr_acd:
                    # subtract constrant for uniform noise
                    constant = ((mdl.noise)/2)*dmaps[lb_idx].shape[0]*dmaps[lb_idx].shape[1]
                    print('replacing predicted densities with empty predictions from feature extractor')
                    outputs = mdl.classification_head(images)  
                    _, preds = torch.max(outputs, 1)  
                    x[(preds == 1).bool(),:,:,:] = torch.zeros(1,mdl.density_map_h,mdl.density_map_w).to(c.device)
                    
            if c.debug_utils:
                print("\nsampled from model\n")
            
            if lb_idx == None:
                lb_idx = 0
            
            if c.one_hot:
                
                if c.subnet_type == 'conv':
                    x = x.argmax(-3).to(torch.float)
                else:
                    x = x.argmax(-1).to(torch.float)
            
            if  plot_n != None:
                lb_idx = range(images.size()[0])
            else:
                lb_idx = [lb_idx]
            
            for lb_idx in lb_idx:
                dmap_rev_np = x[lb_idx].squeeze().cpu().detach().numpy()
                mean_pred = x[lb_idx].mean()
                
                # TODO
                if (mdl.count or mdl.mnist) and not mdl.gap:
                    x_flat = torch.reshape(x,(images.size()[0],c.density_map_h * c.density_map_w)) # torch.Size([200, 12, 12])
                    mode = torch.mode(x_flat,dim = 1).values.cpu().detach().numpy()
                elif mdl.count and mdl.gap:
                    x_flat = x
                                        
                if mdl.mnist and c.subnet_type == 'conv':
                    x = torch.mode(x,dim = 1).values.cpu().detach().numpy() #print(x.shape) (200,)
                else:
                    x = torch.mode(x,dim = 0).values.cpu().detach().numpy()
                
                # TODO!
                if mdl.mnist or mdl.subnet_type == 'conv':
                    sum_pred = dmap_rev_np.sum()-constant #[lb_idx]
                    true_dmap_count = dmaps[lb_idx].sum()-constant
                    if not mdl.dlr_acd:
                        label_count = len(annotations[lb_idx])
                    else:
                        label_count = counts[lb_idx]
                elif mdl.subnet_type == 'fc' and mdl.gap:  
                    mode = x
                    sum_pred = x
                
                if plot:
                    
                    if mdl.unconditional: 
                        hist = False
                    
                    n_plots = 3+hist-mdl.mnist-mdl.count
                           
                    fig, ax = plt.subplots(1,n_plots)
                    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
                    [axi.set_axis_off() for axi in ax.ravel()] # turn off subplot axes
                    
                    if mdl.mnist:
                        fig.suptitle('{} \n mode of reconstruction: {}'.format(title,str(mode)),y=1.0,fontsize=16) # :.2f
                    if mdl.count:   
                        fig.suptitle('{} \n Predicted count: {:.2f}\n True Labelled Count: {:.1f}'.format(title,mean_pred),y=1.0,fontsize=16)
                    else:
                        fig.suptitle('{} \n Predicted count: {:.2f} (-{} from noise adj.)'.format(title,sum_pred,constant),y=1.05,fontsize=16)
                        
                    fig.set_size_inches(24*1,7*1)
                    fig.set_dpi(100)
                    
                    if mdl.mnist:
                        im = images[lb_idx].squeeze().cpu().numpy()
                    else:
                        unnorm = UnNormalize(mean =tuple(c.norm_mean),
                                             std=tuple(c.norm_std))
                        
                        if mdl.dlr_acd:
                            im = images[lb_idx]
                        else:
                            im = unnorm(images[lb_idx])
                        
                        im = im.permute(1,2,0).cpu().numpy()
                     
                    if mdl.feat_extractor.__class__.__name__ == 'NothingNet':
                        dmap_rev_np = dmap_rev_np[1,:,:] # select only data from first duplicated channel
                    
                    if not (mdl.count and mdl.gap) and not (mdl.mnist and c.subnet_type == 'fc'):
                        ax[0].title.set_text('Density Map Prediction')
                        ax[0].imshow(dmap_rev_np)#, cmap='viridis', interpolation='nearest')
                    else:
                        fig.delaxes(ax[0])
                    
                    if mdl.count:
                        # no dmaps provided by dataloader if mdl.count
                        ax[1].title.set_text('Conditioning Aerial Image')
                        ax[1].imshow((im * 255).astype(np.uint8))
                        if hist:
                            ax[2].hist(dmap_rev_np.flatten(),bins = 30)
                    else:
                    
                        if mdl.mnist:
                            ax[1].title.set_text('Conditioning Digit')
                            ax[1].imshow(im)
                            if hist:
                                ax[2].title.set_text('Histogram of Reconstruction Values')
                                ax[2].hist(dmap_rev_np.flatten(),bins = 30)
                        else:
                            ax[1].title.set_text('Conditioning Aerial Image')
                            ax[1].imshow((im * 255).astype(np.uint8))
                            ax[2].title.set_text('Density Map Ground Truth: N={:.2f}\nLabel Count: N={:.2f}'.format(true_dmap_count,label_count))
                            ax[2].imshow(dmaps[lb_idx])
                            if hist:
                                ax[3].title.set_text('Histogram of Reconstruction Values')
                                ax[3].hist(dmap_rev_np.flatten(),bins = 30)
                    
                    if writer != None:
                        writer.add_figure('{} Dmap Pred: epoch {}'.format(writer_mode,writer_epoch), fig)
                        writer.close() 
                    
                    # saving and outs
                    if save:
                        if not os.path.exists(g.VIZ_DIR):
                            os.makedirs(g.VIZ_DIR)
                        plt.savefig("{}/{}.jpg".format(g.VIZ_DIR,mdl.modelname), bbox_inches='tight', pad_inches = 0)
                    
                    if mdl.mnist:
                        out = labels[lb_idx],dmap_rev_np, mean_pred 
                    else:
                        if mdl.count:
                            out = counts[lb_idx],dmap_rev_np, mean_pred
                        else:
                            out = {'True Density Map count':true_dmap_count,
                                   'Predicted Density Map':dmap_rev_np, 
                                   'Predicted Density Map count':sum_pred,
                                   'Label Count':label_count}
                    
                    if digit != None and labels[idx] == digit:
                        return out 
                    
                    if plot_n != None:
                        k = k+1
                        if k >= plot_n:
                            lb_idx = lb_idx
                            return out
                    else:
                        return out
                                
    out = inner_func()       

    return out

# TODO - add plot full set and save opt
@torch.no_grad()
def plot_peaks(mdl, loader,n=10):
    assert not mdl.count
    assert mdl.subnet_type == 'conv'

    mdl = mdl.to(c.device)
    
    for i, data in enumerate(tqdm(loader, disable=False)):
        
        if mdl.dlr_acd:
            images,dmaps,counts,point_maps = data
        elif loader.dataset.classification:
            images,dmaps,labels,binary_labels,annotations = data
        else: 
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
            dummy_z = (randn(images.size()[0], in_channels,ft_dims[0],ft_dims[1])).to(c.device)
            x, _ = mdl(images,dummy_z,rev=True)
            x_list.append(x)
        
        # get back to correct image data
        unnorm = UnNormalize(mean=tuple(c.norm_mean),
                             std=tuple(c.norm_std))
        
        x_agg = torch.stack(x_list,dim=1)
        
        # sum across channel, spatial dims for counts
        x = x_agg.mean(dim=1)
        x_std = x_agg.std(dim=1)
        #x_std_norm = x_std #x_std-x #x_std.div(x) # unused

        for idx in range(images.size()[0]):               
            
            idx = random.randint(0,images.size()[0]-1)
            
            ground_truth_dmap = dmaps[idx].squeeze().cpu().detach().numpy()
            
            if mdl.dlr_acd:
                point_map = point_maps[idx].squeeze().cpu().detach().numpy()
                label_count = counts[idx]
            else:
                label_count = len(annotations[idx])
                
            constant = ((mdl.noise)/2)*ground_truth_dmap.shape[0]*ground_truth_dmap.shape[1]
            
            dmap_rev_np = x[idx].squeeze().cpu().detach().numpy()
            #dmap_uncertainty = x_std[idx].squeeze().cpu().detach().numpy()
            sum_count = dmap_rev_np.sum()
            dist_counts  = x_agg[idx].sum((1,2,3))
            gt_count = ground_truth_dmap.sum().round()
            
            # if sum_count > 0:
            #     thres = int(sum_count)
              
            dmap_uncertainty = dmap_rev_np-ground_truth_dmap #x_std_norm[idx].squeeze().cpu().detach().numpy()
            
            # subtract constrant for uniform noise
            constant = ((mdl.noise)/2)*ground_truth_dmap.shape[0]*ground_truth_dmap.shape[1] # TODO mdl.noise
            sum_count -= constant
            dist_counts -= constant
            gt_count -= constant
            
            if mdl.dlr_acd:
                # https://stackoverflow.com/questions/60782965/extract-x-y-coordinates-of-each-pixel-from-an-image-in-python
                gt_coords = np.argwhere(point_map == 1)
            else:
                gt_coords = annotations[idx].cpu().detach().numpy()
                
            coordinates = peak_local_max(dmap_rev_np,min_distance=4,num_peaks=max(1,int(sum_count)))

            if mdl.dlr_acd:
                im = images[idx]
            else:
                im = unnorm(images[idx])
                                            
            im = images[idx].permute(1,2,0).cpu().numpy()
            fig, ax = plt.subplots(2,3, figsize=(18, 10))
            
            ax[0,0].imshow(dmap_rev_np, aspect="auto")
            ax[0,1].imshow(dmap_uncertainty, aspect="auto")
            ax[0,2].hist(dist_counts.cpu().numpy())
            ax[0,2].axvline(dist_counts.cpu().numpy().mean(), color='red', linestyle='dashed', linewidth=1)
            ax[1,0].imshow(ground_truth_dmap, aspect="auto")
            ax[1,1].imshow((im * 255).astype(np.uint8), aspect="auto")
            
            if len(coordinates) != 0:
                x_coords, y_coords = zip(*coordinates)
                ax[1,2].scatter(y_coords, x_coords,label='Predicted coordinates')
                if mdl.dlr_acd:
                   ax[1,2].scatter(gt_coords[:,1], gt_coords[:,0],c='red',marker='1',label='Ground truth coordinates') 
                else:
                    ax[1,2].scatter(gt_coords[:,1]*mdl.density_map_w, gt_coords[:,2]*mdl.density_map_h,c='red',marker='1',label='Ground truth coordinates')
                ax[1,2].legend()
                ax[1,2].set_ylim(ax[1,0].get_ylim())
                ax[1,2].set_xlim(ax[1,0].get_xlim())
                ax[1,2].title.set_text('Localizations via Peak Finding on Reconstruction\nN = {}'.format(len(coordinates)))   
                
            ax[0,0].title.set_text('Mean (N={}) Reconstructed Density Map\nSum Density Map = {:.2f} (-{} manual noise adj.)'.format(n,sum_count,constant))
            #ax[1,0].title.set_text('Pixel-wise Uncertainty Estimate (normalised std. dev)')
            ax[0,1].title.set_text('Image Differencing (Reconstruction, Ground Truth Density Map)')
            ax[0,2].title.set_text('Sum (Integration) Prediction Distribution (N = {})'.format(n))
            ax[1,0].title.set_text('Ground Truth Density Map\nN = {} | Sum = {:.2f}'.format(label_count,gt_count))
            ax[1,1].title.set_text('Aerial Image')
            
            fig.subplots_adjust(hspace=0.4)
            
            return
        
    return 
  
def plot_image(image_path,mdl):
    pass      

def torch_r2(mdl,loader):
    """calcs r2 (on torch only)"""
    assert mdl.count
    assert loader.dataset.count
    assert not mdl.mnist
    assert mdl.modelname 
    
    mdl = mdl.to(c.device)
    actuals = []
    means = []
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar,desc="Calculating R2")):
            
            images,dmaps,labels,counts = data
            
            if not mdl.gap:  
                num = 0
                dummy_z = (randn(images.size()[0],c.channels*4,c.density_map_h // 2,c.density_map_w // 2, requires_grad=False,device=c.device))
            else:
                num = 1
                dummy_z = (randn(images.size()[0],c.channels,requires_grad=False,device=c.device))
            
            dummy_z = dummy_z.float()
            x, _ = mdl(images,dummy_z,rev=True)
            
            if mdl.subnet_type == 'conv':
                mean_preds = x.mean(dim = tuple(range(1,len(x.shape)-num)))
                means.append(mean_preds)
            else:
                means.append(x)
                 
            actuals.append(counts)
        
        y_hat = torch.cat(means,dim=0).squeeze()
        y = torch.cat(actuals)
        y_bar = y.mean()
        
        ss_res = torch.sum((y-y_hat)**2)
        ss_tot = torch.sum((y-y_bar)**2)
        
        r2 = 1 - (ss_res / ss_tot)
             
    return r2   

def counts_preds_vs_actual(mdl,loader,plot=False,ignore_zeros=False):
    """Plots predicted versus actual counts from the data and returns the R^2 value. Required: count dataloader and count model."""
    
    assert mdl.count
    assert loader.dataset.count
    assert not mdl.mnist
    assert mdl.modelname 
    
    if ignore_zeros:
        assert loader.batch_size == 1
    
    means = []
    actuals = []
    
    if plot:
        desc = "Plotting Counts vs Actuals"
    else:
        desc = "Calculating R2"
        
    for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar,desc=desc)):
        
        images,dmaps,labels,counts = data

        
        if not mdl.gap:  
            num = 0
            dummy_z = (randn(images.size()[0],c.channels*4,c.density_map_h // 2,c.density_map_w // 2, requires_grad=False)).to(c.device)
        else:
            num = 1
            dummy_z = (randn(images.size()[0],c.channels,requires_grad=False)).to(c.device)
        
        images = images.float().to(c.device)
        dummy_z = dummy_z.float().to(c.device)
        mdl = mdl.to(c.device)
        x, _ = mdl(images,dummy_z,rev=True)
        
        if mdl.subnet_type == 'conv':
            mean_preds = x.mean(dim = tuple(range(1,len(x.shape)-num))).detach().cpu().numpy()
            means.append(mean_preds)
        else:
             mean_preds = x.squeeze().detach().cpu().numpy().tolist()
             
             if not isinstance(mean_preds, list):
                means.append(mean_preds) 
             else:
                means.extend(mean_preds)
    
        actuals.append(counts.detach().cpu().numpy())
    
    if mdl.gap:
        means = [means]
    
    means = np.concatenate(means, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    correlation_matrix = np.corrcoef(means, actuals)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    
    # TODO - add model info to plot
    if plot:
        ident = [np.min(np.concatenate([means,actuals], axis=0)), np.max(np.concatenate([means,actuals], axis=0))]
        plt.scatter(means, actuals, alpha=0.5)
        plt.plot(ident,ident)
        plt.title('Predicted versus Actual Counts: R2 = {}'.format(str(round(r_squared,2))))
        plt.xlabel("Predicted Counts")
        plt.ylabel("Actual Counts")
        plt.show()
    
    return means,actuals,r_squared

def get_likelihood(mdl, loader, plot = True,save=False,digit=None,ood=False):
    # TODO - more sophisticated ood test of likelihood
    """"'ood' is simply magnifying the value of the image by 100 and checking the loglikelihood is lower"""
    for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
        
        if c.mnist:
            images,labels = data
        else:
            images,dmaps,labels = data
            
        if labels[0] != digit and c.mnist and digit is not None:
            continue
        
        if labels.size: # triggers only if there is at least one annotation
            
            z = labels
            
            images = images.float().to(c.device)
            z = z.float().to(c.device)
            
            mdl = mdl.to(c.device)
            
            x, log_det_jac = mdl(images,z,rev=False)
            
            if ood:
                images_ood = images * 100
                x_ood, log_det_jac_ood = mdl(images_ood,z,rev=False)
                dmap_rev_np_ood = x_ood[0].squeeze().cpu().detach().numpy()
                mean_ll_ood = x_ood[0].mean()

            
            dmap_rev_np = x[0].squeeze().cpu().detach().numpy()
            
            mean_ll = x[0].mean()
            
            if plot:
                
                if ood: 
                    n_plots = 3 
                else: 
                    n_plots = 2
                
                fig, ax = plt.subplots(n_plots,1)
                plt.ioff()
                #fig.suptitle('{} \n mean reconstruction: {}'.format(title,mean_pred),y=1.0,fontsize=24)
                fig.set_size_inches(8*1,12*1)
                fig.set_dpi(100)
                
                if c.mnist:
                    im = images[0].squeeze().cpu().numpy()
                else:
                    im = images[0].permute(1,2,0).cpu().numpy()
                
                if c.mnist:
                    ax[0].imshow(im)
                else:
                    ax[0].imshow((im * 255).astype(np.uint8))
                
                ax[1].hist(dmap_rev_np.flatten(),bins = 30)
                
                if ood:
                    ax[2].hist(dmap_rev_np_ood.flatten(),bins = 30)
                
                if save:
                    if not os.path.exists(g.VIZ_DIR):
                        os.makedirs(g.VIZ_DIR)
                    plt.savefig("{}/{}.jpg".format(g.VIZ_DIR,mdl.modelname), bbox_inches='tight', pad_inches = 0)
            
            break
        
    if ood:
        out = mean_ll,mean_ll_ood # z[0],x[0], images_ood[0],x_ood[0]
    else:
        out = mean_ll # z[0],x[0],
        
    return out # x.size() #

# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


# https://github.com/Devyanshu/image-split-with-overlap/blob/master/split_image_with_overlap.py
# minor edits to function, remove non-square opt, grey scale
def split_image(img,patch_size,save=True,overlap=50,name=None,path=None,frmt=None):
    
    if save:
        assert name and path and frmt
        

    assert 0 <= overlap < 1
    
    splits = []
    
    # insert single channel dim for greyscale
    if len(img.shape) == 2:
        img_h, img_w = img.shape
    else:
        img_h, img_w, _ = img.shape
    
    split_width = patch_size
    split_height = patch_size
    
    def start_points(size, split_size, overlap=overlap):
        points = [0]
        stride = int(split_size * (1-overlap))
        counter = 1
        while True:
            pt = stride * counter
            if pt + split_size >= size:
                points.append(size - split_size)
                break
            else:
                points.append(pt)
            counter += 1
        return points
    
    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)
    
    count = 0
    
    for i in Y_points:
        for j in X_points:
            split = img[i:i+split_height, j:j+split_width]
            
            #if overlap==0:
            splits.append(split) # appending with large image overlaps causes memory issues
            
            if save:
                patch_name = '{}_{}.{}'.format(name, count, frmt)
                cv2.imwrite(path+patch_name, split)
                
                #print('{} saved'.format(patch_name))
                
            count += 1
    
    return splits

def predict_image():
    
    return None

# def stich_image(img_dir=None,img_prefix=None,raw_img_path=None,patch_width=None):
    
#     img_ls = os.listdir(img_dir)
#     patch_name_ls = [[{'name':'','path':''}]]
    
#     for img in img_ls:
    
#       match = re.match(pattern=img_prefix,string=img)
#       if match != None:
#           if match.group() == img_prefix:
#               patch_name_ls.append(img.removeprefix(img_prefix)) # Python 3.9+
              
#     sorted(patch_name_ls, key=lambda x: int(x.split('.', 1)[0]))
    
#     im_width = cv2.imread(raw_img_path).shape[1]
#     row_split = im_width - patch_width
#     counter = 0
#     for patch_name in patch_name_ls:
#         patch = cv2.imread(os.path.join(img_dir,img_prefix,patch_name))
        

          
              
              


     #if re.match(pattern=img_name,string=img):

    
    
    return None
    
    

# For MNIST
# from: https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
# @ptrblck
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
class AddUniformNoise(object):
    def __init__(self, r1=0., r2=1e-2):
        self.r1 = r1
        self.r2 = r2
        
    def __call__(self, tensor):
        # uniform tensor in pytorch: 
        # https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
        return tensor + torch.FloatTensor(tensor.size()).uniform_(self.r1, self.r2)
    
    def __repr__(self):
        return self.__class__.__name__ + '(r1={0}, r2={1})'.format(self.r1, self.r2)

def make_model_name(train_loader):
     now = datetime.now() 
     
     parts = [a.args.schema,
              a.args.model_name,
              os.uname().nodename,
             "BS"+str(train_loader.batch_size),
             "LR_I"+str(a.args.learning_rate),
             "E"+str(a.args.meta_epochs*a.args.sub_epochs),
             "DIM"+str(c.density_map_h)]
     
     if a.args.mod not in ['UNet','FCRN']:
         parts.append("FE",str(c.feat_extractor))
            
     if a.args.mod == 'NF':
         parts.append("NC"+str(c.n_coupling_blocks))
     
     #"LRS",str(c.scheduler), # only one LR scheduler currently
     if a.args.dlr_acd:
         parts.append('DLRACD')
     
     if a.args.mnist:
         parts.append('MNIST')
     
     if c.joint_optim:
         parts.append('JO')
         
     if c.pretrained:
         parts.append('PT')
         
     if c.pyramid:
         parts.append('PY_{}'.format(a.args.n_pyramid_blocks))
         
     if c.counts and not a.args.mnist:
         parts.append('CT')
     
     if c.fixed1x1conv:
         parts.append('1x1')
         
     if c.scale != 1:
         parts.append('SC_{}'.format(c.scale))
         
     if c.dropout_p != 0:
         parts.append('DP_{}'.format(c.dropout_p))
         
     if a.args.weight_decay != 1e-5:
         parts.extend(["WD",str(a.args.weight_decay)])
         
     if c.train_feat_extractor or c.load_feat_extractor_str != '':
         parts.append('FT')
         
     if c.sigma != 4 and not a.args.mnist:
         parts.extend(["FSG",str(c.sigma)])
         
     if c.clamp_alpha != 1.9 and not a.args.mnist:
         parts.extend(["CLA",str(c.clamp_alpha)])
         
     if c.test_train_split != 70 and not a.args.mnist:
         parts.extend(["SPLIT",str(c.test_train_split)])
         
     if a.args.balance and not a.args.mnist:
         parts.append('BL')
            
     parts.append(str(now.strftime("%d_%m_%Y_%H_%M_%S")))
     
     modelname = "_".join(parts)
     
     print("Training Model: ",modelname)
     
     return modelname
 
def make_hparam_dict(val_loader):
    
    hparam_dict = {'schema':a.args.schema,
                        'arch':a.args.model_name,
                        'learning rate init.':a.args.learning_rate,
                        'weight decay':a.args.weight_decay,
                        'batch size':val_loader.batch_size,
                        'image height':c.density_map_h,
                        'image width':c.density_map_w,
                        'joint optimisation?':c.joint_optim,
                        'global average pooling?':c.gap,
                        'scale:':c.scale,
                        'annotations only?':a.args.annotations_only,
                        'pretrained?':c.pretrained,
                        'feature pyramid?':c.pyramid,
                        'feature extractor?':c.feat_extractor,
                        '1x1convs?':c.fixed1x1conv,
                        'conv filters':a.args.filters,
                        'fc_width':c.width,
                        'finetuned?':c.train_feat_extractor,
                        'mnist?':a.args.mnist,
                        'counts?':c.counts,
                        'n pyramid blocks?':a.args.n_pyramid_blocks,
                        'subnet_type?':c.subnet_type,
                        'prop. of data':c.data_prop,
                        'clamp alpha':c.clamp_alpha,
                        'weight decay':a.args.weight_decay,
                        'epochs':a.args.meta_epochs*a.args.sub_epochs,
                        'no. of coupling blocks':c.n_coupling_blocks,
                        'filter sigma':c.sigma,
                        'feat vec length':c.n_feat}
    
    return hparam_dict