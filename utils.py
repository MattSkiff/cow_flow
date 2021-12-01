# MIT License Marco Rudolph 2021
from torch import randn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch
import numpy as np
import random
from prettytable import PrettyTable

import config as c
import gvars as g

#from scipy import ndimage as ndi
from skimage.feature import peak_local_max # 
#from skimage import data, img_as_float

def ft_dims_select(mdl=None):
    
    if mdl == None:
        fe = c.feat_extractor
    else:
        fe = mdl.feat_extractor.__class__.__name__
    
    if c.downsampling:
    
        if fe in ['resnet18','ResNetPyramid'] or fe == 'Sequential':
            ft_dims = (19,25)
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

def get_loss(z, jac,dims):
    # in differnet, exponentiate over 'channel' dim (n_feat)  
    # here, we exponentiate over channel, height, width to produce single norm val per density map
    
    return torch.mean(0.5 * torch.sum(z ** 2, dim=dims) - jac) / z.shape[1]

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

# TODO split into minst and non mnist funcs
@torch.no_grad()
def plot_preds(mdl, loader, plot = True, save=False,title = "",digit=None,hist=False,sampling="randn",plot_n=None,writer=None,writer_epoch=None,writer_mode=None):
    
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
            elif loader.dataset.count: # model.dataset.count
                images,dmaps,labels,counts = data
            else:
                images,dmaps,labels, _ = data
            
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
            
            elif not mdl.mnist:
                
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
           
            ## create noise vector ---
            if (mdl.mnist and labels.size) or not mdl.mnist: # triggers only if there is at least one annotation
                # Z shape: torch.Size([2, 4, 300, 400]) (batch size = 2)
                
                if not mdl.mnist:
                    # TODO - remove hard coding dims here
                    if (loader.dataset.count and not mdl.gap) or (not mdl.downsampling and mdl.subnet_type == 'conv'):
                        dummy_z = (randn(loader.batch_size,c.channels*4,(c.density_map_h) // 2,(c.density_map_w) // 2)).to(c.device) 
                    elif loader.dataset.count and mdl.gap:
                        dummy_z = (randn(loader.batch_size,1)).to(c.device) 
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
                        
                        dummy_z = (randn(loader.batch_size, in_channels,ft_dims[0],ft_dims[1])).to(c.device)
                else:
                    
                    if mdl.subnet_type == 'conv':
                       
                        if sampling == 'ones':
                            dummy_z = (torch.ones(loader.batch_size, c.channels*4, c.density_map_h // 2,c.density_map_w  // 2).to(c.device))
                        elif sampling == "zeros":
                            dummy_z = torch.zeros(loader.batch_size, c.channels*4, c.density_map_h // 2,c.density_map_w  // 2).to(c.device)
                        elif sampling == "randn":
                            dummy_z = (randn(loader.batch_size, c.channels*4, c.density_map_h // 2,c.density_map_w  // 2)).to(c.device)
                        else:
                            ValueError("Invalid function arg (sampling). Try 'randn', 'zeros' or 'ones'.")
                    else:
                        dummy_z = (randn(loader.batch_size, c.channels)).to(c.device)
                
                ## sample from model ---
                images = images.float().to(c.device)
                dummy_z = dummy_z.float().to(c.device)
                mdl = mdl.to(c.device)
                x, log_det_jac = mdl(images,dummy_z,rev=True)
                
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
                    lb_idx = range(loader.batch_size)
                else:
                    lb_idx = [lb_idx]
                
                for lb_idx in lb_idx:
                    dmap_rev_np = x[lb_idx].squeeze().cpu().detach().numpy()
                    mean_pred = x[lb_idx].mean()
                    
                    # TODO
                    if (mdl.count or mdl.mnist) and not mdl.gap:
                        x_flat = torch.reshape(x,(loader.batch_size,c.density_map_h * c.density_map_w)) # torch.Size([200, 12, 12])
                        mode = torch.mode(x_flat,dim = 1).values.cpu().detach().numpy()
                    elif mdl.count and mdl.gap:
                        x_flat = x
                                            
                    if mdl.mnist and c.subnet_type == 'conv':
                        x = torch.mode(x,dim = 1).values.cpu().detach().numpy() #print(x.shape) (200,)
                    else:
                        x = torch.mode(x,dim = 0).values.cpu().detach().numpy()
                    
                    # TODO!
                    if mdl.mnist or mdl.subnet_type == 'conv':
                        sum_pred = x.sum() #[lb_idx]
                    elif mdl.subnet_type == 'fc' and mdl.gap:  
                        mode = x
                        sum_pred = x
                        
                    if plot:
                        
                        if mdl.unconditional: 
                            hist = False
                        
                        n_plots = 3+hist-mdl.mnist-mdl.count
                               
                        fig, ax = plt.subplots(n_plots,1)
                        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
                        [axi.set_axis_off() for axi in ax.ravel()] # turn off subplot axes
                        
                        if mdl.mnist:
                            fig.suptitle('{} \n mode of reconstruction: {}'.format(title,str(mode)),y=1.0,fontsize=16) # :.2f
                        if mdl.count:   
                            fig.suptitle('{} \n Predicted count: {:.2f}'.format(title,mean_pred),y=1.0,fontsize=16)
                        else:
                            fig.suptitle('{} \n Predicted count: {:.2f}'.format(title,sum_pred),y=1.0,fontsize=16)
                            
                        fig.set_size_inches(8*1,12*1)
                        fig.set_dpi(100)
                        
                        if mdl.mnist:
                            im = images[lb_idx].squeeze().cpu().numpy()
                        else:
                            unnorm = UnNormalize(mean =tuple(c.norm_mean),
                                                 std=tuple(c.norm_std))
                            
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
                                ax[2].title.set_text('Density Map Ground Truth')
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
                                out = dmaps[lb_idx].sum(),dmap_rev_np, sum_pred
                        
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

@torch.no_grad()
# TODO - add plot full set and save opt
def plot_peaks(mdl, loader,n=1):
    assert not mdl.count
    assert mdl.subnet_type == 'conv'

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
        #x_std_norm = x_std #x_std-x #x_std.div(x) # unused

        for idx in range(loader.batch_size):               
        
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
            gt_coords = annotations[idx].cpu().detach().numpy()
            
            # subtract constrant for uniform noise
            constant = ((1e-3)/2)*ground_truth_dmap.shape[0]*ground_truth_dmap.shape[1]
            sum_count -= constant
            dist_counts -= constant
            gt_count -= constant

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
                ax[1,2].scatter(gt_coords[:,1]*mdl.density_map_w, gt_coords[:,2]*mdl.density_map_h,c='red',marker='1',label='Ground truth coordinates')
                ax[1,2].legend()
                ax[1,2].set_ylim(ax[1,0].get_ylim())
                ax[1,2].set_xlim(ax[1,0].get_xlim())
                ax[1,2].title.set_text('Localizations via Peak Finding on Reconstruction\nN = {}'.format(len(coordinates)))   
                
            ax[0,0].title.set_text('Mean (N={}) Reconstructed Density Map from Aerial Imagery\nSum Density Map = {:.2f}'.format(n,sum_count))
            #ax[1,0].title.set_text('Pixel-wise Uncertainty Estimate (normalised std. dev)')
            ax[0,1].title.set_text('Image Differencing (Reconstruction, Ground Truth Density Map)')
            ax[0,2].title.set_text('Sum (Integration) Prediction Distribution (N = {})'.format(n))
            ax[1,0].title.set_text('Ground Truth Density Map\nN = {} | Sum = {:.2f}'.format(len(labels[idx]),gt_count))
            ax[1,1].title.set_text('Aerial Image')
            
            fig.subplots_adjust(hspace=0.4)
            
            return
        
    return 
        

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
