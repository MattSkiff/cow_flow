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
def plot_preds(mdl, loader, plot = True, save=False,title = "",digit=None,hist=True,sampling="randn",plot_n=None):
    
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
                images,dmaps,labels = data
            
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
                    if (loader.dataset.count and not mdl.gap) or (not c.downsampling and c.subnet_type == 'conv'):
                        dummy_z = (randn(loader.batch_size,c.channels*4,(c.density_map_h) // 2,(c.density_map_w) // 2)).to(c.device) 
                    elif loader.dataset.count and mdl.gap:
                        dummy_z = (randn(loader.batch_size,1)).to(c.device) 
                    elif c.downsampling: # self.downsampling
                        in_channels = 1024
                        ft_dims = ft_dims_select(mdl)
                        if mdl.feat_extractor.__class__.__name__ == 'NothingNet':
                            in_channels = 2                      
                        
                        
                        dummy_z = (randn(loader.batch_size, in_channels,ft_dims[0],ft_dims[1])).to(c.device)
                else:
                    if sampling == 'ones':
                        dummy_z = (torch.ones(loader.batch_size, c.channels*4 , c.density_map_h // 2,c.density_map_w  // 2).to(c.device))
                    elif sampling == "zeros":
                        dummy_z = torch.zeros(loader.batch_size, c.channels*4 , c.density_map_h // 2,c.density_map_w  // 2).to(c.device)
                    elif sampling == "randn":
                        dummy_z = (randn(loader.batch_size, c.channels*4, c.density_map_h // 2,c.density_map_w  // 2)).to(c.device)
                    else:
                        ValueError("Invalid function arg (sampling). Try 'randn', 'zeros' or 'ones'.")
                
                ## sample from model ---
                images = images.float().to(c.device)
                dummy_z = dummy_z.float().to(c.device)
                mdl = mdl.to(c.device)
                x, log_det_jac = mdl(images,dummy_z,rev=True)
                
                if lb_idx == None:
                    lb_idx = 0
                
                if c.one_hot:
                    
                    x = x.argmax(-3).to(torch.float)
                
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
                        mode = torch.mode(x_flat,dim = 1).values.cpu().detach().numpy()
                    
                    # TODO!
                    sum_pred = x[lb_idx].sum()
                    
                    if plot:
                        
                        #assert mdl.subnet_type != 'fc'
                        
                        if mdl.unconditional: 
                            hist = False
                        
                        n_plots = 3+hist-mdl.mnist-mdl.count
                               
                        fig, ax = plt.subplots(n_plots,1)
                        plt.ioff()
                        
                        if mdl.mnist:
                            fig.suptitle('{} \n mode of reconstruction: {}'.format(title,str(mode)),y=1.0,fontsize=24) # :.2f
                        if mdl.count:   
                            fig.suptitle('{} \n Predicted count: {:.2f}'.format(title,mean_pred),y=1.0,fontsize=24)
                        else:
                            fig.suptitle('{} \n Predicted count: {:.2f}'.format(title,sum_pred),y=1.0,fontsize=24)
                            
                        fig.set_size_inches(8*1,12*1)
                        fig.set_dpi(100)
                        
                        if mdl.mnist:
                            im = images[lb_idx].squeeze().cpu().numpy()
                        else:
                            unnorm = UnNormalize(mean =tuple(c.norm_mean),
                                                 std=tuple(c.norm_std))
                            
                            im = unnorm(images[lb_idx])
                            
                            im = 255-im.permute(1,2,0).cpu().numpy()
                         
                        if mdl.feat_extractor.__class__.__name__ == 'NothingNet':
                            dmap_rev_np = dmap_rev_np[1,:,:] # select only data from first duplicated channel
                        
                        if not (loader.dataset.count and mdl.gap):
                            ax[0].imshow((255-dmap_rev_np* 255).astype(np.uint8))#, cmap='viridis', interpolation='nearest')
                        else:
                            fig.delaxes(ax[0])
                        
                        if mdl.count:
                            # no dmaps provided by dataloader if mdl.count
                            ax[1].imshow((im * 255).astype(np.uint8))
                            if hist:
                                ax[2].hist(dmap_rev_np.flatten(),bins = 30)
                        else:
                        
                            if mdl.mnist:
                                ax[1].imshow(im)
                                if hist:
                                    ax[2].hist(dmap_rev_np.flatten(),bins = 30)
                            else:
                                ax[1].imshow((im * 255).astype(np.uint8))
                                ax[2].imshow(dmaps[lb_idx])
                                if hist:
                                    ax[3].hist(dmap_rev_np.flatten(),bins = 30)
                            
                        
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

def counts_preds_vs_actual(mdl,loader,plot=False):
    """Plots predicted versus actual counts from the data and returns the R^2 value. Required: count dataloader and count model."""
    assert mdl.count
    assert loader.dataset.count
    assert not mdl.mnist
    assert mdl.modelname 
    
    means = []
    actuals = []
    
    for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
        
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
