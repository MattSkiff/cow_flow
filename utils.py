# MIT License Marco Rudolph 2021
from torch import randn
import config as c
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch
import numpy as np
import random

VIZ_DIR = './viz'

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None

def get_loss(z, jac):
    '''check equation 4 of the paper why this makes sense - oh and just ignore the scaling here'''
    # in differnet, exponentiate over 'channel' dim (n_feat)
    # here, we exponentiate over channel, height, width to produce single norm val per density map
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,2,3)) - jac) / z.shape[1]

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
    def __init__(self, r1=0., r2=1e-3):
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
        torch.nn.init.xavier_uniform(m.weight)
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

def plot_preds(model, loader, plot = True, save=False,title = "",digit=None,hist=True,sampling="randn",mnist=True,plot_n=None):
    """


    Parameters
    ----------
    model
        A saved model, either MNISTFlow or CowFlow
    loader
        A dataloader of samples.
    plot : Bool, optional
        Whether to return a plot or not
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

    Returns
    -------
    dmap_rev_np:
        the samples used to generate the mean reconstruction.
    
    mean_pred:
        The mean reconstruction

    """
    
    def inner_func(model=model):
        
        idx = random.randint(0,len(loader)-1)
        
        k = 0
        
        for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
     
            if mnist:
                images,labels = data
            else:
                images,dmaps,labels = data
            
            lb_idx = random.randint(0,3)
                
            if not mnist:
                
                # check annotations in batch aren't empty
                lb_idx = 0
                for j in range(loader.batch_size):
                    if len(labels[j]) !=0:
                        lb_idx = j
                        break
                    
                if lb_idx == 0:
                    continue  
            
            if i != idx and plot_n == None and digit == None:
                continue
            
            if (mnist and labels.size) or not mnist: # triggers only if there is at least one annotation
                # Z shape: torch.Size([2, 4, 300, 400]) (batch size = 2)
                
                if not mnist:
                    dummy_z = (randn(c.batch_size[0], 1024,17,24, requires_grad=True)).to(c.device)
                else:
                    if sampling == 'ones':
                        dummy_z = (torch.ones(loader.batch_size, c.channels*4 , c.density_map_h // 2,c.density_map_w  // 2, requires_grad=True).to(c.device))
                    elif sampling == "zeros":
                        dummy_z = torch.zeros(loader.batch_size, c.channels*4 , c.density_map_h // 2,c.density_map_w  // 2, requires_grad=True).to(c.device)
                    elif sampling == "randn":
                        dummy_z = (randn(loader.batch_size, c.channels*4, c.density_map_h // 2,c.density_map_w  // 2, requires_grad=True)).to(c.device)
                    else:
                        ValueError("Invalid function arg (sampling). Try 'randn', 'zeros' or 'ones'.")
                
                images = images.float().to(c.device)
                dummy_z = dummy_z.float().to(c.device)
                
                model = model.to(c.device)
                
                x, log_det_jac = model(images,dummy_z,rev=True)
                
                if c.one_hot:
                    
                    x = x.argmax(-3).to(torch.float)
                
                if  plot_n != None:
                    lb_idx = range(loader.batch_size)
                else:
                    lb_idx = [lb_idx]
                
                for lb_idx in lb_idx:
                
                    dmap_rev_np = x[lb_idx].squeeze().cpu().detach().numpy()
                    mean_pred = x[lb_idx].mean()
                    sum_pred = x[lb_idx].sum()
                    
                    if plot:
                        
                        n_plots = 3+hist-mnist
                               
                        fig, ax = plt.subplots(n_plots,1)
                        plt.ioff()
                        if mnist:
                            fig.suptitle('{} \n mean reconstruction: {:.2f}'.format(title,mean_pred),y=1.0,fontsize=24)
                        else:
        
                            fig.suptitle('{} \n Predicted count: {:.2f}'.format(title,sum_pred),y=1.0,fontsize=24)
                        fig.set_size_inches(8*1,12*1)
                        fig.set_dpi(100)
                        
                        if c.mnist:
                            im = images[lb_idx].squeeze().cpu().numpy()
                        else:
                            im = UnNormalize(im, 
                                             mean =[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                            
                            im = 255-images[lb_idx].permute(1,2,0).cpu().numpy()
                            
                        ax[0].imshow(dmap_rev_np, cmap='viridis', interpolation='nearest')
                        
                        if c.mnist:
                            ax[1].imshow(im)
                            if hist:
                                ax[2].hist(dmap_rev_np.flatten(),bins = 30)
                        else:
                            ax[1].imshow((im * 255).astype(np.uint8))
                            ax[2].imshow(dmaps[lb_idx])
                            if hist:
                                ax[3].hist(dmap_rev_np.flatten(),bins = 30)
                            
                        
                        if save:
                            if not os.path.exists(VIZ_DIR):
                                os.makedirs(VIZ_DIR)
                            plt.savefig("{}/{}.jpg".format(VIZ_DIR,c.modelname), bbox_inches='tight', pad_inches = 0)
                        
                        if mnist:
                            out = labels[lb_idx],dmap_rev_np, mean_pred 
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


def get_likelihood(model, validloader, plot = True,save=False,digit=None,ood=False):
    # TODO - more sophisticated ood test of likelihood
    """"'ood' is simply magnifying the value of the image by 100 and checking the loglikelihood is lower"""
    for i, data in enumerate(tqdm(validloader, disable=c.hide_tqdm_bar)):
        
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
            
            model = model.to(c.device)
            
            x, log_det_jac = model(images,z,rev=False)
            
            if ood:
                images_ood = images * 100
                x_ood, log_det_jac_ood = model(images_ood,z,rev=False)
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
                    if not os.path.exists(VIZ_DIR):
                        os.makedirs(VIZ_DIR)
                    plt.savefig("{}/{}.jpg".format(VIZ_DIR,c.modelname), bbox_inches='tight', pad_inches = 0)
            
            break
        
    if ood:
        out = mean_ll,mean_ll_ood # z[0],x[0], images_ood[0],x_ood[0]
    else:
        out = mean_ll # z[0],x[0],
        
    return out # x.size() #
