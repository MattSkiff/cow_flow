# MIT License Marco Rudolph 2021
from torch import randn
import config as c
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch
import numpy as np

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
    def __init__(self, r1=0., r2=1.):
        se.f.r1 = r1
        self.r2 = r2
        
    def __call__(self, tensor):
        # uniform tensor in pytorch: 
        # https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
        r1, r2 = 0, 1
        return tensor + torch.FloatTensor(tensor.size()).uniform_(r1, r2)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def reconstruct_density_map(model, validloader, plot = True, save=True,title = "",digit=None,hist=True,sampling="randn"):
    """


    Parameters
    ----------
    model
        A saved model, either MNISTFlow or CowFlow
    validloader
        A dataloader of validation samples.
    plot : Bool, optional
        Plots the digit that is being predicted. The default is True.
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

    Returns
    -------
    dmap_rev_np:
        the samples used to generate the mean reconstruction.
    
    mean_pred:
        The mean reconstruction

    """
    
    for i, data in enumerate(tqdm(validloader, disable=c.hide_tqdm_bar)):
        
        if c.mnist:
            images,labels = data
        else:
            images,dmaps,labels = data
            
        if labels[0] != digit and c.mnist and digit is not None:
            continue
        
        if labels.size: # triggers only if there is at least one annotation
            
            # Z shape: torch.Size([2, 4, 300, 400]) (batch size = 2)
            
            
            if sampling == 'randn':
                dummy_z = (torch.ones(c.batch_size, 4 , c.density_map_h // 2,c.density_map_w  // 2, requires_grad=True).to(c.device))
            elif sampling == "ones":
                dummy_z = torch.zeros(c.batch_size, 4 , c.density_map_h // 2,c.density_map_w  // 2, requires_grad=True).to(c.device)
            elif sampling == "":
                dummy_z = (randn(c.batch_size, 4 , c.density_map_h // 2,c.density_map_w  // 2, requires_grad=True)).to(c.device)
            else:
                ValueError("Invalid function arg (sampling). Try 'randn', 'zeros' or 'ones'.")
            
            images = images.float().to(c.device)
            dummy_z = dummy_z.float().to(c.device)
            
            model = model.to(c.device)
            
            x, log_det_jac = model(images,dummy_z,rev=True)
            
            dmap_rev_np = x[0].squeeze().cpu().detach().numpy()
            mean_pred = x[0].mean()
            
            if plot:
                
                fig, ax = plt.subplots(3,1)
                plt.ioff()
                fig.suptitle('{} \n mean reconstruction: {}'.format(title,mean_pred),y=1.0,fontsize=24)
                fig.set_size_inches(8*1,12*1)
                fig.set_dpi(100)
                
                if c.mnist:
                    im = images[0].squeeze().cpu().numpy()
                else:
                    im = images[0].permute(1,2,0).cpu().numpy()
                    
                ax[0].imshow(dmap_rev_np, cmap='viridis', interpolation='nearest')
                
                if c.mnist:
                    ax[1].imshow(im)
                else:
                    ax[1].imshow((im * 255).astype(np.uint8))
                
                ax[2].hist(dmap_rev_np.flatten(),bins = 30)
                
                if save:
                    if not os.path.exists(VIZ_DIR):
                        os.makedirs(VIZ_DIR)
                    plt.savefig("{}/{}.jpg".format(VIZ_DIR,c.modelname), bbox_inches='tight', pad_inches = 0)
            
            break
        
    return labels[0],dmap_rev_np, mean_pred