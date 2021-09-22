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

def reconstruct_density_map(model, validloader, plot = True, save=True,title = ""):
    #plt.figure(figsize=(10, 10))

    # TODO n batches
    for i, data in enumerate(tqdm(validloader, disable=c.hide_tqdm_bar)):
        
        if c.mnist:
            images,labels = data
        else:
            images,dmaps,labels = data
        
        if labels.size: # triggers only if there is at least one annotation
            
            # Z shape: torch.Size([2, 4, 300, 400]) (batch size = 2)
            scale = 1
            #dummy_z = (torch.ones(c.batch_size, 4 , c.density_map_h // 2,c.density_map_w  // 2, requires_grad=True).to(c.device))*scale
            #dummy_z = torch.zeros(c.batch_size, 4 , c.density_map_h // 2,c.density_map_w  // 2, requires_grad=True).to(c.device)
            dummy_z = (randn(c.batch_size, 4 , c.density_map_h // 2,c.density_map_w  // 2, requires_grad=True) *scale).to(c.device)
            
            images = images.float().to(c.device)
            dummy_z = dummy_z.float().to(c.device)
            
            model = model.to(c.device)
            
            x, log_det_jac = model(images,dummy_z,rev=True)
            
            if plot:
            
                dmap_rev_np = x[0].squeeze().cpu().detach().numpy()
                
                fig, ax = plt.subplots(2,1)
                plt.ioff()
                fig.suptitle(title,y=1.0,fontsize=24)
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
                
                if save:
                    if not os.path.exists(VIZ_DIR):
                        os.makedirs(VIZ_DIR)
                    plt.savefig("{}/{}.jpg".format(VIZ_DIR,c.modelname), bbox_inches='tight', pad_inches = 0)
            
            break
        
    return x # reconstructed density map tensor