# MIT License Marco Rudolph 2021
from torch import randn
import config as c
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None

def get_loss(z, jac):
    '''check equation 4 of the paper why this makes sense - oh and just ignore the scaling here'''
    # in differnet, exponentiate over 'channel' dim (n_feat)
    # here, we exponentiate over channel, height, width to produce single norm val per density map
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,2,3)) - jac) / z.shape[1]
    


GRADIENT_MAP_DIR = './gradient_maps/'

def reconstruct_density_map(model, validloader, optimizer, n_batches=1):
    plt.figure(figsize=(10, 10))
    validloader.dataset.get_fixed = True

    # TODO n batches
    for i, data in enumerate(tqdm(validloader, disable=c.hide_tqdm_bar)):
        optimizer.zero_grad()
        
        images,dmaps,labels = data
        
        if labels[0]: # triggers only if there is at least one annotations
            
            # Z shape: torch.Size([2, 4, 300, 400]) (batch size = 2)
            dummy_z = randn(c.batch_size, 4 , c.density_map_h // 2,c.density_map_w  // 2, requires_grad=True).to(c.device)
            
            images = images.float().to(c.device)
            dummy_z = dummy_z.float().to(c.device)
            
            x, log_det_jac = model(images,dummy_z,rev=True)
            break
        
    return x # reconstructed density map tensor