# https://github.com/marco-rudolph/differnet/blob/master/localization.py
import numpy as np
from torch.autograd import Variable
import config as c
from utils import *

import torch
from torch import randn

from utils import t2np, get_loss, ft_dims_select
from data_loader import preprocess_batch,UnNormalize

import matplotlib.pyplot as plt
from tqdm import tqdm
import os
#import copy
from scipy.ndimage import rotate, gaussian_filter

import arguments as a

GRADIENT_MAP_DIR = './gradient_maps/'

def save_imgs(inputs,grad, cnt):
    export_dir = os.path.join(GRADIENT_MAP_DIR, a.args.mdl_path)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    normed_grad = (grad - np.min(grad)) / (
            np.max(grad) - np.min(grad))
        
    orig_image = inputs
    for image, file_suffix in [(normed_grad, '_gradient_map.png'), (orig_image, '_orig.png')]:
        plt.clf()
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(os.path.join(export_dir, str(cnt) + file_suffix), bbox_inches='tight', pad_inches=0)
    cnt += 1
    return cnt


def export_gradient_maps(mdl, val_loader, n_batches=1): # export_gradient_maps
    plt.figure(figsize=(10, 10))
    cnt = 0
    # TODO n batches
    for i, data in enumerate(tqdm(val_loader, disable=c.hide_tqdm_bar)):
        
        optimizer = torch.optim.Adam(mdl.parameters(), lr=a.args.learning_rate, betas=(a.args.adam_b1, a.args.adam_b2), eps=a.args.adam_e, weight_decay=a.args.weight_decay)
        optimizer.zero_grad()
                
        images,dmaps,labels, binary_labels, annotations,point_maps = preprocess_batch(data)
        images = images.float().to(c.device)
        
        in_channels = 1024
        ft_dims = ft_dims_select(mdl)
        dummy_z = (randn(images.size()[0], in_channels,ft_dims[0],ft_dims[1])).to(c.device)

        images = Variable(images, requires_grad=True) 
        #dmaps = Variable(dmaps, requires_grad=True) 

        inputs = (images,dummy_z) # inputs features,dmaps
        
        # generate density map - eval
        density_maps, log_det_jac  = mdl(*inputs,jac=True,rev=True)
        
        #density_maps.retain_grad()
        
        inputs = (images,density_maps.squeeze(0)) # warning: if bs=1, will squeeze out batch dim
        
        # generate z from model - train
        z, log_det_jac = mdl(*inputs,jac=a.args.jac,rev=False)
            
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True) [FrEIA]
        # i.e. what y (output) is depends on direction of flow
        
        # this loss needs to calc distance between predicted density and density map
        # note: probably going to get an error with mnist or counts # TODO
        dims = tuple(range(1, len(z.size()))) # z.size()
        loss = get_loss(z, log_det_jac,dims) # z
        
        # dims = tuple(range(1, len(dummy_z.size()))) # z.size()
        # loss = get_loss(dummy_z, log_det_jac,dims) # z
        
        #inputs = Variable(inputs, requires_grad=True)
        loss.backward()

        # replace 'inputs' with images
        # grad = grad[binary_labels > 0] # this filters out all gradients right now
        
        # if grad.shape[0] == 0:
        #     continue
        
        #grad = density_maps.grad
        grad = images.grad #.view(-1, c.n_transforms_test, *inputs.shape[-3:])
        grad = t2np(grad)
        
        for i in range(images.size()[0]):
            
            print(len(annotations[i]))
            
            im_grad = grad[i,:,:,:]
            # print('im grad size')
            # print(im_grad.shape)
            
            #images.view(-1, 1, *images.shape[-3:])[:, 0]
            #images = np.transpose(t2np(images[binary_labels > 0]), [0, 2, 3, 1])
            
            unnorm = UnNormalize(mean=tuple(c.norm_mean),
                                 std=tuple(c.norm_std))
            
            im = images[i].detach().clone() #copy.deepcopy(images[i])
            im = unnorm(im)
            # print(grad.shape)
            
            # for i_item in range(1):
            #     old_shape = grad[:, i_item].shape
            #     img = np.reshape(grad[:, i_item], [-1, *grad.shape[-2:]])
            #     img = np.transpose(img, [1, 2, 0])
            #     img = gaussian_filter(img, (0, 3, 3))
            #     grad[:, i_item] = np.reshape(img, old_shape)
                
            # print(im_grad.shape)
            # print(np.count_nonzero(im_grad))
            # print(im_grad)
            # grad = np.reshape(grad, [grad.shape[0], -1, *grad.shape[-2:]])
            im_grad = np.mean(np.abs(im_grad), axis=0)
            im_grad_sq = im_grad ** 2
            
            im = im.permute(1,2,0)
            im = im.cpu().detach().numpy()

            cnt = save_imgs(im, im_grad_sq, cnt)

        # if i == n_batches:
        #     break

    plt.close()