# MIT License Marco Rudolph 2021
import config as c
import torch
import numpy as np
from pyro import distributions as T

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None

def get_loss(z, jac):
    '''check equation 4 of the paper why this makes sense - oh and just ignore the scaling here'''
    # in differnet, exponentiate over 'channel' dim (n_feat)
    # here, we exponentiate over channel, height, width to produce single norm val per density map
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,2,3)) - jac) / z.shape[1]

# def preprocess_batch(data):
#     '''move data to device and reshape image'''
#     images,annotations,classes = data
#     images,annotations,classes = images.to(c.device), torch.cat(annotations,dim = 0).to(c.device), classes #annotations.to(c.device),classes.to(c.device)
#     images = images.view(-1, *images.shape[-3:])
#     return images, annotations, classes