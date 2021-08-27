# edited from:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# author: Sasank Chilamkurthy

# imports
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import alexnet
import matplotlib.pyplot as plt
import time
import os
import copy

import config as c

# pyro NF implementation
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.nn import ConditionalDenseNN

from data_loader import CowObjectsDataset

# attempted to create class for Normalizing Flow in PyTorch, but quickly tied up
#class nf_head(nn.ModuleList):
    

# class nf_head(input_dim=c.n_feat,context_dim=4):
    
#     # create base distribution for NF
#     # context variable for pyro is extracted features from alexnet
#     # https://docs.pyro.ai/en/dev/_modules/pyro/distributions/transforms/affine_coupling.html
#     def __init__(self,input_dim=c.n_feat,context_dim=4):
        
#         split_dim = 6
#         base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
#         param_dims = [input_dim-split_dim, input_dim-split_dim]
#         hypernet = ConditionalDenseNN(split_dim, context_dim, [10*input_dim],param_dims)
#         transform = T.ConditionalAffineCoupling(split_dim, hypernet)
        
#         modules = torch.nn.ModuleList([transform])
#         print(modules)
#         print(transform)
#         self.modules = modules
#         self.transform = transform
#         self.base_dist = base_dist
    
#     def forward(self, x, context):
        
#         flow_dist = dist.ConditionalTransformedDistribution(self.base_dist,[self.transform]).condition(context)
    

class CowFlow(nn.Module):
    
    def __init__(self):
        super(CowFlow,self).__init__()
        self.feature_extractor = alexnet(pretrained=True,progress=False)
        # self.nf = nf_head()
        
    def forward(self,x):
        # no multi-scale architecture (yet) as per differnet paper
        y_cat = list()
        feat_s = self.feature_extractor.features(x) # extract features
        if c.debug:
            print(x.size())
            print("feature size....")
            print(feat_s.size())
            
        # global average pooling as described in paper:
        # h x w x d -> 1 x 1 x d
        # see: https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/
        # torch.Size([24, 256, 17, 24])
        y_cat.append(torch.mean(feat_s,dim = (2,3))) 
        y = torch.cat(y_cat,dim = 1) # concatenation
        if c.debug:
            print("y concatenated size..........")
            print(y.size())
        
        # define NF inside train.py instead
        # z = self.nf(y)
        return y
    
def load_model():
    pass
    
def save_model():
    pass
    
def save_weights():
    pass
        
def load_weights():
    pass

