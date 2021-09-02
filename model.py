# edited from:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# author: Sasank Chilamkurthy

# imports
from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision.models import alexnet # feature extractor

import config as c

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from freia_funcs import F_fully_connected

def nf_head(input_dim=(c.density_map_w,c.density_map_h),condition_dim=c.n_feat):
    
    # from FrEIA tutorial
    def subnet(dims_in, dims_out):
        
        return nn.Sequential(nn.Linear(dims_in, 128), nn.ReLU(),
                        nn.Linear(128,  128), nn.ReLU(),
                        nn.Linear(128,  dims_out))
            
    nodes = [Ff.InputNode(input_dim[0],input_dim[1],name='input')]
    condition  = Ff.ConditionNode(condition_dim,name = 'condition')
    
    for k in range(c.n_coupling_blocks):
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}, name='permute_{k}'))
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,{'clamp': c.clamp_alpha, 'subnet_constructor':subnet},conditions=condition))
        
    return Ff.ReversibleGraphNet(nodes + [condition,Ff.OutputNode(nodes[-1], name='output')]) 
        
                     

class CowFlow(nn.Module):
    
    def __init__(self):
        super(CowFlow,self).__init__()
        self.feature_extractor = alexnet(pretrained=True,progress=False)
        self.nf = nf_head()   

    # x = attributes, images->features, y = labels = density_maps
    def forward(self,x,condition):
        # no multi-scale architecture (yet) as per differnet paper
        x_cat = list()
        feat_s = self.feature_extractor.features(x) # extract features
        
        if c.debug:
            print(x.size())
            print("feature size....")
            print(feat_s.size())
            
        # global average pooling as described in paper:
        # h x w x d -> 1 x 1 x d
        # see: https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/
        # torch.Size([24, 256, 17, 24])
        
        x_cat.append(torch.mean(feat_s,dim = (2,3))) 
        x = torch.cat(x_cat,dim = 1) # concatenation
        
        if c.debug:
            print("y concatenated size..........")
            print(x.size())
            
        z = self.nf(x,c=condition)
        return z
    
def load_model():
    pass
    
def save_model():
    pass
    
def save_weights():
    pass
        
def load_weights():
    pass

