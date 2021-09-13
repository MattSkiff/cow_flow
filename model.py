# edited from:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# author: Sasank Chilamkurthy

# imports
from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision.models import alexnet # feature extractor

import config as c # hyper params

# FrEIA imports for invertible networks
import FrEIA.framework as Ff
import FrEIA.modules as Fm

# save model 
import os

WEIGHT_DIR = './weights'
MODEL_DIR = './models'

# change default initialisation
# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def nf_head(input_dim=(c.density_map_h,c.density_map_w),condition_dim=c.n_feat):
    
    def subnet(dims_in, dims_out):
        
        # subnet is operating over density map 
        # hence switch from linear to conv2d net
        net = nn.Sequential(
                            nn.Conv2d(dims_in, 32, kernel_size = 3,padding = 1), 
                            nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size = 3,padding = 1), 
                            nn.ReLU(),
                            nn.Conv2d(64, dims_out,kernel_size = 3,padding = 1)
                        )
        

        
        #net.apply(init_weights)
        print('dims in and dims out')
        print(dims_in)
        print(dims_out)
        
        return net
    
    # include batch size as extra dimension here? data is batched along extra dimension
    # input = density maps
    nodes = [Ff.InputNode(1,input_dim[0],input_dim[1],name='input')] 
    
    # haar downsampling to resolves input data only having a single channel
    # affine coupling performs channel wise split
    # https://github.com/VLL-HD/FrEIA/issues/8
    
    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {}, name = 'Downsampling'))
    
    print(nodes[0])
    # first dim should be 1 not c.n_feat -> crashes if don't expand input to condition dims
    
    # 'Because of the construction of the conditional coupling blocks, the condition must have the same spatial dimensions as the data'
    # https://github.com/VLL-HD/FrEIA/issues/9
    
    # condition = exacted image features
    condition  = Ff.ConditionNode(c.n_feat,input_dim[0] // 2,input_dim[1] // 2, name = 'condition')
    
    for k in range(c.n_coupling_blocks):
        print("creating layer {:d}".format(k))
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}, name='permute_{}'.format(k)))
        print(condition)
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,{'clamp': c.clamp_alpha, 'subnet_constructor':subnet},conditions=[condition],
                             name = 'couple_{}'.format(k)))
       
    # perform haar upsampling (reshape) to ensure loss calculation is correct
    # nodes.append(Ff.Node(nodes[-1], Fm.HaarUpsampling, {}, name = 'Upsampling'))
        
    return Ff.ReversibleGraphNet(nodes + [condition,Ff.OutputNode(nodes[-1], name='output')]) 
        
class CowFlow(nn.Module):
    
    def __init__(self):
        super(CowFlow,self).__init__()
        self.feature_extractor = alexnet(pretrained=True,progress=False)
        self.nf = nf_head()   

    # x = attributes, images->features (conditioning), y = labels = density_maps
    def forward(self,x,y):
        # no multi-scale architecture (yet) as per differnet paper
        x_cat = list()
        feat_s = self.feature_extractor.features(x) # extract features
        
        if c.debug:
            print("raw feature size..")
            print(feat_s.size(),"\n")
            
        # global average pooling as described in paper:
        # h x w x d -> 1 x 1 x d
        # see: https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/
        # torch.Size([24, 256, 17, 24])
        
        x_cat.append(torch.mean(feat_s,dim = (2,3))) 
        x = torch.cat(x_cat,dim = 1) # concatenation
        
        # adding spatial dimensions....
        # remove dimension of size one for concatenation in NF coupling layer - squeeze()
        
        if c.debug: 
            print("concatenated and pooled feature size..")
            print(x.size(),"\n")
        
        x = x.unsqueeze(2).unsqueeze(3).expand(-1, -1, c.density_map_h // 2,c.density_map_w // 2)
        
        if c.debug: 
            print("reshaped feature size with spatial dims..")
            print(x.size(),"\n")
        
        # mapping density map dims to match feature dims
        y = y.unsqueeze(1) #.expand(-1,c.n_feat,-1, -1) 
        
        if c.debug: 
            print("expanded density map size")
            print(y.size(),"\n") 
        
        # second argument to NF is the condition - i.e. the features
        # first argument is the 'x' i.e. the data we are mapping to z (NF : x <--> z)
        # is also what we are trying to predict (in a sense we are using 'x' features to predict 'y' density maps)
        # hence ambiguity in notation
        z = self.nf(y,x)
        return z

# unaltered from differnet  
def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR,filename))
    
def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    model = torch.load(path)
    return model
    
def save_weights(model, filename):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    torch.save(model.state_dict(),os.apth.join(WEIGHT_DIR,filename))
        
def load_weights(model, filename):
    path = os.path.join(WEIGHT_DIR, filename)
    model.load_state_dict(torch.load(path))
    return model

