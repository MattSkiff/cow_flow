# edited from:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# author: Sasank Chilamkurthy

# imports
from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision.models import alexnet, resnet18, vgg16_bn  # feature extractors

import torch.nn.functional as F

from torchvision.models.resnet import ResNet, BasicBlock

import config as c # hyper params
from utils import init_weights

# FrEIA imports for invertible networks
import FrEIA.framework as Ff
import FrEIA.modules as Fm

import os # save model
import shutil
import dill # solve error when trying to pickle lambda function in FrEIA

# https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial/
class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)
        
class NothingNet():
    def __init__(self):
        self.features = torch.nn.Identity()

WEIGHT_DIR = './weights'
MODEL_DIR = './models'
C_DIR = './cstates'

if c.feat_extractor == "alexnet":
    feat_extractor = alexnet(pretrained=c.pretrained,progress=False).to(c.device)
elif c.feat_extractor == "resnet18":
    feat_extractor = resnet18(pretrained=c.pretrained,progress=False).to(c.device)
elif c.feat_extractor == "vgg16_bn":
    feat_extractor = vgg16_bn(pretrained=c.pretrained,progress=False).to(c.device)
elif c.feat_extractor == "mnist_resnet":
    feat_extractor = MnistResNet()
elif c.feat_extractor == "none":
    feat_extractor = NothingNet()

def sub_conv2d(dims_in,dims_out):
    net = nn.Sequential(
        nn.Conv2d(dims_in, 32, kernel_size = 3,padding = 1), 
        nn.BatchNorm2d(32),
        nn.ReLU(),
        # TODO batch norm
        nn.Conv2d(32, 64, kernel_size = 3,padding = 1), 
        nn.BatchNorm2d(64),
        nn.ReLU(),
        # TODO batch norm here
        nn.Conv2d(64, dims_out,kernel_size = 3,padding = 1)
        )
    
    net.apply(init_weights)
    
    return net
    
def sub_fc(dims_in,dims_out,internal_size):
    # debugging
    net = nn.Sequential(
        nn.Linear(dims_in, dims_out), # internal_size
        #nn.ReLU(),
        #nn.Dropout(),
        #nn.Linear(internal_size, internal_size), 
        #nn.ReLU(),
        #nn.Dropout(),
        #nn.Linear(internal_size,dims_out)
        )
    
    return net

def nf_head(input_dim=(c.density_map_h,c.density_map_w),condition_dim=c.n_feat,mnist=False):
    
    def subnet(dims_in, dims_out):
        
        # subnet is operating over density map 
        # hence switch from linear to conv2d net
        net = sub_conv2d(dims_in,dims_out)

        #net.apply(init_weights)
        if c.debug:
            print('dims in: {}, dims out: {}'.format(dims_in,dims_out))
        
        return net
    
    # include batch size as extra dimension here? data is batched along extra dimension
    # input = density maps
    nodes = [Ff.InputNode(c.channels,input_dim[0],input_dim[1],name='input')] 
    
    # haar downsampling to resolves input data only having a single channel (from unsqueezed singleton dimension)
    # affine coupling performs channel wise split
    # https://github.com/VLL-HD/FrEIA/issues/8
    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {}, name = 'Downsampling'))
    
    # 'Because of the construction of the conditional coupling blocks, the condition must have the same spatial dimensions as the data'
    # https://github.com/VLL-HD/FrEIA/issues/9
    
    # condition = exacted image features
    condition = Ff.ConditionNode(condition_dim,input_dim[0] // 2,input_dim[1] // 2, name = 'condition')
        
    for k in range(c.n_coupling_blocks):
        if c.verbose:
            print("creating layer {:d}".format(k))
            
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}, name='permute_{}'.format(k)))
        
        if c.verbose:
            print(condition)
            
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,{'clamp': c.clamp_alpha, 'subnet_constructor':subnet},conditions=[condition],
                             name = 'couple_{}'.format(k)))
       
    # perform haar upsampling (reshape) to ensure loss calculation is correct
    # nodes.append(Ff.Node(nodes[-1], Fm.HaarUpsampling, {}, name = 'Upsampling'))
        
    return Ff.ReversibleGraphNet(nodes + [condition,Ff.OutputNode(nodes[-1], name='output')], verbose=c.verbose) 
        
class CowFlow(nn.Module):
    
    def __init__(self):
        super(CowFlow,self).__init__()
        self.feature_extractor = feat_extractor
        self.nf = nf_head()   

    def forward(self,images,dmaps,rev=False):
        # no multi-scale architecture (yet) as per differnet paper
        
        if c.debug:
            print("images size..")
            print(images.size(),"\n")
        
        # x = raw images, y = density maps
        feat_cat = list()
        feat_s = self.feature_extractor.features(images) # extract features
        
        if c.debug:
            print("raw feature size..")
            print(feat_s.size(),"\n")
            
        # global average pooling as described in paper:
        # h x w x d -> 1 x 1 x d
        # see: https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/
        # torch.Size([batch_size, 256, 17, 24])
        
        if c.feat_extractor != "none":
            feat_cat.append(torch.mean(feat_s,dim = (2,3))) 
        
        feats = torch.cat(feat_cat,dim = 1) # concatenation (does nothing at single scale feature extraction)
        
        # adding spatial dimensions....
        # remove dimension of size one for concatenation in NF coupling layer - squeeze()
        
        if c.debug: 
            print("concatenated and pooled feature size..")
            print(feats.size(),"\n")
        
        if c.feat_extractor != "none":
            feats = feats.unsqueeze(2).unsqueeze(3).expand(-1, -1, c.density_map_h // 2,c.density_map_w // 2)
        
        if c.debug: 
            print("reshaped feature size with spatial dims..")
            print(feats.size(),"\n")
        
        # mapping density map dims to match feature dims
        # introduces singleton dimension
        # don't want to introduce this dimension when running z -> x
        if not rev:
            dmaps = dmaps.unsqueeze(1) #.expand(-1,c.n_feat,-1, -1) 
        
        if c.debug: 
            print("expanded density map size")
            print(dmaps.size(),"\n") 
        
        # second argument to NF is the condition - i.e. the features
        # first argument is the 'x' i.e. the data we are mapping to z (NF : x <--> z)
        # is also what we are trying to predict (in a sense we are using 'x' features to predict 'y' density maps)
        # hence ambiguity in notation
        
        z = self.nf(x_or_z = dmaps,c = feats,rev=rev)
        return z

class MNISTFlow(nn.Module):
    
    def __init__(self):
        super(MNISTFlow,self).__init__()
        self.feature_extractor = feat_extractor
        self.nf = nf_head(mnist=True)   

    def forward(self,images,labels,rev=False):
        
        feat_cat = list()
        
        if c.feat_extractor != "none":
            images = images.expand(-1,3,-1,-1) 
        
        if c.debug:
            print('preprocessed mnist imgs size')
            print(images.size(),"\n")
        
        feat_s = self.feature_extractor.features(images) # extract features
        

        if c.debug:
            print("raw feature size..")
            print(feat_s.size(),"\n")
             
        # global average pooling as described in paper:
        # h x w x d -> 1 x 1 x d
        # see: https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/
        # torch.Size([batch_size, 256, 17, 24])
        
        if c.feat_extractor != "none":
            feat_cat.append(torch.mean(feat_s,dim = (2,3))) 
        else:
            feat_cat.append(feat_s) 
            
        feats = torch.cat(feat_cat,dim = 1) # concatenation
        
        # adding spatial dimensions....
        # remove dimension of size one for concatenation in NF coupling layer - squeeze()
        
        if c.debug: 
            print("concatenated and pooled feature size..")
            print(feats.size(),"\n")
        
        if not rev:
            if c.feat_extractor != "none":
                feats = feats.unsqueeze(2).unsqueeze(3).expand(-1, -1, c.density_map_h // 2,c.density_map_w // 2)
        
        if c.debug:
            print("feats size..")
            print(feats.size(),"\n") 
         
        if c.debug:
            print('labels size...')
            print(labels.size())   
         
        if not rev:
            if c.one_hot:
                labels = F.one_hot(labels.to(torch.int64),num_classes=10)
                labels = labels.unsqueeze(2).unsqueeze(3).expand(-1, -1, c.density_map_h,c.density_map_w).to(torch.float)
            else:
                labels = labels.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, -1, c.density_map_h,c.density_map_w)
        
        if c.debug: 
            print("expanded labels size")
            print(labels.size(),"\n") 
        
        z = self.nf(x_or_z = labels,c = feats,rev=rev)
        return z

def save_cstate(cdir,config_file):
    ''' saves a snapshot of the config file before running and saving model '''
    if not os.path.exists(C_DIR):
        os.makedirs(C_DIR)
        
    base, extension = os.path.splitext(config_file)
    
    if c.verbose:
        'Config file copied to {}'.format(C_DIR)
    
    new_name = os.path.join(cdir, base+"_"+c.modelname+".txt")
    
    shutil.copy(config_file, new_name)

def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR,filename), pickle_module=dill)
    save_cstate(cdir=C_DIR,config_file="config.py")
    
def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    model = torch.load(path, pickle_module=dill)
    return model
    
def save_weights(model, filename):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    torch.save(model.state_dict(),os.path.join(WEIGHT_DIR,filename), pickle_module=dill)
        
def load_weights(model, filename):
    path = os.path.join(WEIGHT_DIR, filename)
    model.load_state_dict(torch.load(path,pickle_module=dill))
    return model