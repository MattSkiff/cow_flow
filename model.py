# edited from:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# author: Sasank Chilamkurthy

# imports
from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision.models import alexnet, resnet18, vgg16_bn  # feature extractors
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn.functional as F

import os # save model
import collections
import shutil
import dill # solve error when trying to pickle lambda function in FrEIA
from types import MethodType # overwrtie private forward method

from utils import init_weights, ft_dims_select

import train # train_feat_extractor
# importing all fixes cyclical import 

# FrEIA imports for invertible networks VLL/HDL
import FrEIA.framework as Ff
import FrEIA.modules as Fm

import config as c # hyper params
import arguments as a
import gvars as g

# TODO - shift below 5 util functions to utils
def save_cstate(cdir,modelname,config_file):
    ''' saves a snapshot of the config file before running and saving model '''
    if not os.path.exists(g.C_DIR):
        os.makedirs(g.C_DIR)
        
    base, extension = os.path.splitext(config_file)
    
    if c.verbose:
        'Config file copied to {}'.format(g.C_DIR)
    
    new_name = os.path.join(cdir, base+"_"+modelname+".txt")
    
    shutil.copy(config_file, new_name)

def save_model(model,filename,loc=g.MODEL_DIR):
    if not os.path.exists(loc):
        os.makedirs(loc)
    torch.save(model, os.path.join(loc,filename), pickle_module=dill)
    print("model {} saved to {}".format(filename,loc))
    save_cstate(cdir=g.C_DIR,modelname="",config_file="config.py")
    
def load_model(filename,loc=g.MODEL_DIR):
    path = os.path.join(loc, filename)
    model = torch.load(path, pickle_module=dill)
    print("model {} loaded from {}".format(filename,loc))
    return model
    
def save_weights(model,filename,loc=g.WEIGHT_DIR):
    if not os.path.exists(loc):
        os.makedirs(loc)
    torch.save(model.state_dict(),os.path.join(loc,filename), pickle_module=dill)
        
def load_weights(model, filename,loc=g.WEIGHT_DIR):
    path = os.path.join(loc, filename)
    model.load_state_dict(torch.load(path,pickle_module=dill))
    return model

# https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial/
# TODO
class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)
     
# TODO add argument for loading saved finetuned resnet18 from storage
# ty ptrblck  https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/6
# ty Zeeshan Khan Suri https://zshn25.github.io/ResNet-feature-pyramid-in-Pytorch/
def _forward_impl_my(self, x):
    # change forward here
    x = self.conv1(x)
    x = self.bn1(x)
    x0 = self.relu(x)
    x = self.maxpool(x0)

    x1 = self.layer1(x)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)
    
    return [x0,x1,x2,x3,x4]       
        
class ResNetPyramid(ResNet):

    def __init__(self):
        
        super(ResNetPyramid, self).__init__(BasicBlock, [2, 2, 2, 2])
        
        if c.load_feat_extractor_str == '':
            self.load_state_dict(resnet18(pretrained=c.pretrained).state_dict())
        elif c.load_feat_extractor_str != '':
            # TODO
            num_ftrs = self.fc.in_features
            self.fc = nn.Linear(num_ftrs, 2)  
            finetuned_fe = load_model(c.load_feat_extractor_str,loc=g.FEAT_MOD_DIR)
            self.load_state_dict(finetuned_fe.state_dict())
            del finetuned_fe
    
#    def forward(self, x):
#        
#        return self._forward_impl(x)
 
class NothingNet():
    def __init__(self):
        self.features = torch.nn.Identity()

def select_feat_extractor(feat_extractor,train_loader=None,valid_loader=None):
    
    if c.load_feat_extractor_str and not c.pyramid:
        feat_extractor = load_model(filename=c.load_feat_extractor_str,loc=g.FEAT_MOD_DIR)
    
    if not c.pyramid:
        if c.feat_extractor == "alexnet":
            feat_extractor = alexnet(pretrained=c.pretrained,progress=False).to(c.device)
        elif c.feat_extractor == "resnet18":
             # last but one layer of resnet -> features
             feat_extractor = resnet18(pretrained=c.pretrained,progress=False)
        elif c.feat_extractor == "vgg16_bn":
            feat_extractor = vgg16_bn(pretrained=c.pretrained,progress=False).to(c.device)
        elif c.feat_extractor == "mnist_resnet":
            feat_extractor = MnistResNet()
        elif c.feat_extractor == "none":
            feat_extractor = NothingNet()
    else:
        feat_extractor = ResNetPyramid()
        feat_extractor._forward_impl = MethodType(_forward_impl_my, feat_extractor)
    
    if c.train_feat_extractor:
    # pretrain feature extractor with classification problem
        num_ftrs = feat_extractor.fc.in_features
        feat_extractor.fc = nn.Linear(num_ftrs, 2)  
        feat_extractor = train.train_feat_extractor(feat_extractor,train_loader,valid_loader)
        
    return feat_extractor

def sub_conv2d(dims_in,dims_out,n_filters):
    # naming pytorch layers:
    # https://stackoverflow.com/questions/66152766/how-to-assign-a-name-for-a-pytorch-layer/66162559#66162559
    network_dict = collections.OrderedDict(
                [
                    ("conv1", nn.Conv2d(dims_in, n_filters, kernel_size = 3,padding = 1)), 
                    ('batchnorm1',nn.BatchNorm2d(n_filters)),
                    ("relu1", nn.ReLU()),
                    ("conv2", nn.Conv2d(c.filters, n_filters*2, kernel_size = 3,padding = 1)),
                    ('batchnorm2',nn.BatchNorm2d(n_filters*2)),
                    ("relu2", nn.ReLU()),
                    ("conv3", nn.Conv2d(n_filters*2, dims_out,kernel_size = 3,padding = 1))
                ]
        )
    
    # batchnorm works poorly for very small minibatches, so may want to disable
    if not c.batchnorm:
        del network_dict['batchnorm1']
        del network_dict['batchnorm2']
    
    net = nn.Sequential(network_dict)
    net.apply(init_weights)
    
    # zero init last subnet weights as per glow, cINNs paper
    net.conv3.weight = torch.nn.init.zeros_(net.conv3.weight)
    net.conv3.bias.data.fill_(0.00)
    
    return net

def subnet(dims_in, dims_out):
    
    # subnet is operating over density map 
    # hence switch from linear to conv2d net
    if c.subnet_type == 'conv':
        net = sub_conv2d(dims_in,dims_out,c.filters)
    elif c.subnet_type == 'fc':
        net = sub_fc(dims_in,dims_out,c.width)

    if c.debug:
        print('dims in: {}, dims out: {}'.format(dims_in,dims_out))
    
    return net
   
def sub_fc(dims_in,dims_out,internal_size):
    # debugging
    net = nn.Sequential(
        nn.Linear(dims_in, internal_size), # internal_size
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(internal_size, internal_size), 
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(internal_size,dims_out)
        )
    
    return net

# TODO: turn func into loop
def nf_pyramid(input_dim=(c.density_map_h,c.density_map_w),condition_dim=c.n_feat):
    assert c.subnet_type == 'conv'
    assert not c.gap and not c.counts and not c.mnist
    
    # TODO - will break because of ref to config file
    mdl = ResNetPyramid()
    mdl._forward_impl = MethodType(_forward_impl_my, mdl)
    feats = mdl(torch.randn(c.batch_size[0],3,c.density_map_h,c.density_map_w,requires_grad=False))
    del mdl
    
    if c.verbose:
        print('Feature Pyramid Dimensions:')
        [print(f.shape) for f in feats]
        
    p_dims = [f.shape for f in feats]
    
    nodes = [Ff.InputNode(c.channels,input_dim[0],input_dim[1],name='input')] 
    
    # Conditions
    condition0 = [Ff.ConditionNode(p_dims[0][1],p_dims[0][2],p_dims[0][3],name = 'condition0')]
    condition1 = [Ff.ConditionNode(p_dims[1][1],p_dims[1][2],p_dims[1][3],name = 'condition1')]
    condition2 = [Ff.ConditionNode(p_dims[2][1],p_dims[2][2],p_dims[2][3],name = 'condition2')]
    condition3 = [Ff.ConditionNode(p_dims[3][1],p_dims[3][2],p_dims[3][3],name = 'condition3')]
    condition4 = [Ff.ConditionNode(p_dims[4][1],p_dims[4][2],p_dims[4][3],name = 'condition4')]
    
    # block 1
    k=0
    nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}, name='permute_{}'.format(k)))
    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {}, name = 'Downsampling_{}'.format(k)))
    nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,{'clamp': c.clamp_alpha, 
                         'subnet_constructor':subnet},conditions=condition0,name = 'couple_{}'.format(k)))
      
    # block 2
    k+=1
    nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}, name='permute_{}'.format(k)))
    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {}, name = 'Downsampling_{}'.format(k)))
    nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,{'clamp': c.clamp_alpha, 
                         'subnet_constructor':subnet},conditions=condition1,name = 'couple_{}'.format(k)))
     
    # block 3
    k+=1
    nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}, name='permute_{}'.format(k)))
    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {}, name = 'Downsampling_{}'.format(k)))
    nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,{'clamp': c.clamp_alpha, 
                         'subnet_constructor':subnet},conditions=condition2,name = 'couple_{}'.format(k)))
    
    # block 4
    k+=1
    nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}, name='permute_{}'.format(k)))
    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {}, name = 'Downsampling_{}'.format(k)))
    nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,{'clamp': c.clamp_alpha, 
                         'subnet_constructor':subnet},conditions=condition3,name = 'couple_{}'.format(k)))
    
    # block 5
    k+=1
    nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}, name='permute_{}'.format(k)))
    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {}, name = 'Downsampling_{}'.format(k)))
    nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,{'clamp': c.clamp_alpha, 
                         'subnet_constructor':subnet},conditions=condition4,name = 'couple_{}'.format(k)))
    
    out = Ff.ReversibleGraphNet(nodes + condition0 + condition1 + condition2 + condition3 + condition4 + [Ff.OutputNode(nodes[-1], name='output')], verbose=c.verbose)
    
    return out
    
    
def nf_head(input_dim=(c.density_map_h,c.density_map_w),condition_dim=c.n_feat,mnist=False):
    
    # include batch size as extra dimension here? data is batched along extra dimension
    # input = density maps / mnist labels 
    
    # 'Because of the construction of the conditional coupling blocks, the condition must have the same spatial dimensions as the data'
    # https://github.com/VLL-HD/FrEIA/issues/9
    
    # condition = exacted image features
    if c.mnist or (c.counts and not c.gap):
        condition = [Ff.ConditionNode(condition_dim,input_dim[0] // 2,input_dim[1] // 2, name = 'condition')]
    elif c.counts and c.gap:
        condition = [Ff.ConditionNode(condition_dim,name = 'condition')]
    else:
        # TODO: avoid hardcoding feature spatial dimensions in
        ft_dims = ft_dims_select()
        
        condition = [Ff.ConditionNode(condition_dim,ft_dims[0],ft_dims[1],name = 'condition')]
    
    if c.counts and c.gap:
        nodes = [Ff.InputNode(1,name='input')] # single count value
    else:
        nodes = [Ff.InputNode(c.channels,input_dim[0],input_dim[1],name='input')] 
    
    # haar downsampling to resolves input data only having a single channel (from unsqueezed singleton dimension)
    # affine coupling performs channel wise split
    # https://github.com/VLL-HD/FrEIA/issues/8
    if c.mnist or c.counts or c.feat_extractor == 'none' or c.downsampling and c.subnet_type == 'conv':
        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {}, name = 'Downsampling'))
            
    elif not c.counts and c.feat_extractor != 'none' and c.downsampling:
        # downsamples density maps (not needed for counts)
        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {}, name = 'Downsampling1'))
        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {}, name = 'Downsampling2'))
        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {}, name = 'Downsampling3'))
        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {}, name = 'Downsampling4'))
        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {}, name = 'Downsampling5'))
        
    for k in range(c.n_coupling_blocks):
        if c.verbose:
            print("creating layer {:d}".format(k))
        
        # Don't need permutation layer if flow is univariate
        if not (c.counts and c.subnet_type == 'fc'):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}, name='permute_{}'.format(k)))
        
        if c.verbose:
            print(condition)
            
        if a.args.unconditional:
            condition = None
        
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,{'clamp': c.clamp_alpha, 'subnet_constructor':subnet},conditions=condition,
                            name = 'couple_{}'.format(k)))
        
        #nodes.append(Ff.Node(nodes[-1], Fm.AllInOneBlock,{'subnet_constructor':subnet},conditions=[condition],name = 'allin1_{}'.format(k)))
       
    # perform haar upsampling (reshape) to ensure loss calculation is correct
    # nodes.append(Ff.Node(nodes[-1], Fm.HaarUpsampling, {}, name = 'Upsampling'))
    if a.args.unconditional:
        out = Ff.ReversibleGraphNet(nodes + [Ff.OutputNode(nodes[-1], name='output')], verbose=c.verbose)
    else:
        out = Ff.ReversibleGraphNet(nodes + condition + [Ff.OutputNode(nodes[-1], name='output')], verbose=c.verbose) 
        
    return out
        
class CowFlow(nn.Module):
    
    def __init__(self,modelname,feat_extractor):
        super(CowFlow,self).__init__()
        
        if c.feat_extractor == 'resnet18' and not c.pyramid:
            modules = list(feat_extractor.children())[:-2]
            self.feat_extractor = nn.Sequential(*modules)
        else:
            self.feat_extractor = feat_extractor
        
        if c.pyramid:
            self.nf = nf_pyramid()   
        else:
            self.nf = nf_head()  
        self.modelname = modelname
        self.unconditional = a.args.unconditional
        self.count = c.counts
        self.subnet_type = c.subnet_type
        self.mnist = False
        self.gap = c.gap
        self.n_coupling_blocks = c.n_coupling_blocks
        self.joint_optim = c.joint_optim
        self.pretrained = c.pretrained
        self.finetuned = c.train_feat_extractor
        self.scheduler = c.scheduler
        self.pyramid = c.pyramid

    def forward(self,images,labels,rev=False): # label = dmaps or counts
        # no multi-scale architecture (yet) as per differnet paper
        
        if c.debug:
            print("images size..")
            print(images.size(),"\n")
        
        # x = raw images, y = density maps
        feat_cat = list()
        
        if self.feat_extractor.__class__.__name__ != 'Sequential' and not self.pyramid:
            feat_s = self.feat_extractor.features(images) # extract features
        else:
            feat_s = self.feat_extractor(images)
        
        if c.debug and not self.pyramid:
            print("raw feature size..")
            print(feat_s.size(),"\n")
            
        if c.debug and self.pyramid:
            print("feats list len:")
            print(len(feat_s))
            
        # global average pooling as described in paper:
        # h x w x d -> 1 x 1 x d
        # see: https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/
        # vary according to input image
        # (alexnet) torch.Size([batch_size, 256, 17, 24])
        # (vgg16-bn) torch.Size([batch_size, 512, 18, 25]) 
        # (resnet18) torch.Size([batch_size, 512, 19, 25]) 
        
        if self.feat_extractor.__class__.__name__  == "NothingNet" or self.pyramid:
            feats = feat_s
        else:
            
            if self.gap:
                feat_cat.append(torch.mean(feat_s,dim = (2,3))) 
            else:
                feat_cat.append(feat_s)
             
            feats = torch.cat(feat_cat) # concatenation (does nothing at single scale feature extraction)

        if c.debug and not c.pyramid: 
            print("concatenated and pooled feature size..")
            print(feats.size(),"\n")
        
        # adding spatial dimensions....
        # remove dimension of size one for concatenation in NF coupling layer - squeeze()
        if c.feat_extractor != "none" and c.subnet_type =='conv':
        
            if self.gap:
                feats = feats.unsqueeze(2).unsqueeze(3).expand(-1, -1, c.density_map_h // 2,c.density_map_w // 2)
        
        if c.debug and not self.pyramid: 
            print("reshaped feature size with spatial dims..")
            print(feats.size(),"\n")
        
        # mapping density map dims to match feature dims
        # introduces singleton dimension
        # don't want to introduce this dimension when running z -> x
        if not rev:
            labels = labels.unsqueeze(1) #.expand(-1,c.n_feat,-1, -1) 
        
        if c.debug:
            print('label sizes:')
            print(labels.size(),"\n")
        
        if c.counts and not rev and c.subnet_type == 'conv':
            # expand counts out to spatial dims of feats
            labels = labels.unsqueeze(2).unsqueeze(3).expand(-1,-1,feats.size()[2] * 2,feats.size()[3] * 2)
            
        if self.unconditional and not c.downsampling and not self.count and c.subnet_type == 'conv': 
            labels = labels.expand(-1,c.channels,-1,-1) # expand dmap over channel dimension
            
        if c.debug: 
            print("expanded label (dmap/counts) map size")
            print(labels.size(),"\n") 
        
        if c.debug and c.pyramid:
            print("length of feature vec pyramid tensor list")
            print(len(feats))
        # second argument to NF is the condition - i.e. the features
        # first argument is the 'x' i.e. the data we are mapping to z (NF : x <--> z)
        # is also what we are trying to predict (in a sense we are using 'x' features to predict 'y' density maps)
        # hence ambiguity in notation
        
        if c.pyramid:
            # feats = list of 4 tensors
            z = self.nf(x_or_z = labels,c = feats,rev=rev)
        elif self.unconditional:
            z = self.nf(x_or_z = labels,rev=rev)
        else:
            z = self.nf(x_or_z = labels,c = feats,rev=rev)
          
        return z

class MNISTFlow(nn.Module):
    
    def __init__(self,modelname,feat_extractor):
        super(MNISTFlow,self).__init__()
        self.feat_extractor = feat_extractor
        self.nf = nf_head(mnist=True) 
        self.mnist = True
        self.modelname = modelname
        self.gap = c.gap
        self.n_coupling_blocks = c.n_coupling_blocks
        self.joint_optim = c.joint_optim
        self.pretrained = c.pretrained
        self.scheduler = c.scheduler

    def forward(self,images,labels,rev=False):
        
        feat_cat = list()
        
        if c.feat_extractor != "none":
            images = images.expand(-1,3,-1,-1) 
        
        if c.debug:
            print('preprocessed mnist imgs size')
            print(images.size(),"\n")
        
        feat_s = self.feat_extractor.features(images) # extract features
        

        if c.debug:
            print("raw feature size..")
            print(feat_s.size(),"\n")
             
        # global average pooling as described in paper:
        # h x w x d -> 1 x 1 x d
        # see: https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/
        
        # alex net
        # torch.Size([batch_size, 256, 17, 24])
        # resnet
        #torch.Size([4, 512, 19, 25])
        
        if c.feat_extractor != "none":
            
            if c.gap:
                feat_cat.append(torch.mean(feat_s,dim = (2,3))) 
            else:
                feat_cat.append(feat_s)
                
        feats = torch.cat(feat_cat,dim = 1) # concatenation (does nothing at single scale feature extraction)
        
        # adding spatial dimensions....
        # remove dimension of size one for concatenation in NF coupling layer - squeeze()
        
        if c.debug: 
            print("concatenated and pooled feature size..")
            print(feats.size(),"\n")
        
        if not rev:
            if c.feat_extractor != "none":
                if c.gap:
                    feats = feats.unsqueeze(2).unsqueeze(3).expand(-1, -1, c.density_map_h // 2,c.density_map_w // 2)
        
        if c.debug:
            print("feats size..")
            print(feats.size(),"\n") 
         
        if c.debug:
            print('labels size...')
            print(labels.size())   
        
        if not c.gap and c.feat_extractor == 'none' and not c.one_hot:
            labels = labels.unsqueeze(2).unsqueeze(3).expand(-1, -1, c.density_map_h * 2,c.density_map_h * 2)
        
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