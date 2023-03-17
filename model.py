 # edited from:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# author: Sasank Chilamkurthy

# imports
from __future__ import print_function, division
import os
from datetime import datetime 
import collections

import torch
from torch import Tensor
import torch.nn as nn 
from torchvision.models import alexnet, resnet18, resnet50, vgg16_bn, efficientnet_b3   # feature extractors
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models.vgg import VGG, make_layers, cfgs
import torch.nn.functional as F

import numpy as np

# FrEIA imports for invertible networks VLL/HDL
import FrEIA.framework as Ff
import FrEIA.modules as Fm

# Internal
import utils as u #init_weights, ft_dims_select
import train # train_feat_extractor
import config as c # hyper params
import arguments as a
import gvars as g
import baselines as b
from baselines import MCNN, UNet

# https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial/
# TODO
class ResNetPyramidClassificationHead(ResNet):

    def __init__(self):
        
        super(ResNetPyramidClassificationHead, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 1, 2) # block.expansion, # no. classes
        
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.avgpool(x[4]) # classify based on last output features 
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
class VGGPyramidClassificationHead(VGG):

    def __init__(self):
        
        super(VGGPyramidClassificationHead, self).__init__(make_layers(cfgs["D"]))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=True),
            nn.Linear(4096, 1),
        )
        
    def _forward_impl(self, x: Tensor) -> Tensor:
            x = self.features(x[4])
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
        
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
       
class ResNetPyramid(ResNet):

    def __init__(self,config={}):
        
        if a.args.mode != 'search':
            config['feat_extractor'] = a.args.feat_extractor 
            config['freeze_bn'] = a.args.freeze_bn
        
        if config['feat_extractor']  == 'resnet9':
            block_init = [1,1,1,1]
        elif config['feat_extractor']  == 'resnet18':
            block_init = [2,2,2,2]
        
        if config['feat_extractor']  in ['resnet18','resnet9']:
            super(ResNetPyramid, self).__init__(BasicBlock,block_init, num_classes=1000) 
        elif config['feat_extractor']  == 'resnet50':
            super(ResNetPyramid, self).__init__(Bottleneck,[3, 4, 6, 3], num_classes=1000) 

        #super(ResNetPyramid, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=1000)
        
        if a.args.load_feat_extractor_str == '':
            if a.args.fe_load_imagenet_weights:
                if config['feat_extractor'] == 'resnet18':
                    self.load_state_dict(resnet18(pretrained=a.args.pretrained).state_dict())
                elif config['feat_extractor']  == 'resnet50':
                    self.load_state_dict(resnet50(pretrained=a.args.pretrained).state_dict())
                else:
                    ValueError
                    
                num_ftrs = self.fc.in_features
                self.fc = nn.Linear(num_ftrs, 2)  
            
        elif a.args.load_feat_extractor_str != '':
            finetuned_fe = u.load_model(a.args.load_feat_extractor_str,loc=g.FEAT_MOD_DIR)
            self.load_state_dict(finetuned_fe.state_dict())
            num_ftrs = self.fc.in_features
            self.fc = nn.Linear(num_ftrs, 2)  
            del finetuned_fe
            
        # borrowed from LCFCN
        # # FREEZE BATCH NORMS
        if config['freeze_bn']:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
            
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
    
        return [x0,x1,x2,x3,x4]#[:c.levels]  
    
    def forward(self, x):
        
        return self._forward_impl_my(x)
 
class NothingNet():
    def __init__(self):
        self.features = torch.nn.Identity()
  
class VGGPyramid(VGG):

    def __init__(self,freeze_bn=True):
        super(VGGPyramid, self).__init__(make_layers(cfgs["D"]))
        
        if a.args.load_feat_extractor_str == '':
            if a.args.fe_load_imagenet_weights:
                self.load_state_dict(vgg16_bn(pretrained=a.args.pretrained,progress=False).state_dict())
                num_ftrs = self.fc.in_features
                self.fc = nn.Linear(num_ftrs, 2)  
            
        elif a.args.load_feat_extractor_str != '':
            finetuned_fe = u.load_model(a.args.load_feat_extractor_str,loc=g.FEAT_MOD_DIR)
            self.load_state_dict(finetuned_fe.state_dict())
            num_ftrs = self.fc.in_features
            self.fc = nn.Linear(num_ftrs, 2)  
            del finetuned_fe
        
        # borrowed from LCFCN
        # # FREEZE BATCH NORMS
        if freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
    
    def _forward_impl_my(self, x: torch.Tensor) -> torch.Tensor:
        
        x0 = self.features[0:7](x)
        x1 = self.features[7:14](x0)
        x2 = self.features[14:18](x1)
        x3 = self.features[18:28](x2)
        x4 = self.features[28:44](x3)

        return [x0,x1,x2,x3,x4]    
    
    def forward(self, x):
        
        return self._forward_impl_my(x)

# from https://github.com/VLL-HD/conditional_INNs
def random_orthog(n):
    w = np.random.randn(n, n)
    w = w + w.T
    
    # unitary (w), vectors with singular (S), unitary (w)
    w, S, V = np.linalg.svd(w)
    
    out = torch.FloatTensor(w)
    out.requires_grad_()
    
    return out 

def select_feat_extractor(feat_extractor,train_loader=None,valid_loader=None,config={}):
    
    if a.args.mode != 'search':
        config['freeze_bn'] = a.args.freeze_bn
        config['feat_extractor'] = a.args.feat_extractor 
    
    if a.args.load_feat_extractor_str and not a.args.pyramid:
        feat_extractor = u.load_model(filename=a.args.load_feat_extractor_str,loc=g.FEAT_MOD_DIR)
    
    if not a.args.pyramid:
        if config['feat_extractor']  == "alexnet":
            feat_extractor = alexnet(pretrained=a.args.pretrained,progress=False).to(c.device)
        elif config['feat_extractor']  == "resnet18":
             # last but one layer of resnet -> features
             feat_extractor = resnet18(pretrained=a.args.pretrained,progress=False)
        elif config['feat_extractor']  == "vgg16_bn":
            feat_extractor = vgg16_bn(pretrained=a.args.pretrained,progress=False).to(c.device)
        elif config['feat_extractor']  == "mnist_resnet":
            feat_extractor = MnistResNet(config=config)
        elif config['feat_extractor']  == "none":
            feat_extractor = NothingNet(config=config)
    else:
        if config['feat_extractor']  == 'vgg16_bn':
            feat_extractor = VGGPyramid(freeze_bn=config['freeze_bn'])
        else:
            feat_extractor = ResNetPyramid(config=config)
        
        return feat_extractor
    
    if c.train_feat_extractor:
    # pretrain feature extractor with classification problem
        if config['feat_extractor']  == "resnet18":
            num_ftrs = feat_extractor.fc.in_features
            feat_extractor.fc = nn.Linear(num_ftrs, 2)  
        elif config['feat_extractor']  == "vgg16_bn":
            # find less horific model surgery
            num_ftrs = feat_extractor.classifier._modules['6'].in_features
            feat_extractor.classifier._modules['6'] = nn.Linear(num_ftrs, 2)
        
        feat_extractor = train.train_feat_extractor(feat_extractor,train_loader,valid_loader)
        
    return feat_extractor

def sub_conv2d_shallow(dims_in,dims_out,n_filters):
    # naming pytorch layers:
    # https://stackoverflow.com/questions/66152766/how-to-assign-a-name-for-a-pytorch-layer/66162559#66162559
    
    
    network_dict = collections.OrderedDict(
                [
                    ("conv1", nn.Conv2d(dims_in, n_filters, kernel_size = 1,padding = 0)), 
                    ("relu1", nn.ReLU()),
                    ("conv3", nn.Conv2d(n_filters, dims_out,kernel_size = 1,padding = 0)) 
                ]
        )
    
    net = nn.Sequential(network_dict)
    net.apply(u.init_weights)
    
    # zero init last subnet weights as per glow, cINNs paper
    net.conv3.weight = torch.nn.init.zeros_(net.conv3.weight)
    net.conv3.bias.data.fill_(0.00) 
    
    return net

def sub_conv2d(dims_in,dims_out,config={}): # edited for comptability with search.py
    
    # naming pytorch layers:
    # https://stackoverflow.com/questions/66152766/how-to-assign-a-name-for-a-pytorch-layer/66162559#66162559
    network_dict = collections.OrderedDict(
                [
                    ("conv1", nn.Conv2d(dims_in, config['filters'], kernel_size = 3,padding = 1)), 
                    ('batchnorm1',nn.BatchNorm2d(config['filters'])),
                    ("relu1", nn.ReLU()),
                    # use small filters in subnet per glow paper
                    ("conv2", nn.Conv2d(config['filters'], config['filters']*2, kernel_size = 1,padding = 0)),
                    ('batchnorm2',nn.BatchNorm2d(config['filters']*2)),
                    ("relu2", nn.ReLU()),
                    ("conv3", nn.Conv2d(config['filters']*2, dims_out,kernel_size = 3,padding = 1))
                ]
        )
    
    # batchnorm works poorly for very small minibatches, so may want to disable
    if not config['subnet_bn']:
        del network_dict['batchnorm1']
        del network_dict['batchnorm2']
    
    net = nn.Sequential(network_dict)
    net.apply(u.init_weights)
    
    # zero init last subnet weights as per glow, cINNs paper
    net.conv3.weight = torch.nn.init.zeros_(net.conv3.weight)
    net.conv3.bias.data.fill_(0.00) 
    
    return net

# TODO
# need to work out dimensionality here - try depthwise conv?
# def sub_conv2d_deep(dims_in,dims_out,n_filters):
#     # naming pytorch layers:
#     # https://stackoverflow.com/questions/66152766/how-to-assign-a-name-for-a-pytorch-layer/66162559#66162559
#     network_dict = collections.OrderedDict(
#                 [
#                     ("conv1", nn.Conv2d(dims_in, n_filters, kernel_size = 3,padding = 1)), 
#                     ('batchnorm1',nn.BatchNorm2d(n_filters)),
#                     ("relu1", nn.ReLU()),
#                     # use small filters in subnet per glow paper
                    
#                     ("conv2", nn.Conv2d(n_filters, n_filters, kernel_size = 1,padding = 'same',groups=n_filters)),
#                     ('batchnorm2',nn.BatchNorm2d(n_filters)),
#                     ("relu2", nn.ReLU()),
                    
#                     ("conv3", nn.Conv2d(n_filters,n_filters, kernel_size = 1,padding = 'same',groups=n_filters)),
#                     ('batchnorm3',nn.BatchNorm2d(n_filters)),
#                     ("relu2", nn.ReLU()),
                        
#                     ("conv4", nn.Conv2d(n_filters,n_filters*2, kernel_size = 1,padding = 0)),
#                     ('batchnorm4',nn.BatchNorm2d(n_filters*2)),
#                     ("relu4", nn.ReLU()),
                    
#                     ("conv3", nn.Conv2d(n_filters*2, dims_out,kernel_size = 3,padding = 1))
#                 ]
#         )
    
    # batchnorm works poorly for very small minibatches, so may want to disable
    
    # if not c.batchnorm:
    #     del network_dict['batchnorm1']
    #     del network_dict['batchnorm2']
    #     del network_dict['batchnorm3']
    #     del network_dict['batchnorm4']
    
    # net = nn.Sequential(network_dict)
    # net.apply(u.init_weights)
    
    # # zero init last subnet weights as per glow, cINNs paper
    # net.conv3.weight = torch.nn.init.zeros_(net.conv3.weight)
    # net.conv3.bias.data.fill_(0.00) 
    
    # return net

def subnet(dims_in, dims_out,filters=None):
    
    config = {}
    
    # have checked global var works correctly (remove if possible)
    if not a.args.mode == 'search':
        config['subnet_bn'] = a.args.subnet_bn
        config['filters'] = a.args.filters
    else:
        config['filters'] = g.FILTERS # gnarly, but the only way (I know) to pass into the subnet constructor in FrEIA 
        config['subnet_bn'] = g.SUBNET_BN
    
    # subnet is operating over density map 
    # hence switch from linear to conv2d net
    if a.args.subnet_type == 'conv':
        net = sub_conv2d(dims_in,dims_out,config=config)
    if a.args.subnet_type == 'conv_shallow':
        net = sub_conv2d_shallow(dims_in,dims_out,filters)
    # if a.args.subnet_type == 'conv_deep':
    #     net = sub_conv2d_deep(dims_in,dims_out,filters)
    if a.args.subnet_type == 'fc':
        net = sub_fc(dims_in,dims_out,c.width)
    if a.args.subnet_type == 'MCNN':
        net = sub_mcnn(dims_in,dims_out,filters)
    if a.args.subnet_type == 'UNet':
        net = sub_unet(dims_in,dims_out,filters)
    if a.args.subnet_type == 'nothing':
        net = sub_null(dims_in,dims_out,filters)
        
    if c.debug:
        print('dims in: {}, dims out: {}'.format(dims_in,dims_out))
    
    return net

# TODO
def sub_null(dims_in,dims_out,internal_size):
    
    net = nn.Sequential(
        nn.Identity()
        )
    
    return net

def sub_nf(dims_in,dims_out,internal_size):
    
    return

def sub_fc(dims_in,dims_out,internal_size):
    # debugging
    net = nn.Sequential(
        nn.Linear(dims_in, internal_size), # internal_size
        nn.ReLU(),
        nn.Dropout(p=c.dropout_p),
        nn.Linear(internal_size, internal_size), 
        nn.ReLU(),
        nn.Dropout(p=c.dropout_p),
        nn.Linear(internal_size,dims_out)
        )
    
    return net

def sub_mcnn(dims_in,dims_out,internal_size):
    
    net = MCNN(modelname="subnet",dims_in=dims_in,dims_out=dims_out)
    net.apply(u.init_weights)
    net.fuse[0].weight = torch.nn.init.zeros_(net.fuse[0].weight)
    net.fuse[0].bias.data.fill_(0.00) 
    
    return net
    
def sub_unet(dims_in,dims_out,internal_size):
    
    net = UNet(modelname="subnet",n_channels=dims_in,dims_out=dims_out,subnet=True)
    # net.apply(u.init_weights)
    # net.fuse[0].weight = torch.nn.init.zeros_(net.fuse[0].weight)
    # net.fuse[0].bias.data.fill_(0.00) 
    
    return net
    

def nf_pyramid(input_dim=(c.density_map_h,c.density_map_w),condition_dim=u.n_feats_select(),config = {}):
    
    if not a.args.mode == 'search':
        assert a.args.subnet_type in g.SUBNETS
        config['n_pyramid_blocks'] = a.args.n_pyramid_blocks
    
    assert not c.gap and not c.counts and not a.args.data == 'mnist'
    
    if config['feat_extractor']  in ['resnet18','resnet50','resnet9']:
        mdl = ResNetPyramid(config=config)
    else:
        mdl = VGGPyramid()
        
    # TODO: take out hardcoding activation channels (64)
    channels = 3
    feats = mdl(torch.randn(int(config['batch_size']),channels,c.density_map_h,c.density_map_w,requires_grad=False))
    del mdl
    
    if c.verbose:
        print('Feature Pyramid Dimensions:')
        [print(f.shape) for f in feats]
        
    p_dims = [f.shape for f in feats]
    #assert p_dims[0][1] == channels
    
    nodes = [Ff.InputNode(c.channels,input_dim[0],input_dim[1],name='input')] 
    conditions = []; 
    
    for k in range(c.levels):
        conditions.append(Ff.ConditionNode(p_dims[k][1],p_dims[k][2],p_dims[k][3],name = 'Condition{}'.format(k)))
        
        if a.args.all_in_one:
            
            # All in One
            nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarDownsampling, {}, name = 'Main_Downsampling_{}'.format(k)))
            
            for j in range(config['n_pyramid_blocks']):
                nodes.append(Ff.Node(nodes[-1], Fm.AllInOneBlock,
                                {'affine_clamping': config['clamp'],'subnet_constructor':subnet},
                                conditions=conditions[k],name = 'AllInOne_{}'.format(k)))
            
        else:
            
            # Non 'All in One'
            if config['fixed1x1conv'] and k != 0: # c.channels*4**k
                nodes.append(Ff.Node(nodes[-1].out0, Fm.Fixed1x1Conv,{'M': random_orthog(c.channels*4**k).to(c.device) }, name='1x1_Conv_{}'.format(k)))
            else:
                nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom, {'seed': k}, name='Permute_{}'.format(k)))
            
            nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarDownsampling, {}, name = 'Main_Downsampling_{}'.format(k)))
            
            for j in range(config['n_pyramid_blocks']):
                nodes.append(Ff.Node(nodes[-1].out0, Fm.GLOWCouplingBlock,
                                {'clamp': config['clamp'],'subnet_constructor':subnet},
                                conditions=conditions[k],name = 'Couple_{}'.format(k)))
    
    inn = Ff.GraphINN(nodes + conditions + [Ff.OutputNode(nodes[-1].out0, name='output')], verbose=c.verbose)
    
    return inn

def nf_pyramid_split(input_dim=(c.density_map_h,c.density_map_w),condition_dim=u.n_feats_select(),config={}):
    
    
    if not a.args.mode == 'search':
        assert a.args.subnet_type in g.SUBNETS
        config['n_pyramid_blocks'] = a.args.n_pyramid_blocks
        config['batch_size'] = a.args.batch_size
        
    assert not c.gap and not c.counts and not a.args.data == 'mnist'
    
    if a.args.mode != 'search':
        config['clamp'] = c.clamp_alpha
    
    # TODO - will break because of ref to config file
    mdl = ResNetPyramid()
    # TODO: take out hardcoding activation channels (64)
    channels = 3
    feats = mdl(torch.randn(config['batch_size'],channels,c.density_map_h,c.density_map_w,requires_grad=False))
    del mdl
    
    if c.verbose:
        print('Feature Pyramid Dimensions:')
        [print(f.shape) for f in feats]
        
    p_dims = [f.shape for f in feats]
    
    nodes = [Ff.InputNode(c.channels,input_dim[0],input_dim[1],name='input')] 
    conditions = []; transformed_splits = []; splits_ds = []; split = []
    
    for k in range(c.levels):
        conditions.append(Ff.ConditionNode(p_dims[k][1],p_dims[k][2],p_dims[k][3],name = 'Condition{}'.format(k)))
        
        if config['fixed1x1conv'] and k != 0: # c.channels*4**k
            nodes.append(Ff.Node(nodes[-1].out0, Fm.Fixed1x1Conv,{'M': random_orthog(c.channels*4**k).to(c.device) }, name='1x1_Conv_{}'.format(k)))
        else:
            nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom, {'seed': k}, name='Permute_{}'.format(k)))
        
        nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarDownsampling, {}, name = 'Main_Downsampling_{}'.format(k)))
        
        # split off noise dimensions (see Dinh, 2016) - multiscale architecture (splits along channel, splits into two [default])
        split.append(Ff.Node(nodes[-1].out0, Fm.Split, {}, name='Split_{}'.format(k)))
        
        # deal with split 5 untransformed output here
        if k==4:
            untransformed_split = split[-1]
                
        # after every split, out1 sent to coupling, out0 split off for downsampling and later concatenation
        for j in range(config['n_pyramid_blocks']):
            
            if j == 0:
                
                nodes.append(Ff.Node(split[-1].out0, Fm.GLOWCouplingBlock,
                                     {'clamp': config['clamp'],'subnet_constructor':subnet},
                                     conditions=conditions[k],name = 'Couple_{}_{}'.format(k,j)))
                
                for i in range(4-k):
                    
                    if i == 0:
                        splits_ds.append(Ff.Node(split[-1].out1, Fm.HaarDownsampling, {}, name = 'Split_Downsampling_{}_{}_{}'.format(k,j,i)))
                    else:
                        splits_ds.append(Ff.Node(splits_ds[-1].out0, Fm.HaarDownsampling, {}, name = 'Split_Downsampling_{}_{}_{}'.format(k,j,i)))
                 
                if k != 4:
                    transformed_splits.append(splits_ds[-1])
            
            else:
                
                nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                             {'clamp': config['clamp'],'subnet_constructor':subnet},
                                             conditions=conditions[k],name = 'Couple_{}_{}'.format(k,j)))
    
    # concatenate list of split chain outputs back to output node           
    nodes.append(Ff.Node([nodes[-1].out0,  # last coupling - link up main chain to split chains
                           untransformed_split.out1,
                           transformed_splits[0].out0,
                           transformed_splits[1].out0,
                           transformed_splits[2].out0,
                           transformed_splits[3].out0]
                           ,Fm.Concat,{},name='Concat_Splits'))
    
    nodes.append(Ff.OutputNode(nodes[-1].out0, name='output'))
                          
    # node properties: output_dims, input_dims, outputs, inputs  
    inn = Ff.GraphINN(conditions + split + splits_ds + nodes, verbose=c.verbose)
    
    return inn

def nf_no_fe(input_dim=(c.density_map_h,c.density_map_w),condition_dim=3,mnist=False,config={}):
    
    if a.args.mode != 'search':
        config['clamp'] = c.clamp_alpha
    
    condition = [Ff.ConditionNode(condition_dim,input_dim[0],input_dim[1], name = 'condition')]

    nodes = [Ff.InputNode(1,input_dim[0],input_dim[1],name='input')] 

    for k in range(c.n_coupling_blocks):
                  
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}, name='permute_{}'.format(k)))
        
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,{'clamp': config['clamp'], 'subnet_constructor':subnet},conditions=condition,
                            name = 'couple_{}'.format(k)))
        
        out = Ff.ReversibleGraphNet(nodes + condition + [Ff.OutputNode(nodes[-1], name='output')], verbose=c.verbose) 
    
    return out
    

def nf_head(input_dim=(c.density_map_h,c.density_map_w),condition_dim=u.n_feats_select(),mnist=False,config={}):
    
    if a.args.mode != 'search':
        config['clamp'] = c.clamp_alpha
        
    if a.args.mode == 'search':
        config['clamp']
    
    # include batch size as extra dimension here? data is batched along extra dimension
    # input = density maps / mnist labels 
    
    # 'Because of the construction of the conditional coupling blocks, the condition must have the same spatial dimensions as the data'
    # https://github.com/VLL-HD/FrEIA/issues/9
    
    # condition = exacted image features
    if (a.args.data == 'mnist' or (c.counts and not c.gap)) and a.args.subnet_type in g.SUBNETS: # TODO - update with config dict for search.py compat
        condition = [Ff.ConditionNode(condition_dim,input_dim[0] // 2,input_dim[1] // 2, name = 'condition')]
    elif (c.counts and c.gap) or (a.args.subnet_type == 'fc' and a.args.data == 'mnist'):
        condition = [Ff.ConditionNode(condition_dim,name = 'condition')]
    else:
        # TODO: avoid hardcoding feature spatial dimensions in
        ft_dims = u.ft_dims_select(config=config)
        
        condition = [Ff.ConditionNode(condition_dim,ft_dims[0],ft_dims[1],name = 'condition')]
    
    if c.counts and c.gap:
        nodes = [Ff.InputNode(1,name='input')] # single count value
    else:
        nodes = [Ff.InputNode(c.channels,input_dim[0],input_dim[1],name='input')] 
#    else:
#        nodes = [Ff.InputNode(c.channels,name='input')] 
    
    # haar downsampling to resolves input data only having a single channel (from unsqueezed singleton dimension)
    # affine coupling performs channel wise split
    # https://github.com/VLL-HD/FrEIA/issues/8
    if (a.args.data == 'mnist' or (c.counts and not c.gap) or config['feat_extractor']  == 'none' or not c.downsampling) and a.args.subnet_type in g.SUBNETS:
        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {}, name = 'Downsampling'))
        
    elif not c.counts and config['feat_extractor']  != 'none' and c.downsampling:
            
        # downsamples density maps (not needed for counts)
        for i in range(c.levels):
            nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {}, name = 'Downsampling{}'.format(i+1)))

    for k in range(c.n_coupling_blocks):
        if c.verbose:
            print("creating layer {:d}".format(k))
            
        if not c.downsampling:
            multiplier = 4
        else:
            multiplier = 4**c.levels
        
        if not (c.counts and a.args.subnet_type == 'fc') and config['fixed1x1conv'] and k % a.args.freq_1x1 == 0:
            nodes.append(Ff.Node(nodes[-1], Fm.Fixed1x1Conv,{'M': random_orthog(c.channels*multiplier).to(c.device) }, name='1x1_conv_{}'.format(k)))
        else:
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}, name='permute_{}'.format(k)))
        
        if c.verbose:
            print(condition)
            
        if a.args.unconditional:
            condition = None
        
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,{'clamp': config['clamp'], 'subnet_constructor':subnet},conditions=condition,
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
    
    def __init__(self,modelname,feat_extractor,config = {}):
        super(CowFlow,self).__init__()
        
        if config == {}:
            config['joint_optim'] = a.args.joint_optim
            config['fixed1x1conv'] = a.args.fixed1x1conv
            config['optimiser'] = a.args.optim
            config['noise'] = a.args.noise
            config['scheduler'] = a.args.scheduler
            config['subnet_bn'] = a.args.subnet_bn
            config['filters'] = a.args.filters
            config['feat_extractor'] = a.args.feat_extractor
        
        if config['feat_extractor']  == 'resnet18' and not a.args.pyramid:
            modules = list(feat_extractor.children())[:-2]
            self.feat_extractor = nn.Sequential(*modules)
        else:
            self.feat_extractor = feat_extractor
        
        if config['feat_extractor']  == 'resnet18' and a.args.binary_classifier:
            self.classification_head = resnet18(pretrained=a.args.pretrained,progress=False) #ResNetPyramidClassificationHead()
        elif a.args.binary_classifier:
            self.classification_head = vgg16_bn(pretrained=a.args.pretrained,progress=False)
        else:
            self.classification_head = None
        
        if a.args.data == 'dlr':
            self.dlr_acd = True
        else:
            self.dlr_acd = False
        
        # these attr's are needed to make the model object independant of the config file
        self.modelname = modelname
        self.unconditional = a.args.unconditional
        self.count = c.counts
        self.subnet_type = a.args.subnet_type
        self.mnist = False
        self.gap = c.gap
        self.n_coupling_blocks = c.n_coupling_blocks
        self.joint_optim = config['joint_optim'] 
        self.pretrained = a.args.pretrained
        self.finetuned = c.train_feat_extractor
        self.scheduler = config['scheduler'] #
        self.pyramid = a.args.pyramid
        self.fixed1x1conv = config['fixed1x1conv']
        self.scale = c.scale
        self.density_map_h = c.density_map_h
        self.density_map_w = c.density_map_w
        self.downsampling = c.downsampling
        self.noise = config['noise'] #
        self.sigma = a.args.sigma
        self.seed = c.seed
        self.optim = config['optimiser'] #
        self.dmap_scaling = a.args.dmap_scaling
        self.subnet_bn = config['subnet_bn'] 
        
        # if a.args.temperature:
        #     self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        if a.args.pyramid:
            if a.args.split_dimensions:
                self.nf = nf_pyramid_split(condition_dim=u.n_feats_select(config['feat_extractor']),config=config) 
            else:
                self.nf = nf_pyramid(input_dim=(self.density_map_h,self.density_map_w),condition_dim=u.n_feats_select(config['feat_extractor']),config=config)   
        elif config['feat_extractor']  == 'none':
            self.nf = nf_no_fe()
        else:
            self.nf = nf_head(config=config,condition_dim=u.n_feats_select(config['feat_extractor']))  

    def forward(self,images,labels,rev=False,jac=False): # label = dmaps or counts
        
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

        if c.debug and not a.args.pyramid: 
            print("concatenated and pooled:{} feature size..".format(c.gap))
            print(feats.size(),"\n")
        
        # adding spatial dimensions....
        # remove dimension of size one for concatenation in NF coupling layer squeeze()
        if a.args.subnet_type =='conv': # self.feat_extractor_name 
        
        # a.args.feat_extractor  != "none" -> don't think model can be trained without feature extractor at this stage
        
            if self.gap:
                feats = feats.unsqueeze(2).unsqueeze(3).expand(-1, -1, (c.density_map_h) // 2,(c.density_map_w) // 2)
        
        # mapping density map dims to match feature dims
        # introduces singleton dimension
        # don't want to introduce this dimension when running z -> x
        if not rev:
            labels = labels.unsqueeze(1) #.expand(-1,u.n_feats_select(),-1, -1) 
        
        if c.counts and not rev and a.args.subnet_type in g.SUBNETS:
            # expand counts out to spatial dims of feats
            labels = labels.unsqueeze(2).unsqueeze(3).expand(-1,-1,feats.size()[2] * 2,feats.size()[3] * 2)
            
        if self.unconditional and not c.downsampling and not self.count and a.args.subnet_type in g.SUBNETS: 
            labels = labels.expand(-1,c.channels,-1,-1) # expand dmap over channel dimension
            
        # second argument to NF is the condition - i.e. the features
        # first argument is the 'x' i.e. the data we are mapping to z (NF : x <--> z)
        # is also what we are trying to predict (in a sense we are using 'x' features to predict 'y' density maps)
        # hence ambiguity in notation
        
        if a.args.pyramid:
            # feats = list of 5 tensors
            z = self.nf(x_or_z = labels,c = feats,rev=rev,jac=jac) # [-c.levels:]
        elif self.unconditional:
            z = self.nf(x_or_z = labels,rev=rev,jac=jac)
        else:
            z = self.nf(x_or_z = labels,c = feats,rev=rev,jac=jac)
          
        return z

class MNISTFlow(nn.Module):
    
    def __init__(self,modelname,feat_extractor):
        super(MNISTFlow,self).__init__()
        
        if a.args.feat_extractor  == 'resnet18':
            modules = list(feat_extractor.children())[:-2]
            self.feat_extractor = nn.Sequential(*modules)
        else:
            self.feat_extractor = feat_extractor
        
        self.nf = nf_head(mnist=True) 
        self.mnist = True
        self.count = False
        self.unconditional = a.args.unconditional
        self.modelname = modelname
        self.gap = c.gap
        self.n_coupling_blocks = c.n_coupling_blocks
        self.fixed1x1conv = a.args.fixed1x1conv
        self.joint_optim = a.args.joint_optim
        self.pretrained = a.args.pretrained
        self.scheduler = a.args.scheduler

    def forward(self,images,labels,rev=False):
        
        feat_cat = list()
        
        if a.args.feat_extractor  != "none":
            images = images.expand(-1,3,-1,-1) 
        
        if c.debug:
            print('preprocessed mnist imgs size')
            print(images.size(),"\n")
        
        if self.feat_extractor.__class__.__name__ != 'Sequential':
            feat_s = self.feat_extractor.features(images) # extract features
        else:
            feat_s = self.feat_extractor(images)
        
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
        
        if a.args.feat_extractor  != "none":
            
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
            if a.args.feat_extractor  != "none" and a.args.subnet_type in g.SUBNETS:
                if c.gap:
                    feats = feats.unsqueeze(2).unsqueeze(3).expand(-1, -1, c.density_map_h // 2,c.density_map_w // 2)
        
        if c.debug:
            print("feats size..")
            print(feats.size(),"\n") 
         
        if c.debug:
            print('labels size...')
            print(labels.size())   
        
        if not c.gap and a.args.feat_extractor  == 'none' and not c.one_hot:
            labels = labels.unsqueeze(2).unsqueeze(3).expand(-1, -1, c.density_map_h * 2,c.density_map_h * 2)
        
        if not rev:
            if c.one_hot:
                labels = F.one_hot(labels.to(torch.int64),num_classes=10).to(torch.float)
                if a.args.subnet_type in g.SUBNETS:
                    labels = labels.unsqueeze(2).unsqueeze(3).expand(-1, -1, c.density_map_h,c.density_map_w)
            else:
                labels = labels.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, -1, c.density_map_h,c.density_map_w)
        
        if c.debug: 
            print("expanded labels size")
            print(labels.size(),"\n") 
        
        z = self.nf(x_or_z = labels,c = feats,rev=rev)
        
        return z

def make_model_name(train_loader=None):
     now = datetime.now() 
     
     if train_loader == None:
         bs = 'search'
     else:
         bs = train_loader.batch_size
            
     parts = [a.args.schema,
              a.args.model_name,
              os.uname().nodename,
             "BS"+str(bs),
             "LR_I"+str(a.args.learning_rate),
             "E"+str(a.args.meta_epochs*a.args.sub_epochs),
             "DIM"+str(c.density_map_h),
             "OPTIM"+str(a.args.optim)]
     
     if a.args.model_name not in g.BASELINE_MODEL_NAMES:
         parts.append("FE_"+str(a.args.feat_extractor))    
     
     if a.args.dmap_scaling != 1:
         parts.append('SC{}'.format(a.args.dmap_scaling))
     
     if a.args.model_name == 'NF':
         parts.append('NC'+str(c.n_coupling_blocks))
         parts.append(a.args.subnet_type)
         
         if a.args.subnet_bn:
             parts.append('BN')
             
     if a.args.jac:
         parts.append('JC')
    
     if a.args.all_in_one:
         parts.append('IN1')
         
     if a.args.split_dimensions:
         parts.append('SPLIT')

     parts.append(a.args.sampler)
     
     if a.args.model_name in ['UNet_seg','LCFCN']:
         parts.append('MX_SZ_'+str(a.args.max_filter_size))
        
     parts.append(str(a.args.scheduler))
     
     if a.args.data == 'dlr':
         parts.append('DLRACD')
     
     if a.args.data == 'mnist':
         parts.append('MNIST')
     
     if a.args.model_name == 'NF' and a.args.joint_optim:
         parts.append('JO')
         
     if a.args.model_name == 'NF' and a.args.pretrained:
         parts.append('PT')
         
     if a.args.model_name == 'NF' and a.args.pyramid:
         parts.append('PY_{}'.format(a.args.n_pyramid_blocks))
         
     if c.counts and not a.args.data == 'mnist':
         parts.append('CT')
     
     if a.args.model_name == 'NF' and a.args.fixed1x1conv:
         parts.append('1x1')
         
     if c.scale != 1:
         parts.append('SC_{}'.format(c.scale))
         
     if a.args.model_name == 'NF' and c.dropout_p != 0:
         parts.append('DP_{}'.format(c.dropout_p))
         
     parts.extend(["WD",str(a.args.weight_decay)])
         
     if c.train_feat_extractor or a.args.load_feat_extractor_str != '':
         parts.append('FT')
         
     if a.args.sigma != 4 and not a.args.data == 'mnist':
         parts.extend(["FSG",str(a.args.sigma)])
         
     if a.args.model_name == 'NF' and c.clamp_alpha != 1.9 and not a.args.data == 'mnist':
         parts.extend(["CLA",str(c.clamp_alpha)])
         
     if c.test_train_split != 70 and not a.args.data == 'mnist':
         parts.extend(["SPLIT",str(c.test_train_split)])
         
            
     parts.append(str(now.strftime("%d_%m_%Y_%H_%M_%S")))
     
     modelname = "_".join(parts)
     
     print("Training Model: ",modelname)
     
     if a.args.mode == 'search':
         modelname = 'ray_'+modelname
     return modelname    

def init_model(feat_extractor=None,config=None):
    
    name = make_model_name()
    # a.args.model_name empty somehow?
    
    if 'model_name' in config.keys():
        assert a.args.model_name == config['model_name']
    
    model_name = a.args.model_name #config['model_name']
    
    if model_name == 'NF':
         mdl = CowFlow(modelname=name,feat_extractor=feat_extractor,config=config)
    elif model_name == 'UNet':
         mdl = b.UNet(modelname=name)
    elif model_name == 'UNet_seg':
         mdl = b.UNet(modelname=name,seg=True)
    elif model_name == 'CSRNet': 
         mdl = b.CSRNet(modelname=name)
    elif model_name ==  'MCNN':
         mdl = b.MCNN(modelname=name)
    elif model_name ==  'FCRN':
         mdl = b.FCRN_A(modelname=name)
    elif model_name ==  'VGG':
         mdl = b.VGG_density(modelname=name)
    elif model_name ==  'LCFCN':
         mdl = b.LCFCN(modelname=name)
    elif model_name ==  'Res50':
         mdl = b.Res50(modelname=name)       
        
    return mdl
