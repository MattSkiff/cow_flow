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
import numpy as np

import os # save model
import collections
import shutil
import dill # solve error when trying to pickle lambda function in FrEIA

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

def save_model(mdl,filename,loc=g.MODEL_DIR):
    if not os.path.exists(loc):
        os.makedirs(loc)
    torch.save(mdl, os.path.join(loc,filename), pickle_module=dill)
    print("model {} saved to {}".format(filename,loc))
    save_cstate(cdir=g.C_DIR,modelname="",config_file="config.py")
    
def load_model(filename,loc=g.MODEL_DIR):
    path = os.path.join(loc, filename)
    mdl = torch.load(path, pickle_module=dill)
    
    print("model {} loaded from {}".format(filename,loc))
    return mdl
    
def save_weights(mdl,filename,loc=g.WEIGHT_DIR):
    if not os.path.exists(loc):
        os.makedirs(loc)
    torch.save(mdl.state_dict(),os.path.join(loc,filename), pickle_module=dill)
        
def load_weights(mdl, filename,loc=g.WEIGHT_DIR):
    path = os.path.join(loc, filename)
    mdl.load_state_dict(torch.load(path,pickle_module=dill)) 
    return mdl

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
            #del finetuned_fe
            
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

# from https://github.com/VLL-HD/conditional_INNs
def random_orthog(n):
    w = np.random.randn(n, n)
    w = w + w.T
    
    # unitary (w), vectors with singular (S), unitary (w)
    w, S, V = np.linalg.svd(w)
    
    out = torch.FloatTensor(w)
    out.requires_grad_()
    
    return out 

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
                    # use small filters in subnet per glow paper
                    ("conv2", nn.Conv2d(c.filters, n_filters*2, kernel_size = 1,padding = 0)),
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
        nn.Dropout(p=c.dropout_p),
        nn.Linear(internal_size, internal_size), 
        nn.ReLU(),
        nn.Dropout(p=c.dropout_p),
        nn.Linear(internal_size,dims_out)
        )
    
    return net

def nf_pyramid(input_dim=(c.density_map_h,c.density_map_w),condition_dim=c.n_feat):
    assert c.subnet_type == 'conv'
    assert not c.gap and not c.counts and not c.mnist
    
    # TODO - will break because of ref to config file
    mdl = ResNetPyramid()
    # TODO: take out hardcoding activation channels (64)
    channels = 3
    feats = mdl(torch.randn(c.batch_size[0],channels,c.density_map_h,c.density_map_w,requires_grad=False))
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
        
        if c.fixed1x1conv and k != 0: # c.channels*4**k
            nodes.append(Ff.Node(nodes[-1].out0, Fm.Fixed1x1Conv,{'M': random_orthog(c.channels*4**k).to(c.device) }, name='1x1_Conv_{}'.format(k)))
        else:
            nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom, {'seed': k}, name='Permute_{}'.format(k)))
        
        nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarDownsampling, {}, name = 'Main_Downsampling_{}'.format(k)))
        
        for j in range(c.n_pyramid_blocks):
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                            {'clamp': c.clamp_alpha,'subnet_constructor':subnet},
                            conditions=conditions[k],name = 'Couple_{}'.format(k)))
    
    inn = Ff.GraphINN(nodes + conditions + [Ff.OutputNode(nodes[-1], name='output')], verbose=c.verbose)
    
    return inn

def nf_pyramid_split(input_dim=(c.density_map_h,c.density_map_w),condition_dim=c.n_feat):
    assert c.subnet_type == 'conv'
    assert not c.gap and not c.counts and not c.mnist
    
    # TODO - will break because of ref to config file
    mdl = ResNetPyramid()
    # TODO: take out hardcoding activation channels (64)
    channels = 3
    feats = mdl(torch.randn(c.batch_size[0],channels,c.density_map_h,c.density_map_w,requires_grad=False))
    del mdl
    
    if c.verbose:
        print('Feature Pyramid Dimensions:')
        [print(f.shape) for f in feats]
        
    p_dims = [f.shape for f in feats]
    #assert p_dims[0][1] == channels
    
    nodes = [Ff.InputNode(c.channels,input_dim[0],input_dim[1],name='input')] 
    conditions = []; #transformed_splits = []; # splits = [];
    
    for k in range(c.levels):
        conditions.append(Ff.ConditionNode(p_dims[k][1],p_dims[k][2],p_dims[k][3],name = 'Condition{}'.format(k)))
        
        if c.fixed1x1conv and k != 0: # c.channels*4**k
            nodes.append(Ff.Node(nodes[-1].out0, Fm.Fixed1x1Conv,{'M': random_orthog(c.channels*4**k).to(c.device) }, name='1x1_Conv_{}'.format(k)))
        else:
            nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom, {'seed': k}, name='Permute_{}'.format(k)))
        
        nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarDownsampling, {}, name = 'Main_Downsampling_{}'.format(k)))
        
        # split off noise dimensions (see Dinh, 2016) - multiscale architecture (splits along channel, splits into two [default])
        #if split_count < c.n_splits:
        #    split_count += 1
        
        # splitting once
        if k == 0:
            split = Ff.Node(nodes[-1].out0, Fm.Split, {}, name='Split_{}'.format(k))
            nodes.append(split)
            
            # after every split, out1 sent to coupling, out0 split off for downsampling and later concatenation
            for j in range(c.n_pyramid_blocks):
                if j == 0:
                    nodes.append(Ff.Node(nodes[-1].out1, Fm.GLOWCouplingBlock,
                                         {'clamp': c.clamp_alpha,'subnet_constructor':subnet},
                                         conditions=conditions[k],name = 'Couple_{}_{}'.format(k,j)))
                else:
                    # no split
                    nodes.append(Ff.Node(nodes[-1].out0, Fm.GLOWCouplingBlock,
                                         {'clamp': c.clamp_alpha,'subnet_constructor':subnet},
                                         conditions=conditions[k],name = 'Couple_{}_{}'.format(k,j)))
                    
        else:
            for j in range(c.n_pyramid_blocks):
                nodes.append(Ff.Node(nodes[-1].out0, Fm.GLOWCouplingBlock,
                            {'clamp': c.clamp_alpha,'subnet_constructor':subnet},
                            conditions=conditions[k],name = 'Couple_{}'.format(k)))
                
    last_coupling = nodes[-1]  # link up main chain to split chains
    
        # else:
        #     for j in range(c.n_pyramid_blocks):
        #         # no split
        #         nodes.append(Ff.Node(nodes[-1].out0, Fm.GLOWCouplingBlock,
        #                                      {'clamp': c.clamp_alpha,'subnet_constructor':subnet},
        #                                      conditions=conditions[k],name = 'Couple_{}_{}'.format(k,j)))    
       
    #if c.debug:
    
    # main chain of split
    nodes.append(Ff.Node(split.out0, Fm.HaarDownsampling, {}, name = 'Split_Downsampling1'))
    nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarDownsampling, {}, name = 'Split_Downsampling2'))
    nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarDownsampling, {}, name = 'Split_Downsampling3'))
    
    last_split_ds = Ff.Node(nodes[-1].out0, Fm.HaarDownsampling, {}, name = 'Split_Downsampling4')
    
    nodes.append(last_split_ds)
        
    # loop to add downsampling to split dimensions before concatenation
    # for each split node, append a variable amount of downsampling blocks to ensure dimensional compatability
    # for i in range(len(splits)):
    #     for j in range(5-i):
    #         if i == len(splits)-1:
    #             transformed_splits.append(splits[i])  
    #         elif j == 0:
    #             #print(splits[i])
    #             # out1 of split nodes proceeds into main node chain
    #             nodes.append(Ff.Node(splits[i].out0, Fm.HaarDownsampling, {}, name = 'Split_{}_Downsampling_{}'.format(i,j)))
    #         elif j == (5-i)-1:
    #             transformed_splits.append(nodes[-1])
    #         else:
    #             nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarDownsampling, {}, name = 'Split_{}_Downsampling_{}'.format(i,j)))
    
    # concatenate list of split chain outputs back to output node           
    concat_node = Ff.Node([last_coupling.out0,
                           last_split_ds.out0]#,
                           #transformed_splits[1].out0,
                           #transformed_splits[2].out0,
                           #transformed_splits[3].out0, 
                           #transformed_splits[4].out0]
                           ,Fm.Concat,{},name='Concat_Splits')
    
    nodes.append(concat_node)
    
    for node in nodes:
        print('node')
        print(node)
        #print(node.output_dims)
        #print(node.input_dims)
        
        print('outs')
        print(node.outputs)
        print('ins')
        print(node.inputs)
        print('---')   
    
    inn = Ff.GraphINN(nodes + conditions + [Ff.OutputNode(nodes[-1], name='output')], verbose=c.verbose)
    
    return inn
    
def nf_head(input_dim=(c.density_map_h,c.density_map_w),condition_dim=c.n_feat,mnist=False):
    
    # include batch size as extra dimension here? data is batched along extra dimension
    # input = density maps / mnist labels 
    
    # 'Because of the construction of the conditional coupling blocks, the condition must have the same spatial dimensions as the data'
    # https://github.com/VLL-HD/FrEIA/issues/9
    
    # condition = exacted image features
    if (c.mnist or (c.counts and not c.gap)) and c.subnet_type == 'conv':
        condition = [Ff.ConditionNode(condition_dim,input_dim[0] // 2,input_dim[1] // 2, name = 'condition')]
    elif (c.counts and c.gap) or (c.subnet_type == 'fc' and c.mnist):
        condition = [Ff.ConditionNode(condition_dim,name = 'condition')]
    else:
        # TODO: avoid hardcoding feature spatial dimensions in
        ft_dims = ft_dims_select()
        
        condition = [Ff.ConditionNode(condition_dim,ft_dims[0],ft_dims[1],name = 'condition')]
    
    if c.counts and c.gap:
        nodes = [Ff.InputNode(1,name='input')] # single count value
    #elif c.subnet_type == 'conv':
    else:
        nodes = [Ff.InputNode(c.channels,input_dim[0],input_dim[1],name='input')] 
#    else:
#        nodes = [Ff.InputNode(c.channels,name='input')] 
    
    # haar downsampling to resolves input data only having a single channel (from unsqueezed singleton dimension)
    # affine coupling performs channel wise split
    # https://github.com/VLL-HD/FrEIA/issues/8
    if (c.mnist or (c.counts and not c.gap) or c.feat_extractor == 'none' or not c.downsampling) and c.subnet_type == 'conv':
        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {}, name = 'Downsampling'))
        
    elif not c.counts and c.feat_extractor != 'none' and c.downsampling:
            
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
        
        if not (c.counts and c.subnet_type == 'fc') and c.fixed1x1conv and k % c.freq_1x1 == 0:
            nodes.append(Ff.Node(nodes[-1], Fm.Fixed1x1Conv,{'M': random_orthog(c.channels*multiplier).to(c.device) }, name='1x1_conv_{}'.format(k)))
        else:
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
            if a.args.split_dimensions:
                self.nf = nf_pyramid_split() 
            else:
                self.nf = nf_pyramid()   
        else:
            self.nf = nf_head()  
        
        # these attr's are needed to make the model object independant of the config file
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
        self.fixed1x1conv = c.fixed1x1conv
        self.scale = c.scale
        self.density_map_h = c.density_map_h
        self.density_map_w = c.density_map_w
        self.downsampling = c.downsampling
        self.scale = c.scale
        self.noise = c.noise
        self.seed = c.seed

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
            print("concatenated and pooled:{} feature size..".format(c.gap))
            print(feats.size(),"\n")
        
        # adding spatial dimensions....
        # remove dimension of size one for concatenation in NF coupling layer squeeze()
        if c.feat_extractor != "none" and c.subnet_type =='conv':
        
            if self.gap:
                feats = feats.unsqueeze(2).unsqueeze(3).expand(-1, -1, (c.density_map_h) // 2,(c.density_map_w) // 2)
        
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
            # feats = list of 5 tensors
            z = self.nf(x_or_z = labels,c = feats,rev=rev) # [-c.levels:]
        elif self.unconditional:
            z = self.nf(x_or_z = labels,rev=rev)
        else:
            z = self.nf(x_or_z = labels,c = feats,rev=rev)
          
        return z

class MNISTFlow(nn.Module):
    
    def __init__(self,modelname,feat_extractor):
        super(MNISTFlow,self).__init__()
        
        if c.feat_extractor == 'resnet18':
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
        self.fixed1x1conv = c.fixed1x1conv
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
            if c.feat_extractor != "none" and c.subnet_type == 'conv':
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
                labels = F.one_hot(labels.to(torch.int64),num_classes=10).to(torch.float)
                if c.subnet_type == 'conv':
                    labels = labels.unsqueeze(2).unsqueeze(3).expand(-1, -1, c.density_map_h,c.density_map_w)
            else:
                labels = labels.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, -1, c.density_map_h,c.density_map_w)
        
        if c.debug: 
            print("expanded labels size")
            print(labels.size(),"\n") 
        
        z = self.nf(x_or_z = labels,c = feats,rev=rev)
        
        return z
    
class unet():
    # from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
    pass

class csrnet():
    pass

class lcffcn():
    pass

class fcrn():
    pass



