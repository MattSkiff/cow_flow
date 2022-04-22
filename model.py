from __future__ import print_function, division
import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models import alexnet, resnet18, vgg16_bn, efficientnet_b3   # feature extractors
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn.functional as F
import numpy as np
import collections
import utils as u #init_weights, ft_dims_select
import train # train_feat_extractor
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import config as c # hyper params
import arguments as a
import gvars as g

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
       
class ResNetPyramid(ResNet):

    def __init__(self):
        super(ResNetPyramid, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=1)
        
        if c.load_feat_extractor_str == '':
            #self.load_state_dict(resnet18(pretrained=c.pretrained).state_dict())
            pass
        elif c.load_feat_extractor_str != '':
            num_ftrs = self.fc.in_features
            self.fc = nn.Linear(num_ftrs, 2)  
            finetuned_fe = u.load_model(c.load_feat_extractor_str,loc=g.FEAT_MOD_DIR)
            self.load_state_dict(finetuned_fe.state_dict())
            
    def _forward_impl_my(self, x):
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

def random_orthog(n):
    w = np.random.randn(n, n)
    w = w + w.T
    w, S, V = np.linalg.svd(w)
    
    out = torch.FloatTensor(w)
    out.requires_grad_()
    
    return out 

def select_feat_extractor(feat_extractor,train_loader=None,valid_loader=None):
    
    if c.load_feat_extractor_str:
        feat_extractor = u.load_model(filename=c.load_feat_extractor_str,loc=g.FEAT_MOD_DIR)
    
    feat_extractor = ResNetPyramid()
    
    if c.train_feat_extractor:
        if c.feat_extractor == "resnet18":
            num_ftrs = feat_extractor.fc.in_features
            feat_extractor.fc = nn.Linear(num_ftrs, 2)  
        elif c.feat_extractor == "vgg16_bn":
            num_ftrs = feat_extractor.classifier._modules['6'].in_features
            feat_extractor.classifier._modules['6'] = nn.Linear(num_ftrs, 2)
        
        feat_extractor = train.train_feat_extractor(feat_extractor,train_loader,valid_loader)
        
    return feat_extractor

def sub_conv2d(dims_in,dims_out,n_filters):
    network_dict = collections.OrderedDict(
                [
                    ("conv1", nn.Conv2d(dims_in, n_filters, kernel_size = 3,padding = 1)), 
                    ('batchnorm1',nn.BatchNorm2d(n_filters)),
                    ("relu1", nn.ReLU()),
                    ("conv2", nn.Conv2d(a.args.filters, n_filters*2, kernel_size = 1,padding = 0)),
                    ('batchnorm2',nn.BatchNorm2d(n_filters*2)),
                    ("relu2", nn.ReLU()),
                    ("conv3", nn.Conv2d(n_filters*2, dims_out,kernel_size = 3,padding = 1))
                ]
        )
    
    if not c.batchnorm:
        del network_dict['batchnorm1']
        del network_dict['batchnorm2']
    
    net = nn.Sequential(network_dict)
    net.apply(u.init_weights)
    net.conv3.weight = torch.nn.init.zeros_(net.conv3.weight)
    net.conv3.bias.data.fill_(0.00) 
    
    return net

def subnet(dims_in, dims_out):
    
    if c.subnet_type == 'conv':
        net = sub_conv2d(dims_in,dims_out,a.args.filters)
    elif c.subnet_type == 'fc':
        net = sub_fc(dims_in,dims_out,c.width)
    
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
    
    mdl = ResNetPyramid()
    channels = 3
    feats = mdl(torch.randn(a.args.batch_size,channels,c.density_map_h,c.density_map_w,requires_grad=False))
    del mdl
        
    p_dims = [f.shape for f in feats]
    
    nodes = [Ff.InputNode(c.channels,input_dim[0],input_dim[1],name='input')] 
    conditions = []; 
    
    for k in range(c.levels):
        conditions.append(Ff.ConditionNode(p_dims[k][1],p_dims[k][2],p_dims[k][3],name = 'Condition{}'.format(k)))
        
        if c.fixed1x1conv and k != 0: 
            nodes.append(Ff.Node(nodes[-1].out0, Fm.Fixed1x1Conv,{'M': random_orthog(c.channels*4**k).to(c.device) }, name='1x1_Conv_{}'.format(k)))
        else:
            nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom, {'seed': k}, name='Permute_{}'.format(k)))
        
        nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarDownsampling, {}, name = 'Main_Downsampling_{}'.format(k)))
        
        for j in range(a.args.n_pyramid_blocks):
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                            {'clamp': c.clamp_alpha,'subnet_constructor':subnet},
                            conditions=conditions[k],name = 'Couple_{}'.format(k)))
    
    inn = Ff.GraphINN(nodes + conditions + [Ff.OutputNode(nodes[-1], name='output')])
    
    return inn
        
class CowFlow(nn.Module):
    
    def __init__(self,modelname,feat_extractor):
        super(CowFlow,self).__init__()
        
        self.feat_extractor = feat_extractor
        self.nf = nf_pyramid()    
        self.classification_head = resnet18(pretrained=c.pretrained,progress=False)
        self.dlr_acd = a.args.data == 'dlr_acd'
        self.modelname = modelname
        self.subnet_type = c.subnet_type
        self.n_coupling_blocks = c.n_coupling_blocks
        self.pretrained = c.pretrained
        self.finetuned = c.train_feat_extractor
        self.scheduler = a.args.scheduler
        self.fixed1x1conv = c.fixed1x1conv
        self.density_map_h = c.density_map_h
        self.density_map_w = c.density_map_w
        self.noise = a.args.noise
        self.seed = c.seed
        self.optim = a.args.optim
        self.dmap_scaling = a.args.dmap_scaling

    def forward(self,images,labels,rev=False): 
        
        feat_s = self.feat_extractor(images)
        feats = feat_s

        if not rev:
            labels = labels.unsqueeze(1) 

        z = self.nf(x_or_z = labels,c = feats,rev=rev) 
          
        return z