import numpy as np
import torch
from tqdm import tqdm # progress bar

import config as c
from model import CowFlow, save_model, save_weights

from utils import preprocess_batch
from utils import gaussianise_annotations

# pyro NF implementation
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.nn import ConditionalDenseNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#def train(train_loader, test_loader):
def train(data_loader):
    model = CowFlow()
    model.to(c.device)
    
    # define NF
    # https://pyro.ai/examples/normalizing_flows_i.html
    # https://docs.pyro.ai/en/dev/_modules/pyro/distributions/transforms/affine_coupling.html
    
    input_dim=c.n_feat
    annotations_dim = 4 # dim annotations = 4
    # split_dim = 6 - defaults to input_dim / 2 ; don't need to specify
    
    # x1 in tutorial - 'univariate distribution'
    
    
    # 2d density of object annotations
    base_dist = dist.Normal(torch.zeros(input_dim, device=c.device).cuda(), torch.ones(input_dim, device=c.device).cuda())
    
    
    features_transform = T.spline(input_dim).cuda()
    dist_features = dist.TransformedDistribution(base_dist, [features_transform])
    
    # old : may need to manually define hypernet as using helper function default
    # param_dims = [input_dim-split_dim, input_dim-split_dim]
    # hypernet = ConditionalDenseNN(split_dim, context_dim, [10*input_dim],param_dims)
    # transform = T.ConditionalAffineCoupling(split_dim, hypernet)
    
    # x2 in tutorial - 'conditional distribution'
    annotations_transform = T.conditional_affine_coupling(input_dim=annotations_dim, context_dim = c.n_feat).cuda()
    dist_annotations_given_features = dist.ConditionalTransformedDistribution(base_dist, [annotations_transform])
    
    modules = torch.nn.ModuleList([features_transform, annotations_transform])
    optimizer = torch.optim.Adam(modules.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)
    
    for epoch in range(c.meta_epochs):
        model.train()
        if c.verbose:
            print(F'\nTrain epoch {epoch}')
            
        for sub_epoch in range(c.sub_epochs):
            
            train_loss = list()
            
            for i, data in enumerate(tqdm(data_loader, disable=c.hide_tqdm_bar)):
                
                images,annotations,classes = preprocess_batch(data)
 
                y = model(images.float()) # retrieve features
                z = annotations
                
                gaussianise_annotations(z)
                1/0
                
                print(y.device)
                print(z.device)
                
                if c.debug:
                    print("number of elements in annotation list:")
                    print(len(annotations))
                    print("number of images in image tensor:")
                    print(len(images))
                    
                
                for step in range(c.steps):
                    
                    optimizer.zero_grad()
                    # nb: x1 = features (y), x2 = annotations
                    
                    print(y.device)
                    print(z.device)
                    
                    ln_p_x1 = dist_features.log_prob(y)
                    # this loss needs to calc distance between predicted density and density map
                    ln_p_x2_given_x1 = dist_annotations_given_features.condition(z.detach()).log_prob(y.detach())
                    loss = -(ln_p_x1 + ln_p_x2_given_x1).mean()
                    
                    loss.backward()
                    optimizer.step()
                    
                    dist_features.clear_cache()
                    dist_annotations_given_features.clear_cache()
                    
                    #z = model(data[0]) # load images into model

            
    #loss = get_loss(z, model.nf.jacobian(run_forward=False))
    
    # train_loss.append(t2np(loss))
    
    # loss.backward()
    # optimizer.step()
    # flow_dist.clear_cache() # new
                       
    return y#z