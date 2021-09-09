import numpy as np
import torch
from tqdm import tqdm # progress bar
import time 

import config as c
from utils import get_loss
from utils import t2np
from model import CowFlow #, save_model, save_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(data_loader): #def train(train_loader, test_loader):
    
    model = CowFlow()
    optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)
    model.to(c.device)
    
    for epoch in range(c.meta_epochs):
        
        model.train()
        
        if c.verbose:
            print(F'\nTrain epoch {epoch}',"\n")
            
        for sub_epoch in range(c.sub_epochs):
            
            if c.verbose:
                t_e1 = time.perf_counter()
            
            train_loss = list()
            
            for i, data in enumerate(tqdm(data_loader, disable=c.hide_tqdm_bar)):
                
                t1 = time.perf_counter()
                
                optimizer.zero_grad()
                
                # x=images->features,y=dmaps
                x,y,classes = data
                x = x.float().to(c.device)
                y = y.float().to(c.device)
                
                if c.debug:
                    print("density maps from data batch size, device..")
                    print(y.size())
                    print(y.device,"\n")
                    
                    print("images from data batch size, device..")
                    print(x.size())
                    print(x.device,"\n")
                
                    # z is the probability density under the latent normal distribution
                    print("output of model - two elements: y (z), jacobian")
                z, log_det_jac = model(x,y) # inputs features,dmaps
                
                # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True) [FrEIA]
                # i.e. what y (output) is depends on direction of flow
                if c.debug:
                    print(z.size()) # log likelihoods
                    print("z shape")
                    print(z.shape[1]) # 256
                
                # this loss needs to calc distance between predicted density and density map
                loss = get_loss(z, log_det_jac) # model.nf.jacobian(run_forward=False)
                
                print("loss: {}".format(loss))
                train_loss.append(t2np(loss))
                loss.backward()
                optimizer.step()
                   
                # nb: x1 = features (y), x2 = annotations
                
                mean_train_loss = np.mean(train_loss)
                
                t2 = time.perf_counter()
                est_total_train = (t2-t1) * len(data_loader) * c.sub_epochs * c.meta_epochs
                
                if c.verbose:
                    print('Batch Time: {:f},Batch: {:d},Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(t2-t1,i, epoch, sub_epoch, mean_train_loss))
                    print('{:d} Batches in sub-epoch remaining'.format(len(data_loader)-i))
                    print('Estimated Total Training Time: {:f}'.format(est_total_train))
                
                if c.debug:
                    print("number of elements in density maps list:")
                    print(len(y)) # images
                    print("number of images in image tensor:")
                    print(len(x)) # features
            
            if c.verbose:
                t_e2 = time.perf_counter()
                print("Sub Epoch Time: {:f}".format(t_e2-t_e1))

                       
    return model