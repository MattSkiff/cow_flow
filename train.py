import numpy as np
import torch
from tqdm import tqdm # progress bar
import time 

import config as c
from utils import get_loss
from utils import t2np
from model import CowFlow #, save_model, save_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def train(train_loader,valid_loader): #def train(train_loader, test_loader):
    
    model = CowFlow()
    optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)
    model.to(c.device)
    
    k = 0 # track total mini batches
    j = 0 # track total sub epochs
    
    for epoch in range(c.meta_epochs):
        
        model.train()
        
        if c.verbose:
            print(F'\nTrain epoch {epoch}',"\n")
            
        for sub_epoch in range(c.sub_epochs):
            
            if c.verbose:
                t_e1 = time.perf_counter()
            
            train_loss = list()
            
            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                
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
                    print("z shape + contents")
                    print(z.size()) # log likelihoods
                    print(z.shape) # equiv to batch size
                
                if c.debug:
                    print('\nlog det jacobian size + contents')
                    print(log_det_jac.size()) # length = batch size
                    print(log_det_jac)
                
                # this loss needs to calc distance between predicted density and density map
                loss = get_loss(z, log_det_jac) # model.nf.jacobian(run_forward=False)
                k += 1
                
                print("loss: {}".format(loss))
                
                loss_t = t2np(loss)
                writer.add_scalar('training_minibatch_loss',loss, k)
                
                train_loss.append(loss_t)
                loss.backward()
                optimizer.step()
                   
                # nb: x1 = features (y), x2 = annotations
                
                mean_train_loss = np.mean(train_loss)
                
                t2 = time.perf_counter()
                est_total_train = ((t2-t1) * len(data_loader) * c.sub_epochs * c.meta_epochs) // 60
                
                if c.verbose:
                    print('Batch Time: {:f},Mini-Batch: {:d}, Epoch: {:d}, Sub Epoch: {:d}, \t train loss: {:.4f}'.format(t2-t1,i, epoch, sub_epoch, mean_train_loss))
                    print('{:d} Batches in sub-epoch remaining'.format(len(data_loader)-i))
                    print('Estimated Total Training Time (minutes): {:f}'.format(est_total_train))
                
                if c.debug:
                    print("number of elements in density maps list:")
                    print(len(y)) # images
                    print("number of images in image tensor:")
                    print(len(x)) # features
            
            writer.add_scalar('training_subpeoch_loss',loss, j)
            
            if c.verbose:
                t_e2 = time.perf_counter()
                print("Sub Epoch Time: {:f}".format(t_e2-t_e1))
            
            valid_loss = list()
            valid_z = list()
            valid_labels = list()
            
            with torch.no_grad():
                for i, data in enumerate(tqdm(valid_loader, disable=c.hide_tqdm_bar)):
                
                # validation
                x,y,classes = data
                x = x.float().to(c.device)
                y = y.float().to(c.device)
                z, log_det_jac = model(x,y)
                
                valid_z.append(z)
                valid_labels.append()
                loss = get_loss(z, log_det_jac)
                valid_loss.append(t2np(loss))
                
            valid_loss = np.mean(np.array(valid_loss))
                
            
            writer.add_scalar('valid_subpeoch_loss',valid_loss, j)

    writer.flush()
    
    if c.save_model:
        model.to('cpu')
        save_model(model,c.modelname)
        save_weights(model, c.modelname)
    
    return model