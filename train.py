import numpy as np
import torch
from tqdm import tqdm # progress bar
import time 

import config as c

from utils import get_loss, reconstruct_density_map, t2np
from model import CowFlow, MNISTFlow, save_model, save_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def train(train_loader,valid_loader): #def train(train_loader, test_loader):
    
    if c.toy:
        model = MNISTFlow()    
    else:
        model = CowFlow()
        
    optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)
    model.to(c.device)
    
    k = 0 # track total mini batches
    j = 0 # track total sub epochs
    l = 0 # track epochs
    
    for meta_epoch in range(c.meta_epochs):
        
        model.train()
        
        if c.verbose:
            print(F'\nTrain meta epoch {meta_epoch}',"\n")
            
        for sub_epoch in range(c.sub_epochs):
            
            if c.verbose:
                t_e1 = time.perf_counter()
            
            train_loss = list()
            
            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                
                t1 = time.perf_counter()
                
                optimizer.zero_grad()
                
                
                # x=images->y=dmaps
                if not c.toy:
                    images,dmaps,labels = data
                else:
                    images,labels = data
                
                if c.debug:
                    print("labels:")
                    print(labels)
                    
                images = images.float().to(c.device)
                
                if not c.toy:
                    dmaps = dmaps.float().to(c.device)
                else: 
                    labels = torch.Tensor([labels])
                    labels = labels.to(device)
                
                if c.debug and not c.toy:
                    print("density maps from data batch size, device..")
                    print(dmaps.size())
                    print(dmaps.device,"\n")
                    
                    print("images from data batch size, device..")
                    print(images.size())
                    print(images.device,"\n")
                
                    # z is the probability density under the latent normal distribution
                    print("output of model - two elements: y (z), jacobian")
                  
                if not c.toy:
                    z, log_det_jac = model(images,dmaps) # inputs features,dmaps
                else:
                    z, log_det_jac = model(images,labels)
                    
                
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
                
                loss_t = t2np(loss)
                writer.add_scalar('training_minibatch_loss',loss, k)
                
                train_loss.append(loss_t)
                loss.backward()
                optimizer.step()
                
                mean_train_loss = np.mean(train_loss)
                
                t2 = time.perf_counter()
                est_total_train = ((t2-t1) * len(train_loader) * c.sub_epochs * c.meta_epochs) // 60
                
                if i % c.report_freq == 0 and c.verbose:
                        print('Mini-Batch Time: {:f}, Mini-Batch: {:d}, Mini-Batch train loss: {:.4f}'.format(t2-t1,i, mean_train_loss))
                        print('Meta Epoch: {:d}, Sub Epoch: {:d}'.format(meta_epoch, sub_epoch))
                        print('{:d} Mini-Batches in sub-epoch remaining'.format(len(train_loader)-i))
                        print('Est Total Training Time (minutes): {:f}'.format(est_total_train))
                
                if c.debug and not c.toy:
                    print("number of elements in density maps list:")
                    print(len(dmaps)) # images
                    print("number of images in image tensor:")
                    print(len(images)) # features
            
            writer.add_scalar('training_subpeoch_loss',mean_train_loss, j)
            
            if c.verbose:
                t_e2 = time.perf_counter()
                print("Sub Epoch Time (s): {:f}".format(t_e2-t_e1))
            
            if c.validation: 
                valid_loss = list()
                valid_z = list()
                valid_counts = list()
                # valid_coords = list() # todo
                
                with torch.no_grad():
                    
                    for i, data in enumerate(tqdm(valid_loader, disable=c.hide_tqdm_bar)):
                        
                        # validation
                        if not c.toy:
                             images,dmaps,labels = data
                             dmaps = dmaps.float().to(c.device)
                             images = images.float().to(c.device)
                             z, log_det_jac = model(images,dmaps)
                             
                             valid_counts.append(dmaps.sum())
                        else:
                            images,labels = data
                            labels = torch.Tensor([labels]).to(c.device)
                            images = images.float().to(c.device)
                            z, log_det_jac = model(images,labels)
                            
                        valid_z.append(z)
                        
                        
                    if i % c.report_freq == 0 and c.verbose and not c.toy:
                            print('count: {:f}'.format(dmaps.sum()))
    
                    loss = get_loss(z, log_det_jac)
                    valid_loss.append(t2np(loss))
                     
                valid_loss = np.mean(np.array(valid_loss))
                 
                if c.verbose:
                        print('Sub Epoch: {:d} \t valid_loss: {:4f}'.format(sub_epoch,valid_loss))
                
                j += 1
                writer.add_scalar('valid_subpeoch_loss',valid_loss, j)
            

    writer.flush()
    
    l += 1
    
    # post training: visualise a random reconstruction
    if c.dmap_viz:
    
        reconstruct_density_map(model, valid_loader, plot = True)
    
    # broken # todo
    if c.save_model:
        model.to('cpu')
        save_model(model,c.modelname)
        save_weights(model, c.modelname)
    
    return model