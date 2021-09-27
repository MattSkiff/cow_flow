import numpy as np
import torch
from tqdm import tqdm # progress bar
import time 

import config as c

from eval import eval_mnist, eval_model
from utils import get_loss, reconstruct_density_map, t2np
from model import CowFlow, MNISTFlow, save_model, save_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment=c.modelname)

def train(train_loader,valid_loader): #def train(train_loader, test_loader):
    
    if c.mnist:
        model = MNISTFlow()    
    else:
        model = CowFlow()
        
    
    if c.joint_optim:
        optimizer = torch.optim.Adam([
                    {'params': model.nf.parameters()},
                    {'params': model.feature_extractor.parameters(), 'lr_init': 1e-3,'betas':(0.9,0.999),'eps':1e-08, 'weight_decay':0}
                ], lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=c.weight_decay )
    else:
        optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=c.weight_decay)
    
    model.to(c.device)
    
    k = 0 # track total mini batches
    j = 0 # track total sub epochs
    l = 0 # track epochs
    
    for meta_epoch in range(c.meta_epochs):
        
        model.train()
        
        # add param tensorboard scalars
        writer.add_hparams(hparam_dict = {'learning rate initialisation':c.lr_init,
                    'batch size':c.batch_size,
                    'image height':c.density_map_h,
                    'image width':c.density_map_w,
                    'mnist?':c.mnist,
                    'test run?':c.test_run,
                    'proportion of data':c.data_prop,
                    'clamp alpha':c.clamp_alpha,
                    'weight decay':c.weight_decay,
                    'epochs':c.meta_epochs*c.sub_epochs,
                    'number of coupling blocks':c.n_coupling_blocks,
                    'feature vector length':c.n_feat},
                   metric_dict = {'placeholder':0},
                   run_name = c.modelname)
        
        if c.verbose:
            print(F'\nTrain meta epoch {meta_epoch}',"\n")
            
        for sub_epoch in range(c.sub_epochs):
            
            if c.verbose:
                t_e1 = time.perf_counter()
            
            train_loss = list()
            
            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                
                t1 = time.perf_counter()
                
                optimizer.zero_grad()
                
                if not c.mnist:
                    images,dmaps,labels = data
                else:
                    images,labels = data
                
                if c.debug:
                    print("labels:")
                    print(labels)
                    
                images = images.float().to(c.device)
                
                if not c.mnist:
                    dmaps = dmaps.float().to(c.device)
                else: 
                    labels = labels.float().to(c.device)
                
                if c.debug and not c.mnist:
                    print("density maps from data batch size, device..")
                    print(dmaps.size())
                    print(dmaps.device,"\n")
                    
                    print("images from data batch size, device..")
                    print(images.size())
                    print(images.device,"\n")
                
                    # z is the probability density under the latent normal distribution
                    print("output of model - two elements: y (z), jacobian")
                  
                if not c.mnist:
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
                
                if c.debug and not c.mnist:
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
                        if not c.mnist:
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
                        
                        
                    if i % c.report_freq == 0 and c.verbose and not c.mnist:
                            print('count: {:f}'.format(dmaps.sum()))
    
                    loss = get_loss(z, log_det_jac)
                    valid_loss.append(t2np(loss))
                     
                valid_loss = np.mean(np.array(valid_loss))
                 
                if c.verbose:
                        print('Sub Epoch: {:d} \t valid_loss: {:4f}'.format(sub_epoch,valid_loss))
                
                j += 1
                writer.add_scalar('valid_subpeoch_loss',valid_loss, j)
        
        l += 1
        
        if c.mnist:
            valid_accuracy, training_accuracy = eval_mnist(model,valid_loader,train_loader)
        else:
            valid_accuracy, training_accuracy = eval_model(model,valid_loader,train_loader)
        
        writer.add_scalar('training_accuracy',training_accuracy, l)
        print(training_accuracy,l)
        writer.add_scalar('valid_accuracy',valid_accuracy, l)
        print(valid_accuracy,l)
    
    
    writer.flush()
    
    
    
    # post training: visualise a random reconstruction
    if c.dmap_viz:
    
        reconstruct_density_map(model, valid_loader, plot = True)
    
    # broken # todo
    if c.save_model:
        model.to('cpu')
        save_model(model,c.modelname)
        save_weights(model, c.modelname)
    
    return model