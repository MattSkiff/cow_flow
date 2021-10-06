import numpy as np
import torch
from tqdm import tqdm # progress bar
import time 

import config as c

from eval import eval_mnist, eval_model
from utils import get_loss, plot_preds, t2np
from torch.nn.utils import clip_grad_value_
from torch.optim.lr_scheduler import ExponentialLR
from model import CowFlow, MNISTFlow, save_model, save_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# tensorboard
from torch.utils.tensorboard import SummaryWriter

def train_battery(train_loader,valid_loader,lr_i = c.lr_init):
    
        if len(lr_i) >= 1 or type(train_loader) == list and type(valid_loader) == list:
            
            j = 0
            
            for tl, vl in zip(train_loader,valid_loader):
            
                for param in lr_i:
                    
                    j = j+1
                    
                    battery_string = 'battery_'+str(j)
                    bt_id = 'runs/'+c.schema+'/'+battery_string
                    print('beginning: {}\n'.format(bt_id))
                    writer = SummaryWriter(log_dir=bt_id+"_"+c.modelname)
                        
                    train(train_loader=tl,
                          valid_loader=vl,
                          battery = True,
                          lr_i = [param],
                          writer = writer)
            
        else:
            ValueError("lr_i must have more than one value, or the dataloaders must be supplied as lists")
                

def train(train_loader,valid_loader,battery = False,lr_i=c.lr_init,writer=None): #def train(train_loader, test_loader):
            
            if c.verbose:
                print("Training using {} train samples and {} validation samples...".format(len(train_loader)*train_loader.batch_size,
                                                                                            len(valid_loader)*valid_loader.batch_size))
    
            if not battery:
                writer = SummaryWriter(log_dir='runs/'+c.schema+'/'+c.modelname)
            else:
                writer = writer
        
            if c.mnist:
                model = MNISTFlow()    
            else:
                model = CowFlow()
                    
            if c.joint_optim:
                optimizer = torch.optim.Adam([
                            {'params': model.nf.parameters()},
                            {'params': model.feature_extractor.parameters(), 'lr_init': 1e-3,'betas':(0.9,0.999),'eps':1e-08, 'weight_decay':0}
                        ], lr=lr_i[0], betas=(0.8, 0.8), eps=1e-04, weight_decay=c.weight_decay )
            else:
                optimizer = torch.optim.Adam(model.nf.parameters(), lr=lr_i[0], betas=(0.8, 0.8), eps=1e-04, weight_decay=c.weight_decay)
            
            
            # add scheduler to improve stability further into training
            if c.scheduler == "exponential":
                scheduler = ExponentialLR(optimizer, gamma=0.9)
        
            
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
                    
                    if c.debug:
                        print(scheduler.get_lr()[0])
                    
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
                        
                        writer.add_scalar('loss/training_minibatch',loss, k)
                        
                        train_loss.append(loss_t)
                        loss.backward()
                        clip_grad_value_(model.parameters(), c.clip_value)
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
                    
                    if c.scheduler != "none":
                        scheduler.step()
                    
                    writer.add_scalar('loss/training_subpeoch',mean_train_loss, j)
                    
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
                        writer.add_scalar('loss/valid_subpeoch',valid_loss, j)
                

                l += 1
                
                if c.mnist:
                    valid_accuracy, training_accuracy = eval_mnist(model,valid_loader,train_loader)
                else:
                    valid_accuracy, training_accuracy = eval_model(model,valid_loader,train_loader)
                
                if (c.save_model or battery) and c.checkpoints:
                    model.to('cpu')
                    save_model(model,str(l)+"_"+c.modelname)
                    save_weights(model,str(l)+"_"+c.modelname)
                    model.to(c.device)
                
                writer.add_scalar('accuracy/training',training_accuracy, l)
                print(training_accuracy,l)
                writer.add_scalar('accuracy/valid',valid_accuracy, l)
                print(valid_accuracy,l)
                
                # add param tensorboard scalars
                writer.add_hparams(hparam_dict = {'learning rate init.':lr_i[0],
                            'batch size':valid_loader.batch_size,
                            'image height':c.density_map_h,
                            'image width':c.density_map_w,
                            'mnist?':c.mnist,
                            'test run?':c.test_run,
                            'prop. of data':c.data_prop,
                            'clamp alpha':c.clamp_alpha,
                            'weight decay':c.weight_decay,
                            'epochs':c.meta_epochs*c.sub_epochs,
                            'no. of coupling blocks':c.n_coupling_blocks,
                            'feat vec length':c.n_feat},
                           metric_dict = {'accuracy/valid':valid_accuracy,
                                          'accuracy/training':training_accuracy},
                           run_name = c.modelname)
            
            
                writer.flush()
            
            
            
            # post training: visualise a random reconstruction
            if c.dmap_viz:
            
                plot_preds(model, valid_loader, plot = True,mnist = c.mnist)
            
            if c.save_model or battery:
                model.to('cpu')
                save_model(model,c.modelname)
                save_weights(model, c.modelname)
                model.to(c.device)
            
            if not battery:
                return model