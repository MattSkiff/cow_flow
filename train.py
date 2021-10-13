import numpy as np
import torch
from tqdm import tqdm # progress bar
import time 
from datetime import datetime 

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
        
        print("Starting battery: ",str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        t1 = time.perf_counter()
        
    
        if len(lr_i) >= 1 or type(train_loader) == list and type(valid_loader) == list:
            
            j = 0
            
            for tl, vl in zip(train_loader,valid_loader):
            
                for param in lr_i:
                    
                    j = j+1
                    
                    if  c.tb:
                        battery_string = 'battery_'+str(j)
                        bt_id = 'runs/'+c.schema+'/'+battery_string
                        print('beginning: {}\n'.format(bt_id))
                        writer = SummaryWriter(log_dir=bt_id+"_"+c.modelname)
                    else:
                        writer = None
                        
                    train(train_loader=tl,
                          valid_loader=vl,
                          battery = True,
                          lr_i = [param],
                          writer = writer)
            
        else:
            ValueError("lr_i must have more than one value, or the dataloaders must be supplied as lists")
            
        t2 = time.perf_counter()
        print("Battery finished. Time Elapsed (hours): ",round((t2-t1) / 60*60 ,2),"| Datetime:",str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
                

def train(train_loader,valid_loader,battery = False,lr_i=c.lr_init,writer=None): #def train(train_loader, test_loader):
            
            run_start = time.perf_counter()
            print("Starting run: ",str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    
            if c.verbose:
                print("Training using {} train samples and {} validation samples...".format(len(train_loader)*train_loader.batch_size,
                                                                                            len(valid_loader)*valid_loader.batch_size))
    
            if not battery and c.tb:
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
                
                for sub_epoch in range(c.sub_epochs):
                    
                    if c.verbose:
                        t_e1 = time.perf_counter()
                    
                    train_loss = list()
                    
                    if c.debug:
                        print(scheduler.get_lr()[0])
                    
                    for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                        
                        t1 = time.perf_counter()
                        
                        optimizer.zero_grad()
                        
                        if not c.mnist and not c.counts:
                            images,dmaps,labels = data
                        elif not c.mnist:
                            images,dmaps,labels,counts = data
                        else:
                            images,labels = data
                        
                        if c.debug:
                            print("labels:")
                            print(labels)
                            
                        images = images.float().to(c.device)
                        
                        if not c.mnist and not c.counts:
                            dmaps = dmaps.float().to(c.device)
                        elif not c.mnist:
                            counts = counts.float().to(c.device)
                        else: 
                            labels = labels.float().to(c.device)
                        
                        if c.debug and not c.mnist and not c.counts:
                            print("density maps from data batch size, device..")
                            print(dmaps.size())
                            print(dmaps.device,"\n")
                            
                        elif c.debug and not c.mnist:
                            print("counts from data batch size, device..")
                            print(counts.size())
                            print(counts.device,"\n")
                            
                        if c.debug:    
                            print("images from data batch size, device..")
                            print(images.size())
                            print(images.device,"\n")
                        
                        # z is the probability density under the latent normal distribution
                        # output of model - two elements: y (z), jacobian
                          
                        if not c.mnist and not c.counts:
                            input_data = (images,dmaps) # inputs features,dmaps
                        elif not c.mnist:
                            input_data = (images,counts) 
                        else:
                            input_data = (images,labels)
                            
                        z, log_det_jac = model(*input_data)
                            
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
                        
                        if writer != None:
                            writer.add_scalar('loss/training_minibatch',loss, k)
                        
                        train_loss.append(loss_t)
                        loss.backward()
                        clip_grad_value_(model.parameters(), c.clip_value)
                        optimizer.step()
                        
                        if c.scheduler != "none":
                            scheduler.step()
                        
                        mean_train_loss = np.mean(train_loss)
                        
                        t2 = time.perf_counter()
                        
                        # todo: account for validation iterations
                        total_iter = len(train_loader) * c.sub_epochs * c.meta_epochs + (len(valid_loader) * c.meta_epochs)
                        #passed_iter = len(train_loader) * j + (len(valid_loader) * j / c.sub_epochs) + i
                        est_total_time = round(((t2-t1) * (total_iter-k)) / 60,2)
                        est_remain_time = round(((t2-t1) * (total_iter-k)) / 60,2)
                        
                        if k % c.report_freq == 0 and c.verbose and k != 0:
                                print('Mini-Batch Time: {:f}, Mini-Batch: {:d}, Mini-Batch train loss: {:.4f}'.format(t2-t1,i+1, mean_train_loss))
                                print('Meta Epoch: {:d}, Sub Epoch: {:d}, | Epoch {:d} out of {:d} Total Epochs'.format(meta_epoch, sub_epoch,meta_epoch*c.sub_epochs + sub_epoch,c.meta_epochs*c.sub_epochs))
                                print('{:d} Mini-Batches in sub-epoch remaining'.format(len(train_loader)-i))
                                print('Total Iterations: ',total_iter,'| Passed Iterations: ',k)
                                print('Total Training Time (mins): {:f3} | Remaining Time (mins): {:f3}'.format(est_total_time,est_remain_time))
                        
                        if c.debug and not c.mnist:
                            print("number of elements in density maps list:")
                            print(len(dmaps)) # images
                            print("number of images in image tensor:")
                            print(len(images)) # features                
                    
                    if writer != None:
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
                                if not c.mnist and not c.counts:
                                     images,dmaps,labels = data
                                     dmaps = dmaps.float().to(c.device)
                                     images = images.float().to(c.device)
                                     z, log_det_jac = model(images,dmaps)
                                     
                                     valid_counts.append(dmaps.sum())
                                elif not c.mnist: 
                                     images,dmaps,labels,counts = data
                                     counts = counts.float().to(c.device)
                                     images = images.float().to(c.device)
                                     z, log_det_jac = model(images,counts)
                                     
                                     valid_counts.append(counts.mean())
                                else:
                                    images,labels = data
                                    labels = torch.Tensor([labels]).to(c.device)
                                    images = images.float().to(c.device)
                                    z, log_det_jac = model(images,labels)
                                    
                                valid_z.append(z)
                                
                                
                            if i % c.report_freq == 0 and c.verbose and not c.mnist:
                                    print('count: {:f}'.format(dmaps.sum()))
            
                            loss = get_loss(z, log_det_jac)
                            k += 1
                            valid_loss.append(t2np(loss))
                             
                        valid_loss = np.mean(np.array(valid_loss))
                         
                        if c.verbose:
                                print('Sub Epoch: {:d} \t valid_loss: {:4f}'.format(sub_epoch,valid_loss))
                        
                        if writer != None:
                            writer.add_scalar('loss/valid_subpeoch',valid_loss, j)
                            
                    j += 1
                l += 1
                
                if c.mnist:
                    valid_accuracy, training_accuracy = eval_mnist(model,valid_loader,train_loader)
                else:
                    valid_accuracy, training_accuracy = eval_model(model,valid_loader,train_loader) # does nothing for now
                
                if (c.save_model or battery) and c.checkpoints:
                    model.to('cpu')
                    save_model(model,str(l)+"_"+c.modelname)
                    save_weights(model,str(l)+"_"+c.modelname)
                    model.to(c.device)
                
                print("\n")
                print("Training Accuracy: ", training_accuracy,"| Epoch: ",l)
                print("Valid Accuracy: ",valid_accuracy,"| Epoch: ",l)
                
                if writer != None:
                    writer.add_scalar('accuracy/training',training_accuracy, l)
                    writer.add_scalar('accuracy/valid',valid_accuracy, l)
                
                    # add param tensorboard scalars
                    # TODO: write this to text file and store once per run (instead of cstate copy)
                    writer.add_hparams(
                                hparam_dict = {
                                'learning rate init.':lr_i[0],
                                'batch size':valid_loader.batch_size,
                                'image height':c.density_map_h,
                                'image width':c.density_map_w,
                                'joint optimisation?':c.joint_optim,
                                'pretrained':c.pre_trained,
                                'mnist?':c.mnist,
                                'counts?':c.counts,
                                'test run?':c.test_run,
                                'prop. of data':c.data_prop,
                                'clamp alpha':c.clamp_alpha,
                                'weight decay':c.weight_decay,
                                'epochs':c.meta_epochs*c.sub_epochs,
                                'no. of coupling blocks':c.n_coupling_blocks,
                                'filter size':c.filter_size,
                                'filter sigma':c.sigma,
                                'feat vec length':c.n_feat},
                               metric_dict = {
                                'accuracy/valid':valid_accuracy,
                                'accuracy/training':training_accuracy
                                              },
                               run_name = c.modelname
                               )
                
                
                    writer.flush()
            
            
            
            # post training: visualise a random reconstruction
            if c.dmap_viz:
            
                plot_preds(model, valid_loader, plot = True,mnist = c.mnist)
            
            # save final model, unless models are being saved at end of every meta peoch
            if c.save_model and not c.checkpoints:
                model.to('cpu')
                save_model(model,c.modelname)
                save_weights(model, c.modelname)
                model.to(c.device)
            
            run_end = time.perf_counter()
            print("Run finished. Time Elapsed (mins): ",round((run_end-run_start)/60,2),"| Datetime:",str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
            if not battery:
                return model