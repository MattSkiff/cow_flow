# External
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, StepLR, CyclicLR
import torch.nn.functional as TF
from torchvision.models import vgg16_bn, resnet18, efficientnet_b3 
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint

# tensorboard
from tqdm import tqdm # progress bar
from lcfcn  import lcfcn_loss # lcfcn
import time 
import copy
import os
import math
import types
from datetime import datetime 

# Internal
# from utils import get_loss, plot_preds,plot_preds_baselines, counts_preds_vs_actual, t2np, torch_r2, make_model_name, make_hparam_dict, save_model, init_model
import model as m# importing entire file fixes 'cyclical import' issues
import utils as u
import config as c
import gvars as g
import arguments as a
import baselines as b
import data_loader as dl
# from data_loader import preprocess_batch
from eval import eval_mnist, dmap_metrics, dmap_pr_curve, eval_baselines
               
def train_baselines(model_name,train_loader,val_loader,config={},writer=None):
    
    if a.args.mode == '': # retrieve arguments from object store if arguments missing due to conflict with ray
        a.args = config['args']
    
    model_metric_dict = {}
    modelname = m.make_model_name(train_loader)
    model_hparam_dict = u.make_hparam_dict(val_loader)
    loaded_checkpoint = session.get_checkpoint()
    start = 0
    
    # this block of code was screwing up config obj, as args don't get passed into ray worker
    # if a.args.mode != 'search':
    #     config['lr'] = a.args.learning_rate
    #     config['scheduler']  = a.args.scheduler
    #     config['optimiser']  = a.args.optim
    #     config['weight_decay'] = a.args.weight_decay
    
    if a.args.tensorboard:
        writer = SummaryWriter(log_dir=g.ABSDIR+'runs/'+a.args.schema+'/'+modelname) # /home/mks29/clones/cow_flow/runs/ # config['schema']
    else:
        writer = writer
    
    #if not a.args.model_name == 'UNet':
    # if a.args.model_name == 'CSRNet':
    #     loss = torch.nn.MSELoss(size_average=False)
    if a.args.model_name == 'UNet_seg':
        loss = torch.nn.BCEWithLogitsLoss()
    else:
        loss = torch.nn.MSELoss(size_average=True)
    
    mdl = m.init_model(feat_extractor=None,config=config)
    optimizer = choose_optimizer(model=mdl,config=config) 
    scheduler = choose_scheduler(optimizer=optimizer,config=config)
        
    if config['optimiser'] == 'adam':   
        optimizer = torch.optim.Adam(mdl.parameters(), lr=config['lr'],
                                     betas=(a.args.adam_b1, a.args.adam_b2), eps=a.args.adam_e, weight_decay=config['weight_decay'])
    if config['optimiser'] == 'adamw':
        optimizer = torch.optim.AdamW(mdl.parameters(), lr=config['lr'], 
                                     betas=(a.args.adam_b1, a.args.adam_b2), eps=a.args.adam_e, weight_decay=config['weight_decay'])
    if config['optimiser'] == 'sgd':
        optimizer = torch.optim.SGD(mdl.parameters(), lr=config['lr'],momentum=a.args.sgd_mom)
        
    mdl.to(c.device) 
    
    train_loss = []; val_loss = []; best_loss = float('inf'); l = 0
    
    start = 0; meta_epoch_start = 0
    if loaded_checkpoint:
        last_step = loaded_checkpoint.to_dict()["step"]
        start = last_step + 1
        meta_epoch_start = start // config['sub_epochs']
    
    for meta_epoch in range(meta_epoch_start,config['meta_epochs']):
        
        for sub_epoch in range(config['sub_epochs']):
            
            t_e1 = time.perf_counter()
            
            mdl.train()
            
            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                
                optimizer.zero_grad(set_to_none=False)
                images,dmaps,labels, binary_labels, annotations,point_maps = dl.preprocess_batch(data)
                
                # todo test delete
                # for i in range(10): print(images.size())
                # import time as sleepy
                # sleepy.sleep(10)
                
                results = mdl(images)
                
                if a.args.model_name == 'LCFCN':
                    
                    iter_loss = lcfcn_loss.compute_loss(points=point_maps, probs=results.sigmoid())

                # TODO: check no one-hot encoding here is ok (single class only)
                elif a.args.model_name == 'UNet_seg':
                    
                    iter_loss = (loss(input = results.squeeze(1),target = dmaps) \
                              + b.dice_loss(TF.softmax(results, dim=0).float().squeeze(1),dmaps,multiclass=False))
                    
                else:
                    
                    iter_loss = loss(results.squeeze(),dmaps.squeeze())
                    
                t_loss = u.t2np(iter_loss)
                iter_loss.backward()
                train_loss.append(t_loss)
                # disable gradient clipping
                clip_grad_value_(mdl.parameters(), c.clip_value)
                optimizer.step()
                
                if config['scheduler'] != "none":
                    scheduler.step()
            
            # Validation Loop -----
            mdl.eval()
            
            with torch.no_grad():
                
                for i, data in enumerate(tqdm(val_loader, disable=c.hide_tqdm_bar)):
                    
                    images,dmaps,labels, binary_labels, annotations,point_maps = dl.preprocess_batch(data)
                    results = mdl(images)
                    
                    if a.args.model_name == 'LCFCN':
                        
                        iter_loss = lcfcn_loss.compute_loss(points = point_maps, probs=results.sigmoid())
                        
                    elif a.args.model_name == 'UNet_seg':
                    
                        iter_loss = (loss(input = results.squeeze(1),target = dmaps) \
                                  + b.dice_loss(TF.softmax(results, dim=0).float().squeeze(1),dmaps,multiclass=False))
                    else:
                        
                        iter_loss = loss(results.squeeze(),dmaps.squeeze())
                        
                    v_loss = u.t2np(iter_loss)
                    val_loss.append(v_loss)
                    
                    l = l + 1       
        
            mean_train_loss = np.mean(train_loss)
            mean_val_loss = np.mean(val_loss)
            
            tune_save_report(epoch=l,net=mdl,optimizer=optimizer,loss=mean_val_loss)

            if mean_val_loss < best_loss and c.save_model and not a.args.mode == 'search':
                best_loss = mean_val_loss
                # At this point also save a snapshot of the current model
                u.save_model(mdl,"best"+"_"+modelname) # want to overwrite - else too many copies stored # +str(l)
                
                if a.args.viz and l % a.args.viz_freq == 0:
                    u.plot_preds_baselines(mdl,val_loader,mode="val",mdl_type=a.args.model_name,writer=writer,writer_epoch=meta_epoch,writer_mode='val')
                
            t_e2 = time.perf_counter()
            print("\nTrain | Sub Epoch Time (s): {:f}, Epoch train loss: {:.4f},Epoch val loss: {:.4f}".format(t_e2-t_e1,mean_train_loss,mean_val_loss))
            print('Meta Epoch: {:d}, Sub Epoch: {:d}, | Epoch {:d} out of {:d} Total Epochs'.format(meta_epoch, sub_epoch,meta_epoch*a.args.sub_epochs + sub_epoch+1,a.args.meta_epochs*a.args.sub_epochs))
        
        if writer != None:
            writer.add_scalar('loss/epoch_train',mean_train_loss, l)
            writer.add_scalar('loss/epoch_val',mean_val_loss, l)
            val_metric_dict = eval_baselines(mdl,val_loader,mode='val')
            model_metric_dict.update(val_metric_dict)
            print(val_metric_dict)
    
            writer.add_hparams(
                      hparam_dict = model_hparam_dict,
                      metric_dict = model_metric_dict,
                      # this will create an entry per meta epoch
                      run_name = "epoch_{}".format(meta_epoch)
                      )
        
        mdl.hparam_dict = model_hparam_dict
        mdl.metric_dict = model_metric_dict
    
    filename = "./models/"+"final"+modelname+".txt"
    
    # could switch to using json and print params on model reload
    if not os.path.exists("./models/"):
        os.makedirs("./models/")
    
    with open(filename, 'w') as f:
        print(model_hparam_dict, file=f)
    
    mdl.to('cpu')
    mdl.to(c.device)
    
    if a.args.save_final_mod:
        u.save_model(mdl,"final"+"_"+modelname)
        
    # if not a.args.skip_final_eval:
    #     val_metric_dict = eval_baselines(mdl,val_loader,mode='val')
    #     model_metric_dict.update(val_metric_dict)
    #     print(val_metric_dict)
    
    return mdl    

def train(train_loader,val_loader,head_train_loader=None,head_val_loader=None,config={},writer=None):
            
            if a.args.mode == '': # retrieve arguments from object store if arguments missing due to conflict with ray
                a.args = get(arguments_store)
        
            model_metric_dict = {}
            modelname = m.make_model_name(train_loader)
            model_hparam_dict = u.make_hparam_dict(val_loader)
            loaded_checkpoint = session.get_checkpoint()
            start = 0

            # TODO - move this out of trainable function
            # if a.args.mode != 'search':
            #     config['fixed1x1conv'] = a.args.fixed1x1conv
            #     config['noise'] = a.args.noise
            #     config['subnet_bn'] = a.args.subnet_bn
            #     config['filters'] = a.args.filters
            #     config['batch_size'] = a.args.batch_size
            #     config['lr'] = a.args.learning_rate
            #     config['feat_extractor']= a.args.feat_extractor
            #     config['weight_decay'] = a.args.weight_decay
            #     config['joint_optim'] = a.args.joint_optim
            #     config['scheduler']  = a.args.scheduler
            #     config['optimiser']  = a.args.optim
            #     config['clamp']  = c.clamp_alpha
            #     config['lr'] = a.args.learning_rate
            #     config['feat_extractor']= a.args.feat_extractor
            #     config['joint_optim'] = a.args.joint_optim
            #     config['scheduler']  = a.args.scheduler
            #     config['optimiser']  = a.args.optim
            #     config['weight_decay']  = a.args.weight_decay
    
            if c.debug:
                torch.autograd.set_detect_anomaly(True)
    
            if c.verbose: 
                print("Training run using {} train samples and {} valid samples...".format(str(len(train_loader)*int(train_loader.batch_size)),str(len(val_loader)*int(val_loader.batch_size))))
                print("Using device: {}".format(c.device))
            
            run_start = time.perf_counter()
            print("Starting run: ",str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    
            if c.verbose:
                print("Training using {} train samples and {} val samples...".format(len(train_loader)*train_loader.batch_size,
                                                                                            len(val_loader)*val_loader.batch_size))            
            if a.args.tensorboard:
                writer = SummaryWriter(log_dir='/home/mks29/clones/cow_flow/runs/'+a.args.schema+'/'+modelname)
            else:
                writer = writer
            
            # define backbone here 
            feat_extractor = m.select_feat_extractor(config['feat_extractor'],train_loader,val_loader,config=config)
            
            if a.args.data == 'mnist':
                mdl = m.MNISTFlow(modelname=modelname,feat_extractor = feat_extractor)
            else:
                # backbone attached attached during mdl init
                mdl = m.CowFlow(modelname=modelname,feat_extractor = feat_extractor,config=config)
            
            c_head_trained = False
            
            optimizer = choose_optimizer(model=mdl,config=config)   
            scheduler = choose_scheduler(config=config,optimizer=optimizer)
        
            
            if config['joint_optim'] and config['feat_extractor'] != 'none':
                # TODO
                if config['optimiser'] == 'adam':   
                    optimizer = torch.optim.Adam([
                                {'params': mdl.nf.parameters()},
                                {'params': mdl.feat_extractor.parameters(), 'lr_init': a.args.fe_lr,'betas':(a.args.fe_b1,a.args.fe_b2),'eps':1e-08, 'weight_decay':a.args.fe_wd}
                            ], lr=config['lr'], betas=(a.args.adam_b1, a.args.adam_b2), eps=a.args.adam_e, weight_decay=config['weight_decay'] )      
                    
                if config['optimiser'] == 'adamw':   
                    optimizer = torch.optim.AdamW([
                                {'params': mdl.nf.parameters()},
                                {'params': mdl.feat_extractor.parameters(), 'lr_init': a.args.fe_lr,'betas':(a.args.fe_b1,a.args.fe_b2),'eps':1e-08, 'weight_decay':a.args.fe_wd}
                            ], lr=config['lr'], betas=(a.args.adam_b1, a.args.adam_b2), eps=a.args.adam_e, weight_decay=config['weight_decay'] )  
                
                if config['optimiser'] == 'sgd':
                    optimizer = torch.optim.SGD([
                                {'params': mdl.nf.parameters()},
                                {'params': mdl.feat_extractor.parameters(), 'lr_init': a.args.fe_lr,'weight_decay':config['weight_decay']}
                            ], lr=config['lr'],momentum=a.args.sgd_mom)
            else:
                
                if config['optimiser'] == 'adam':   
                    optimizer = torch.optim.Adam(mdl.parameters(), lr=config['lr'], betas=(a.args.adam_b1, a.args.adam_b2), eps=a.args.adam_e, weight_decay=config['weight_decay'])
                if config['optimiser'] == 'adamw':   
                    optimizer = torch.optim.AdamW(mdl.parameters(), lr=config['lr'], betas=(a.args.adam_b1, a.args.adam_b2), eps=a.args.adam_e, weight_decay=config['weight_decay'])
                if config['optimiser'] == 'sgd':
                    optimizer = torch.optim.SGD(mdl.parameters(), lr=config['lr'],momentum=a.args.sgd_mom)   
            
            mdl.to(c.device)   
            
            k = 0 # track total mini batches
            val_mb_iter = 0
            train_mb_iter = 0
            best_loss = float('inf')
            j = 0 # track total sub epochs
            l = 0 # track meta epochs
            
            start = 0; meta_epoch_start = 0
            if loaded_checkpoint:
                last_step = loaded_checkpoint.to_dict()["step"]
                start = last_step + 1
                meta_epoch_start = start // a.args.sub_epochs
            
            for meta_epoch in range(meta_epoch_start,a.args.meta_epochs):
                
                ### Train Loop ------
                for sub_epoch in range(a.args.sub_epochs):
                    
                    mdl.train()
                    
                    if c.verbose:
                        t_e1 = time.perf_counter()
                    
                    train_loss = list()
                    
                    if c.debug and config['scheduler'] != 'none':
                        print('Initial Scheduler Learning Rate: ',scheduler.get_lr()[0])
                    
                    for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                        
                        t1 = time.perf_counter()
                        
                        optimizer.zero_grad(set_to_none=False)
                        
                        # TODO - can this section
                        if a.args.data == 'dlr':
                            images,dmaps,counts,point_maps = data = dl.preprocess_batch(data,dlr=True)
                        elif not a.args.data == 'mnist' and not c.counts and not train_loader.dataset.classification:
                            images,dmaps,labels,annotations, point_maps = data 
                        elif not a.args.data == 'mnist' and not c.counts:
                            images,dmaps,labels, binary_labels, annotations,point_maps = dl.preprocess_batch(data)
                            #images,dmaps,labels,annotations,binary_labels,point_maps  = data 
                        elif not a.args.data == 'mnist':
                            images,dmaps,labels,counts, point_maps = data
                        else:
                            images,labels = data
                            
                        images = images.float().to(c.device)
                        
                        if not a.args.data == 'mnist' and not c.counts:
                            dmaps = dmaps.float().to(c.device)
                        elif not a.args.data == 'mnist':
                            counts = counts.float().to(c.device)
                        else: 
                            labels = labels.float().to(c.device)
                        
                        # z is the probability density under the latent normal distribution
                        # output of model - two elements: y (z), jacobian
                          
                        if not a.args.data == 'mnist' and not c.counts:
                            input_data = (images,dmaps*a.args.dmap_scaling) # inputs features,dmaps
                        elif not a.args.data == 'mnist':
                            input_data = (images,counts) 
                        else:
                            input_data = (images,labels)
                        
                        z, log_det_jac = mdl(*input_data,jac=a.args.jac)
                            
                        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True) [FrEIA]
                        # i.e. what y (output) is depends on direction of flow
                        
                        # this loss needs to calc distance between predicted density and density map
                        # note: probably going to get an error with mnist or counts # TODO
                        dims = tuple(range(1, len(z.size())))
                        loss = u.get_loss(z, log_det_jac,dims) 
                        k += 1
                        train_mb_iter += 1
                        
                        loss_t = u.t2np(loss)
                        
                        if c.debug:
                            print('loss/minibatch_train',train_mb_iter)
                        
                        if writer != None:
                            writer.add_scalar('loss/minibatch_train',loss, train_mb_iter)
                        
                        train_loss.append(loss_t)
                        loss.backward()
                        clip_grad_value_(mdl.parameters(), c.clip_value)
                        optimizer.step()
                        
                        if config['scheduler'] != "none":
                            scheduler.step()
                        
                        t2 = time.perf_counter()
                        
                        total_iter = len(train_loader) * a.args.sub_epochs * a.args.meta_epochs + (len(val_loader) * a.args.meta_epochs)
                        
                        if k % c.report_freq == 0 and c.verbose and k != 0 and c.report_freq != -1:
                                print('\nTrain | Mini-Batch Time: {:.1f}, Mini-Batch: {:d}, Mini-Batch loss: {:.4f}'.format(t2-t1,i+1, loss_t))
                                print('{:d} Mini-Batches in sub-epoch remaining'.format(len(train_loader)-i))
                                print('Total Iterations: ',total_iter,'| Passed Iterations: ',k)
                                
                        if c.debug and not a.args.data == 'mnist':
                            print("number of elements in density maps list:")
                            print(len(dmaps)) # images
                            print("number of images in image tensor:")
                            print(len(images)) # features                
                    
                    mean_train_loss = np.mean(train_loss)
                    
                    if c.verbose:
                        t_e2 = time.perf_counter()
                        print("\nTrain | Sub Epoch Time (s): {:f}, Epoch loss: {:.4f}".format(t_e2-t_e1,mean_train_loss))
                        print('Meta Epoch: {:d}, Sub Epoch: {:d}, | Epoch {:d} out of {:d} Total Epochs'.format(meta_epoch, sub_epoch,meta_epoch*a.args.sub_epochs + sub_epoch+1,a.args.meta_epochs*a.args.sub_epochs))
                    
                    if c.debug:
                        print("loss/epoch_train:",meta_epoch)
                    
                    if writer != None:
                        writer.add_scalar('loss/epoch_train',mean_train_loss, meta_epoch)
                    
                    mdl.eval()
                    
                    ### Validation Loop ------
                    if c.validation: 
                        val_loss = list()
                        val_z = list()
                        # val_coords = list() # todo
                        
                        with torch.no_grad():
                            
                            for i, data in enumerate(tqdm(val_loader, disable=c.hide_tqdm_bar)):
                                
                                # Validation
                                # TODO - DRY violation...
                                if a.args.data == 'dlr':
                                    images,dmaps,counts,point_maps = data
                                elif not a.args.data == 'mnist' and not c.counts and not train_loader.dataset.classification:
                                    images,dmaps,labels, annotations,point_maps = data 
                                elif not a.args.data == 'mnist' and not c.counts:
                                    #images,dmaps,labels, binary_labels, annotations,point_maps = preprocess_batch(data)
                                    images,dmaps,labels,annotations,binary_labels,point_maps  = data 
                                elif not a.args.data == 'mnist':
                                    images,dmaps,labels,counts = data
                                else:
                                    images,labels = data
                            
                                z, log_det_jac = mdl(*input_data,jac=a.args.jac)
                                    
                                val_z.append(z)
                                
                                if i % c.report_freq == 0 and c.debug and not a.args.data == 'mnist' and not c.counts:
                                        print('val true count: {:f}'.format(dmaps.sum()))
                                    
                                dims = tuple(range(1, len(z.size())))
                                loss = u.get_loss(z, log_det_jac,dims)
                                k += 1
                                val_mb_iter += 1
                                val_loss.append(u.t2np(loss))
                                
                                if writer != None:
                                    writer.add_scalar('loss/minibatch_val',loss, val_mb_iter)
                             
                        mean_val_loss = np.mean(np.array(val_loss))
                            
                        tune_save_report(epoch=j,net=mdl,optimizer=optimizer,loss=mean_val_loss)
                         
                        if c.verbose:
                                print('Validation | Sub Epoch: {:d} \t Epoch loss: {:4f}'.format(sub_epoch,mean_val_loss))
                        
                        if writer != None:
                            writer.add_scalar('loss/epoch_val',mean_val_loss, meta_epoch)
                        
                        # simple early stopping based on val loss
                        # https://stackoverflow.com/questions/68929471/implementing-early-stopping-in-pytorch-without-torchsample
                        if mean_val_loss < best_loss and c.save_model and not a.args.mode == 'search':
                            best_loss = mean_val_loss
                            
                            # At this point also save a snapshot of the current model
                            u.save_model(mdl,"best"+"_"+modelname) # want to overwrite - else too many copies stored # +str(l)
                            
                        j += 1
                
                # train classification head to filter out nul patches
                
                if not a.args.data == 'dlr' and not mdl.mnist and not c_head_trained and a.args.train_classification_head:
                    c_head_trained = True
                    mdl.classification_head = train_classification_head(mdl,head_train_loader,head_val_loader)
                 
                #### Meta epoch code (metrics) ------
                with torch.no_grad():
                    if mdl.mnist:
                        val_acc, train_acc = eval_mnist(mdl,val_loader,train_loader)
                        print("\n")
                        print("Training Accuracy: ", train_acc,"| Epoch: ",meta_epoch)
                        print("val Accuracy: ",val_acc,"| Epoch: ",meta_epoch)
                    
                    if c.save_model and c.checkpoints and not a.args.mode == 'search':
                        mdl.to('cpu')
                        u.save_model(mdl,"checkpoint_"+str(l)+"_"+modelname)
                        #save_weights(model,"checkpoint_"+str(l)+"_"+modelname) # currently have no use for saving weights
                        mdl.to(c.device)
                    
                    if writer != None:
                        # DMAP Count Metrics - y,y_n,y_hat_n,y_hat_n_dists,y_hat_coords
                        # add images to TB writer
                        if a.args.viz and j % a.args.viz_freq == 0:
                            u.plot_preds(mdl,train_loader,writer=writer,writer_epoch=meta_epoch,writer_mode='train',null_filter=False)
                            
                        train_metric_dict = dmap_metrics(mdl, train_loader,n=c.eval_n,mode='train')
                        print(train_metric_dict)
                        model_metric_dict.update(train_metric_dict)
                        
                        if a.args.viz and mdl.dlr_acd and j % a.args.viz_freq == 0:
                            u.plot_preds(mdl,train_loader,writer=writer,writer_epoch=meta_epoch,writer_mode='train',null_filter=False)
                        
                        if c.validation:
                            if a.args.viz and j % a.args.viz_freq == 0:
                                u.plot_preds(mdl,val_loader,writer=writer,writer_epoch=meta_epoch,writer_mode='val',null_filter=False)
                                
                            val_metric_dict = dmap_metrics(mdl, val_loader,n=c.eval_n,mode='val')
                            model_metric_dict.update(val_metric_dict)
    
                    # MNIST Model metrics
                    if writer != None and mdl.mnist:
                        writer.add_scalar('acc/meta_epoch_train',train_acc, meta_epoch)
                        model_metric_dict['acc/meta_epoch_train'] = train_acc
                        
                        writer.add_scalar('acc/meta_epoch_val',val_acc, meta_epoch)
                        model_metric_dict['acc/meta_epoch_val'] = val_acc
                    
                    # Count Model Metrics
                    if writer != None and mdl.count:
                        train_R2 = u.torch_r2(mdl,train_loader)
                        writer.add_scalar('R2/meta_epoch_train',train_R2, meta_epoch)
                        model_metric_dict['R2/meta_epoch_train'] = train_R2
                        
                        if c.verbose:
                            print("Train R2: ",train_R2)
                        
                        # TODO (MAPE or RMSE)
                        #writer.add_scalar('acc/meta_epoch_val',val_acc, l)
                        #model_metric_dict['acc/meta_epoch_val'] = val_acc
                        
                        if c.validation:
                            val_R2 = u.torch_r2(mdl,val_loader)
                            writer.add_scalar('R2/meta_epoch_val',val_R2, meta_epoch)
                            model_metric_dict['R2/meta_epoch_val'] = val_R2
                            
                            if c.verbose:
                                print("Val R2: ",val_R2)
                    
                    # add param tensorboard scalars
                    if writer != None:
                        
                        for name,value in model_metric_dict.items():
                            writer.add_scalar(tag=name, scalar_value=value,global_step=meta_epoch)
                        
                        writer.add_hparams(
                                  hparam_dict = model_hparam_dict,
                                  metric_dict = model_metric_dict,
                                  # TOD0 - this will create an entry per meta epoch
                                  run_name = "epoch_{}".format(meta_epoch)
                                  )
                        
                        writer.flush()
                    
                    l += 1
                    
                # #### Meta epoch code (metrics) ------
                
                # # train classification head to filter out null patches
                # if not a.args.data == 'dlr' and not mdl.mnist and not c_head_trained and a.args.train_classification_head:
                #     c_head_trained = True
                #     mdl.classification_head = train_classification_head(mdl,head_train_loader,head_val_loader)
                
                # if mdl.mnist:
                #     val_acc, train_acc = eval_mnist(mdl,val_loader,train_loader)
                #     print("\n")
                #     print("Training Accuracy: ", train_acc,"| Epoch: ",meta_epoch)
                #     print("val Accuracy: ",val_acc,"| Epoch: ",meta_epoch)
                
                # if c.save_model and c.checkpoints and not a.args.mode == 'search':
                #     mdl.to('cpu')
                #     save_model(mdl,"checkpoint_"+str(l)+"_"+modelname)
                #     #save_weights(model,"checkpoint_"+str(l)+"_"+modelname) # currently have no use for saving weights
                #     mdl.to(c.device)
                
                # if writer != None:
                #     # DMAP Count Metrics - y,y_n,y_hat_n,y_hat_n_dists,y_hat_coords
                #     # add images to TB writer
                #     if a.args.viz and j % a.args.viz_freq == 0:
                #         plot_preds(mdl,train_loader,writer=writer,writer_epoch=meta_epoch,writer_mode='train',null_filter=False)
                        
                #     train_metric_dict = dmap_metrics(mdl, train_loader,n=c.eval_n,mode='train')
                #     print(train_metric_dict)
                #     model_metric_dict.update(train_metric_dict)
                    
                #     if a.args.viz and mdl.dlr_acd and j % a.args.viz_freq == 0:
                #         plot_preds(mdl,train_loader,writer=writer,writer_epoch=meta_epoch,writer_mode='train',null_filter=False)
                    
                #     if c.validation:
                #         if a.args.viz and j % a.args.viz_freq == 0:
                #             plot_preds(mdl,val_loader,writer=writer,writer_epoch=meta_epoch,writer_mode='val',null_filter=False)
                            
                #         val_metric_dict = dmap_metrics(mdl, val_loader,n=c.eval_n,mode='val')
                #         model_metric_dict.update(val_metric_dict)

                # # MNIST Model metrics
                # if writer != None and mdl.mnist:
                #     writer.add_scalar('acc/meta_epoch_train',train_acc, meta_epoch)
                #     model_metric_dict['acc/meta_epoch_train'] = train_acc
                    
                #     writer.add_scalar('acc/meta_epoch_val',val_acc, meta_epoch)
                #     model_metric_dict['acc/meta_epoch_val'] = val_acc
                
                # # Count Model Metrics
                # if writer != None and mdl.count:
                #     train_R2 = torch_r2(mdl,train_loader)
                #     writer.add_scalar('R2/meta_epoch_train',train_R2, meta_epoch)
                #     model_metric_dict['R2/meta_epoch_train'] = train_R2
                    
                #     if c.verbose:
                #         print("Train R2: ",train_R2)
                    
                #     # TODO (MAPE or RMSE)
                #     #writer.add_scalar('acc/meta_epoch_val',val_acc, l)
                #     #model_metric_dict['acc/meta_epoch_val'] = val_acc
                    
                #     if c.validation:
                #         val_R2 = torch_r2(mdl,val_loader)
                #         writer.add_scalar('R2/meta_epoch_val',val_R2, meta_epoch)
                #         model_metric_dict['R2/meta_epoch_val'] = val_R2
                        
                #         if c.verbose:
                #             print("Val R2: ",val_R2)
                
                # # add param tensorboard scalars
                # if writer != None:
                    
                #     for name,value in model_metric_dict.items():
                #         writer.add_scalar(tag=name, scalar_value=value,global_step=meta_epoch)
                    
                #     writer.add_hparams(
                #               hparam_dict = model_hparam_dict,
                #               metric_dict = model_metric_dict,
                #               # TOD0 - this will create an entry per meta epoch
                #               run_name = "epoch_{}".format(meta_epoch)
                #               )
                    
                #     writer.flush()
                
                # l += 1
            
            # visualise a random reconstruction
            if a.args.viz:
                
                if mdl.dlr_acd:
                    preds_loader = train_loader
                    dmap_pr_mode = 'train'
                else:
                    preds_loader = val_loader
                    dmap_pr_mode = 'val'
                    
                u.plot_preds(mdl, preds_loader, plot = True,null_filter=False)
                # TODO skip PR curve, as this funciton takes 20 minutes
                # dmap_pr_curve(mdl, preds_loader,n = 10,mode = dmap_pr_mode)
                
                if c.counts:
                    print("Plotting Train R2")
                    u.counts_preds_vs_actual(mdl,train_loader,plot=a.args.viz)
            
            ### Post-Training ---
            mdl.hparam_dict = model_hparam_dict
            mdl.metric_dict = model_metric_dict
            
            # save model hparams, unless models are being saved at end of every meta peoch
            if c.save_model and not a.args.mode == 'search':
                
                hp_filename = "./models/"+"hparams"+modelname+".txt"
                
                # could switch to using json and print params on model reload
                with open(hp_filename, 'w') as f:
                    print(model_hparam_dict, file=f)
            
            #if not a.args.data == 'dlr':
            # print("Performing final evaluation without trained null classifier (train loader)...")
            
            if not a.args.skip_final_eval:
                final_metrics = dmap_metrics(mdl, train_loader,n=1,mode='train',null_filter = False)
                print(final_metrics)
            
            if a.args.save_final_mod and not a.args.mode == 'search':
                u.save_model(mdl,"final"+"_"+modelname)
            
            run_end = time.perf_counter()
            print("Finished Model: ",modelname)
            print("Run finished. Time Elapsed (mins): ",round((run_end-run_start)/60,2),"| Datetime:",str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

            return mdl

# train classification head to predict empty patches
def train_classification_head(mdl,full_trainloader,full_valloader,criterion = nn.CrossEntropyLoss()):

    if c.verbose:
        print("Optimizing classification head for null filtering...")
    
    if mdl != None:
        mdl.to(c.device) 
    else:
        mdl = types.SimpleNamespace()
        
        if a.args.feat_extractor == 'resnet18':
            mdl.classification_head = resnet18(pretrained=a.args.pretrained,progress=False)
        elif a.args.feat_extractor == 'vgg16_bn':
            mdl.classification_head = vgg16_bn(pretrained=a.args.pretrained,progress=False)
            
        mdl.classification_head.to(c.device)
    
    now = datetime.now() 
    filename = "_".join([
                a.args.feat_extractor,
                'FTE',str(c.feat_extractor_epochs),
                str(now.strftime("%d_%m_%Y_%H_%M_%S")),
                "PT",str(c.pretrained),
                'BS',str(full_trainloader.batch_size),
                'classification_head'
            ])
    
    if not c.debug:
        writer = SummaryWriter(log_dir='/home/mks29/clones/cow_flow/runs/feat_pretrained/'+filename)
    else:
        writer = None
    
    optimizer = torch.optim.Adam(mdl.classification_head.parameters(),lr=1e-3,betas=(0.9,0.999),eps=1e-08, weight_decay=0)
    best_model_wts = copy.deepcopy(mdl.classification_head.state_dict())
    best_acc = 0.0
    
    t_sz = len(full_trainloader)*full_trainloader.batch_size
    v_sz = len(full_valloader)*full_valloader.batch_size
    minibatch_count = 0 
    
    dataset_sizes = {'train': t_sz,'val': v_sz}

    for epoch in range(c.feat_extractor_epochs):
        
        for phase in ['train', 'val']:
            if phase == 'train':
                mdl.classification_head.train()  
                loader = full_trainloader
            else:
                mdl.classification_head.eval()
                loader = full_valloader
            
            if c.verbose:
                print('Classification Head {} Epoch {}/{}'.format(phase,epoch, c.feat_extractor_epochs)) 
                
            running_loss = 0.0; running_corrects = 0
            
            for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
                
                optimizer.zero_grad(set_to_none=False)
                
                images = data[0].to(c.device)
                binary_labels = data[3].to(c.device)
                
                if c.debug:
                    print('feature extractor labels: ',binary_labels)
                
                with torch.set_grad_enabled(phase == 'train'):
                    
                    #features = mdl.feat_extractor(images)
                    #outputs = mdl.classification_head(features)
                    
                    if c.debug:
                        print('images shape')
                        print(images.shape)
                    
                    outputs = mdl.classification_head(images)
                    _, preds = torch.max(outputs, 1) 
                    
                    loss = criterion(outputs,binary_labels)
                    minibatch_count += 1
                
                    if writer != None:
                         writer.add_scalar('loss/minibatch_{}'.format(phase),loss.item(), minibatch_count)
                         
                if phase == 'train':     
                    loss.backward()
                    clip_grad_value_(mdl.classification_head.parameters(), c.clip_value)
                    optimizer.step()
                
                # statistics
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == binary_labels.data)
                      
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
            if c.verbose:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)) 
            
            if writer != None:
                writer.add_scalar('acc/epoch_{}'.format(phase),epoch_acc, epoch) # TODO
                writer.add_scalar('loss/epoch_{}'.format(phase),epoch_loss, epoch)
            
            running_loss = 0.0; running_corrects = 0
        
            # deep copy the model with best val acc
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(mdl.classification_head.state_dict())
                
    if c.verbose:
        print("Finetuning finished.")
    
    mdl.classification_head.load_state_dict(best_model_wts)
    
    if not a.args.mode == 'search':
        u.save_model(mdl.classification_head,filename=filename,loc=g.FEAT_MOD_DIR)
    
    return mdl.classification_head
        
# from pytorch tutorial...           
def train_feat_extractor(feat_extractor,trainloader,valloader,criterion = nn.CrossEntropyLoss()):

    if c.verbose:
        print("Finetuning feature extractor...")
    
    now = datetime.now() 
    filename = "_".join([
                a.args.feat_extractor,
                'FTE',str(c.feat_extractor_epochs),
                str(now.strftime("%d_%m_%Y_%H_%M_%S")),
                "PT",str(c.pretrained),
                'BS',str(trainloader.batch_size)
            ])
    
    if not c.debug:
        writer = SummaryWriter(log_dir='/home/mks29/clones/cow_flow/runs/feat_pretrained/'+filename)
    else:
        writer = None
    
    optimizer = torch.optim.Adam(feat_extractor.parameters(),lr=1e-3,betas=(0.9,0.999),eps=1e-08, weight_decay=0)
    best_model_wts = copy.deepcopy(feat_extractor.state_dict())
    best_acc = 0.0
    
    t_sz = len(trainloader)*trainloader.batch_size
    v_sz = len(valloader)*valloader.batch_size
    minibatch_count = 0 
    
    dataset_sizes = {'train': t_sz,'val': v_sz}
   
    feat_extractor.to(c.device)        
    
    for epoch in range(c.feat_extractor_epochs):
        
        for phase in ['train', 'val']:
            if phase == 'train':
                feat_extractor.train()  
                loader = trainloader
            else:
                feat_extractor.eval()
                loader = valloader
            
            if c.verbose:
                print('Feature Extractor {} Epoch {}/{}'.format(phase,epoch, c.feat_extractor_epochs)) 
                
            running_loss = 0.0; running_corrects = 0
            
            for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):

                optimizer.zero_grad(set_to_none=False)
                
                images = data[0].to(c.device)
                binary_labels = data[3].to(c.device)
                
                if c.debug:
                    print('feature extractor labels: ',binary_labels)
                
                with torch.set_grad_enabled(phase == 'train'):
                    
                    
                    
                    outputs = feat_extractor(images)
                    _, preds = torch.max(outputs, 1)   
                    
                    loss = criterion(outputs,binary_labels)
                    minibatch_count += 1
                
                    if writer != None:
                         writer.add_scalar('loss/minibatch_{}'.format(phase),loss.item(), minibatch_count)
                         
                if phase == 'train':     
                    loss.backward()
                    clip_grad_value_(feat_extractor.parameters(), c.clip_value)
                    optimizer.step()
                
                # statistics
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == binary_labels.data)
                
                if c.debug:
                    print('minibatch: ',minibatch_count)
                    print('binary_labels.data: ',binary_labels.data)
                    print('preds: ',preds)
                    print('running corrects: ',running_corrects)
                      
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
            if c.verbose:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)) 
            
            if writer != None:
                writer.add_scalar('acc/epoch_{}'.format(phase),epoch_acc, epoch) # TODO
                writer.add_scalar('loss/epoch_{}'.format(phase),epoch_loss, epoch)
            
            running_loss = 0.0; running_corrects = 0
            
            # deep copy the model with best val acc
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(feat_extractor.state_dict())
                
    if c.verbose:
        print("Finetuning finished.")
        
    feat_extractor.load_state_dict(best_model_wts)
    u.save_model(feat_extractor,filename=filename,loc=g.FEAT_MOD_DIR)
    
    return feat_extractor

def tune_save_report(epoch,net,optimizer,loss):
    
    if a.args.mode == 'search':
        
        if math.isnan(loss):
            loss = 1e99
        else:
            loss = loss
        
        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((net.state_dict(), optimizer.state_dict()), path)
        
        state_dict = net.state_dict()
        metrics = {"loss": loss}
        checkpoint = Checkpoint.from_dict(
            dict(epoch=epoch, model_weights=state_dict)
        )
        
        session.report(metrics, checkpoint=checkpoint)
        
def choose_scheduler(config=None,optimizer=None):
    
    assert config, optimizer
    
    # add scheduler to improve stability further into training
    if config['scheduler'] == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=a.args.expon_gamma)
    elif config['scheduler'] == "step":
        scheduler = StepLR(optimizer,step_size=a.args.step_size,gamma=a.args.step_gamma)
    elif config['scheduler'] == "cyclic":
        scheduler = CyclicLR(optimizer,base_lr=g.MIN_LR,max_lr=g.MAX_LR)
    else:
        scheduler = "None"
        
    return scheduler

def choose_optimizer(config=None,model=None):
    
    mdl = model   

    if config['model_name'] == 'NF' and config['joint_optim'] and config['feat_extractor'] != 'none':

        if config['optimiser'] == 'adam':   
            optimizer = torch.optim.Adam([
                        {'params': mdl.nf.parameters()},
                        {'params': mdl.feat_extractor.parameters(), 'lr_init': a.args.fe_lr,'betas':(a.args.fe_b1,a.args.fe_b2),'eps':1e-08, 'weight_decay':a.args.fe_wd}
                    ], lr=config['lr'], betas=(a.args.adam_b1, a.args.adam_b2), eps=a.args.adam_e, weight_decay=config['weight_decay'] )      
            
        if config['optimiser'] == 'adamw':   
            optimizer = torch.optim.AdamW([
                        {'params': mdl.nf.parameters()},
                        {'params': mdl.feat_extractor.parameters(), 'lr_init': a.args.fe_lr,'betas':(a.args.fe_b1,a.args.fe_b2),'eps':1e-08, 'weight_decay':a.args.fe_wd}
                    ], lr=config['lr'], betas=(a.args.adam_b1, a.args.adam_b2), eps=a.args.adam_e, weight_decay=config['weight_decay'] )  
        
        if config['optimiser'] == 'sgd':
            optimizer = torch.optim.SGD([
                        {'params': mdl.nf.parameters()},
                        {'params': mdl.feat_extractor.parameters(), 'lr_init': a.args.fe_lr,'weight_decay':config['weight_decay']}
                    ], lr=config['lr'],momentum=a.args.sgd_mom)
    else:
    
        if config['optimiser'] == 'adam':   
            optimizer = torch.optim.Adam(mdl.parameters(), lr=config['lr'], betas=(a.args.adam_b1, a.args.adam_b2), eps=a.args.adam_e, weight_decay=config['weight_decay'])
        if config['optimiser'] == 'adamw':   
            optimizer = torch.optim.AdamW(mdl.parameters(), lr=config['lr'], betas=(a.args.adam_b1, a.args.adam_b2), eps=a.args.adam_e, weight_decay=config['weight_decay'])
        if config['optimiser'] == 'sgd':
            optimizer = torch.optim.SGD(mdl.parameters(), lr=config['lr'],momentum=a.args.sgd_mom) 
    
    return optimizer
