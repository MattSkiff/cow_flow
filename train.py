import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as TF
from tqdm import tqdm # progress bar

import time 
import copy
from datetime import datetime 

from eval import dmap_metrics, eval_baselines
from utils import get_loss, plot_preds, t2np, make_model_name, make_hparam_dict, save_model
from lcfcn  import lcfcn_loss # lcfcn
import model # importing entire file fixes 'cyclical import' issues

import config as c
import gvars as g
import arguments as a
import baselines as b

# tensorboard
from torch.utils.tensorboard import SummaryWriter
               
def train_baselines(model_name,train_loader,val_loader):
    assert c.train_model
    model_metric_dict = {}
    modelname = make_model_name(train_loader)
    model_hparam_dict = make_hparam_dict(val_loader)
    writer = SummaryWriter(log_dir='runs/'+a.args.schema+'/'+modelname)   
    
    #if not a.args.model_name == 'UNet':
    if a.args.model_name == 'CSRNet':
        loss = torch.nn.MSELoss(size_average=False)
    elif a.args.model_name == 'UNet_seg':
        loss = torch.nn.BCEWithLogitsLoss()
    else:
        loss = torch.nn.MSELoss()
    
    if a.args.model_name == "FCRN":
        mdl = b.FCRN_A(modelname=modelname)
    elif a.args.model_name == "UNet":
        mdl = b.UNet(modelname=modelname)
    elif a.args.model_name == "UNet_seg":
        mdl = b.UNet(modelname=modelname,seg=True)        
    elif a.args.model_name == "CSRNet":
        mdl = b.CSRNet(modelname=modelname)
    elif a.args.model_name == "LCFCN":
        mdl = b.LCFCN(modelname=modelname)
        
    if a.args.optim == 'adam':   
        optimizer = torch.optim.Adam(mdl.parameters(), lr=a.args.learning_rate, betas=(a.args.adam_b1, a.args.adam_b2), eps=a.args.adam_e, weight_decay=a.args.weight_decay)
    if a.args.optim == 'sgd':
        optimizer = torch.optim.SGD(mdl.parameters(), lr=a.args.learning_rate,momentum=a.args.sgd_mom)

    # add scheduler to improve stability further into training
    if a.args.scheduler == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif a.args.scheduler == "step":
        scheduler = StepLR(optimizer,step_size=a.args.step_size,gamma=a.args.step_gamma)
        
    mdl.to(c.device) 
    
    train_loss = []; val_loss = []; best_loss = float('inf'); l = 0
    
    for meta_epoch in range(a.args.meta_epochs):
        
        for sub_epoch in range(a.args.sub_epochs):
            
            t_e1 = time.perf_counter()
            
            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                
                optimizer.zero_grad()
                
                images,dmaps,labels, binary_labels, annotations,point_maps = data
                images = images.float().to(c.device)
                results = mdl(images)
                
                if a.args.model_name == 'LCFCN':
                    
                    iter_loss = lcfcn_loss.compute_loss(points=point_maps, probs=results.sigmoid())*a.args.dmap_scaling
                 
                # TODO: check no one-hot encoding here is ok (single class only)
                elif a.args.model_name == 'UNet_seg':
                    
                    iter_loss = (loss(input = results.squeeze(),target = dmaps) \
                              + b.dice_loss(TF.softmax(results, dim=0).float().squeeze(),dmaps,multiclass=False))*a.args.dmap_scaling
                    
                else:
                    
                    iter_loss = loss(results.squeeze(),dmaps.squeeze()*a.args.dmap_scaling)
                    
                t_loss = t2np(iter_loss)
                iter_loss.backward()
                train_loss.append(t_loss)
                clip_grad_value_(mdl.parameters(), c.clip_value)
                optimizer.step()
                
                if a.args.scheduler != "none":
                    scheduler.step()
            
            with torch.no_grad():
                
                for i, data in enumerate(tqdm(val_loader, disable=c.hide_tqdm_bar)):
                    
                    images,dmaps,labels, binary_labels, annotations,point_maps = data 
                    images = images.float().to(c.device)
                    results = mdl(images)
                    
                    if a.args.model_name == 'LCFCN':
                        iter_loss = lcfcn_loss.compute_loss(points=point_maps, probs=results.sigmoid())
                    elif a.args.model_name == 'UNet_seg':
                        
                        iter_loss = (loss(input = results.squeeze(),target = dmaps) \
                                  + b.dice_loss(TF.softmax(results, dim=0).float().squeeze(),dmaps,multiclass=False))*a.args.dmap_scaling
                    else:
                        iter_loss = loss(results.squeeze(),dmaps.squeeze()*a.args.dmap_scaling)
                        
                    v_loss = t2np(iter_loss)
                    val_loss.append(v_loss)
                
            mean_train_loss = np.mean(train_loss)
            mean_val_loss = np.mean(val_loss)
            l = l + 1
            
            if mean_val_loss < best_loss and c.save_model:
                best_loss = mean_val_loss
                # At this point also save a snapshot of the current model
                save_model(mdl,"best"+"_"+modelname) # want to overwrite - else too many copies stored # +str(l)
                
            t_e2 = time.perf_counter()
            print("\nTrain | Sub Epoch Time (s): {:f}, Epoch train loss: {:.4f},Epoch val loss: {:.4f}".format(t_e2-t_e1,mean_train_loss,mean_val_loss))
            print('Meta Epoch: {:d}, Sub Epoch: {:d}, | Epoch {:d} out of {:d} Total Epochs'.format(meta_epoch, sub_epoch,meta_epoch*a.args.sub_epochs + sub_epoch+1,a.args.meta_epochs*a.args.sub_epochs))
            
            writer.add_scalar('loss/epoch_train',mean_train_loss, l)
            writer.add_scalar('loss/epoch_val',mean_val_loss, l)
    
        val_metric_dict = eval_baselines(mdl,val_loader,mode='val',thres=c.sigma*2)
        model_metric_dict.update(val_metric_dict)
    
    writer.add_hparams(
              hparam_dict = model_hparam_dict,
              metric_dict = model_metric_dict,
              run_name = "epoch_{}".format(meta_epoch)
              )
    
    mdl.hparam_dict = model_hparam_dict
    mdl.metric_dict = model_metric_dict
    
    filename = "./models/"+"final"+modelname+".txt"
    
    with open(filename, 'w') as f:
        print(model_hparam_dict, file=f)
    
    mdl.to('cpu')
    save_model(mdl,modelname)
    mdl.to(c.device)
    
    return mdl    

def train(train_loader,val_loader,head_train_loader=None,head_val_loader=None,writer=None):
    
            print("Training run using {} train samples and {} valid samples...".format(str(len(train_loader)*int(train_loader.batch_size)),str(len(val_loader)*int(val_loader.batch_size))))
            print("Using device: {}".format(c.device))
                     
            modelname = make_model_name(train_loader)
            model_hparam_dict = make_hparam_dict(val_loader)
            
            run_start = time.perf_counter()
            print("Starting run: ",str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
            print("Training using {} train samples and {} val samples...".format(len(train_loader)*train_loader.batch_size,
                                                                                            len(val_loader)*val_loader.batch_size))
    
            if a.args.tensorboard:
                writer = SummaryWriter(log_dir='runs/'+a.args.schema+'/'+modelname)
            else:
                writer = writer
            
            feat_extractor = model.select_feat_extractor(c.feat_extractor,train_loader,val_loader)
            mdl = model.CowFlow(modelname=modelname,feat_extractor = feat_extractor)
            
            c_head_trained = False    
            
            if a.args.optim == 'adam':   
                
                optimizer = torch.optim.Adam([
                            {'params': mdl.nf.parameters()},
                            {'params': mdl.feat_extractor.parameters(), 'lr_init': 1e-3,'betas':(0.9,0.999),'eps':1e-08, 'weight_decay':0}
                        ], lr=a.args.learning_rate, betas=(a.args.adam_b1, a.args.adam_b2), eps=a.args.adam_e, weight_decay=a.args.weight_decay )              
            
            if a.args.optim == 'sgd':
                optimizer = torch.optim.SGD([
                            {'params': mdl.nf.parameters()},
                            {'params': mdl.feat_extractor.parameters(),'lr': a.args.learning_rate,'momentum': 0.9 }
                        ], lr=a.args.learning_rate,momentum=a.args.sgd_mom)  
                    
            # add scheduler to improve stability further into training
            if a.args.scheduler == "exponential":
                scheduler = ExponentialLR(optimizer, gamma=a.args.ex_gamma)
            elif a.args.scheduler == "step":
                scheduler = StepLR(optimizer,step_size=a.args.step_size,gamma=a.args.step_gamma)
        
            mdl.to(c.device)   
            
            k = 0 # track total mini batches
            val_mb_iter = 0
            train_mb_iter = 0
            best_loss = float('inf')
            j = 0 # track total sub epochs
            l = 0 # track meta epochs
            
            for meta_epoch in range(a.args.meta_epochs):
                
                mdl.train()
                
                for sub_epoch in range(a.args.sub_epochs):
                    
                    train_loss = list()
                    
                    for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                        
                        optimizer.zero_grad()
                        
                        if a.args.data == 'dlr':
                            images,dmaps,counts,point_maps = data
                        else:
                            images,dmaps,labels,annotations,binary_labels,point_maps  = data 
                        
                        images = images.float().to(c.device)
                        dmaps = dmaps.float().to(c.device)
                        
                        input_data = (images,dmaps*a.args.dmap_scaling) # inputs features,dmaps 
                        z, log_det_jac = mdl(*input_data)

                        dims = tuple(range(1, len(z.size())))
                        loss = get_loss(z, log_det_jac,dims) # mdl.nf.jacobian(run_forward=False)
                        k += 1
                        train_mb_iter += 1
                        
                        loss_t = t2np(loss)
                        
                        if writer != None:
                            writer.add_scalar('loss/minibatch_train',loss, train_mb_iter)
                        
                        train_loss.append(loss_t)
                        loss.backward()
                        clip_grad_value_(mdl.parameters(), c.clip_value)
                        optimizer.step()
                        
                        if a.args.scheduler != "none":
                            scheduler.step()
                    
                    mean_train_loss = np.mean(train_loss)
                    
                    if writer != None:
                        writer.add_scalar('loss/epoch_train',mean_train_loss, meta_epoch)
                    
                    ### Validation Loop ------
                    if c.validation: 
                        val_loss = list()
                        val_z = list()
                        
                        with torch.no_grad():
                            
                            for i, data in enumerate(tqdm(val_loader, disable=c.hide_tqdm_bar)):
                                
                                if a.args.data == 'dlr':
                                    images,dmaps,counts,point_maps = data
                                else:
                                    images,dmaps,labels,annotations,binary_labels,point_maps  = data 
                            
                                z, log_det_jac = mdl(*input_data)
                                    
                                val_z.append(z)
                                    
                                dims = tuple(range(1, len(z.size())))
                                loss = get_loss(z, log_det_jac,dims)
                                k += 1
                                val_mb_iter += 1
                                val_loss.append(t2np(loss))
                                
                                if writer != None:
                                    writer.add_scalar('loss/minibatch_val',loss, val_mb_iter)
                             
                        mean_val_loss = np.mean(np.array(val_loss))
                         
                        print('Validation | Sub Epoch: {:d} \t Epoch loss: {:4f}'.format(sub_epoch,mean_val_loss))
                        
                        if writer != None:
                            writer.add_scalar('loss/epoch_val',mean_val_loss, meta_epoch)
                        
                        if mean_val_loss < best_loss and c.save_model:
                            best_loss = mean_val_loss
                            save_model(mdl,"best"+"_"+modelname)
                            
                        j += 1
                
                #### Meta epoch code (metrics) ------
                if c.viz:
                    plot_preds(mdl,train_loader)
                
                # train classification head to filter out null patches
                if not a.args.data == 'dlr' and not c_head_trained:
                    c_head_trained = True
                    mdl.classification_head = train_classification_head(mdl,head_train_loader,head_val_loader)
                
                if c.save_model and c.checkpoints:
                    mdl.to('cpu')
                    save_model(mdl,"checkpoint_"+str(l)+"_"+modelname)
                    #save_weights(model,"checkpoint_"+str(l)+"_"+modelname) # currently have no use for saving weights
                    mdl.to(c.device)
                
                model_metric_dict = {}
                
                if writer != None:
                    # DMAP Count Metrics - y,y_n,y_hat_n,y_hat_n_dists,y_hat_coords
                    # add images to TB writer
                    if c.viz:
                        plot_preds(mdl,train_loader,writer=writer,writer_epoch=meta_epoch,writer_mode='train')
                        
                    train_metric_dict = dmap_metrics(mdl, train_loader,n=c.eval_n,mode='train')
                    model_metric_dict.update(train_metric_dict)
                    
                    if c.viz and mdl.dlr_acd:
                        plot_preds(mdl,train_loader,writer=writer,writer_epoch=meta_epoch,writer_mode='train')
                    
                    if c.validation:
                        if c.viz:
                            plot_preds(mdl,val_loader,writer=writer,writer_epoch=meta_epoch,writer_mode='val')
                            
                        val_metric_dict = dmap_metrics(mdl, val_loader,n=c.eval_n,mode='val')
                        model_metric_dict.update(val_metric_dict)               
                
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
            
            mdl.hparam_dict = model_hparam_dict
            mdl.metric_dict = model_metric_dict
            
            if c.viz:
                
                if mdl.dlr_acd:
                    preds_loader = train_loader
                else:
                    preds_loader = val_loader
                    
                plot_preds(mdl, preds_loader, plot = True)
                # TODO skip PR curve, as this funciton takes 20 minutes
                # dmap_pr_curve(mdl, preds_loader,n = 10,mode = dmap_pr_mode)
            
            if c.save_model and not c.checkpoints:
                
                filename = "./models/"+"final"+modelname+".txt"
                
                with open(filename, 'w') as f:
                    print(model_hparam_dict, file=f)
                
                mdl.to('cpu')
                save_model(mdl,"final_"+modelname)
                mdl.to(c.device)
                
            if not a.args.data == 'dlr':
                print("Performing final evaluation with trained null classifier...")
                final_metrics = dmap_metrics(mdl, train_loader,n=1,mode='train',null_filter = True)
                print(final_metrics)
            
            run_end = time.perf_counter()
            print("Finished Model: ",modelname)
            print("Run finished. Time Elapsed (mins): ",round((run_end-run_start)/60,2),"| Datetime:",str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

            return mdl

def train_classification_head(mdl,full_trainloader,full_valloader,criterion = nn.CrossEntropyLoss()):

    print("Optimizing classification head for null filtering...")
    now = datetime.now() 
    filename = "_".join([
                c.feat_extractor,
                'FTE',str(c.feat_extractor_epochs),
                str(now.strftime("%d_%m_%Y_%H_%M_%S")),
                "PT",str(c.pretrained),
                'BS',str(full_trainloader.batch_size),
                'classification_head'
            ])
    
    writer = SummaryWriter(log_dir='runs/feat_pretrained/'+filename)
    optimizer = torch.optim.Adam(mdl.classification_head.parameters(),lr=1e-3,betas=(0.9,0.999),eps=1e-08, weight_decay=0)
    best_model_wts = copy.deepcopy(mdl.classification_head.state_dict())
    best_acc = 0.0
    
    t_sz = len(full_trainloader)*full_trainloader.batch_size
    v_sz = len(full_valloader)*full_valloader.batch_size
    minibatch_count = 0 
    
    dataset_sizes = {'train': t_sz,'val': v_sz}
   
    mdl.to(c.device)        
    
    for epoch in range(c.feat_extractor_epochs):
        
        for phase in ['train', 'val']:
            if phase == 'train':
                mdl.classification_head.train()  
                loader = full_trainloader
            else:
                mdl.classification_head.eval()
                loader = full_valloader
            
            print('Classification Head {} Epoch {}/{}'.format(phase,epoch, c.feat_extractor_epochs)) 
            running_loss = 0.0; running_corrects = 0
            
            for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
                
                optimizer.zero_grad()
                images = data[0].to(c.device)
                binary_labels = data[3]
                
                with torch.set_grad_enabled(phase == 'train'):
                    
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
        
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)) 
            
            if writer != None:
                writer.add_scalar('acc/epoch_{}'.format(phase),epoch_acc, epoch) # TODO
                writer.add_scalar('loss/epoch_{}'.format(phase),epoch_loss, epoch)
            
            running_loss = 0.0; running_corrects = 0
        
            # deep copy the model with best val acc
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(mdl.classification_head.state_dict())
        
    mdl.classification_head.load_state_dict(best_model_wts)
    save_model(mdl.classification_head,filename=filename,loc=g.FEAT_MOD_DIR)
    
    return mdl.classification_head
        
# from pytorch tutorial...           
def train_feat_extractor(feat_extractor,trainloader,valloader,criterion = nn.CrossEntropyLoss()):
    
    now = datetime.now() 
    filename = "_".join([
                c.feat_extractor,
                'FTE',str(c.feat_extractor_epochs),
                str(now.strftime("%d_%m_%Y_%H_%M_%S")),
                "PT",str(c.pretrained),
                'BS',str(trainloader.batch_size)
            ])
    
    writer = SummaryWriter(log_dir='runs/feat_pretrained/'+filename)
    
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
            
            print('Feature Extractor {} Epoch {}/{}'.format(phase,epoch, c.feat_extractor_epochs)) 
                
            running_loss = 0.0; running_corrects = 0
            
            for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
                
                optimizer.zero_grad()
                
                images = data[0].to(c.device)
                binary_labels = data[3].to(c.device)
                
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
                      
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)) 
            
            if writer != None:
                writer.add_scalar('acc/epoch_{}'.format(phase),epoch_acc, epoch) # TODO
                writer.add_scalar('loss/epoch_{}'.format(phase),epoch_loss, epoch)
            
            running_loss = 0.0; running_corrects = 0
            
            # deep copy the model with best val acc
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(feat_extractor.state_dict())
                
    print("Finetuning finished.")
        
    feat_extractor.load_state_dict(best_model_wts)
    save_model(feat_extractor,filename=filename,loc=g.FEAT_MOD_DIR)
    
    return feat_extractor