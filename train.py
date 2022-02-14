import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm # progress bar

import time 
import copy
import os
from datetime import datetime 

from eval import eval_mnist, dmap_metrics, dmap_pr_curve
from utils import get_loss, plot_preds, counts_preds_vs_actual, t2np, torch_r2
import model # importing entire file fixes 'cyclical import' issues
#from model import CowFlow, MNISTFlow, select_feat_extractor, save_model #, save_weights

import config as c
import gvars as g
import arguments as a

# tensorboard
from torch.utils.tensorboard import SummaryWriter

def train_battery(train_loader,val_loader,lr_i = c.lr_init):
        assert c.train_model
        print("Starting battery: ",str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        t1 = time.perf_counter()
        
    
        if len(lr_i) >= 1 or type(train_loader) == list and type(val_loader) == list:
            
            j = 0
            
            for tl, vl in zip(train_loader,val_loader):
            
                for param in lr_i:
                    
                    j = j+1
                    
                    if  c.tb:
                        battery_string = 'battery_'+str(j)
                        bt_id = 'runs/'+c.schema+'/'+battery_string
                        print('beginning: {}\n'.format(bt_id))
                        writer = SummaryWriter(log_dir=bt_id)
                    else:
                        writer = None
                        
                    train(train_loader=tl,
                          val_loader=vl,
                          battery = True,
                          lr_i = [param],
                          writer = writer)
            
        else:
            ValueError("lr_i must have more than one value, or the dataloaders must be supplied as lists")
            
        t2 = time.perf_counter()
        print("Battery finished. Time Elapsed (hours): ",round((t2-t1) / 60*60 ,2),"| Datetime:",str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
                

def train(train_loader,val_loader,head_train_loader=None,head_val_loader=None,battery = False,lr_i=c.lr_init,writer=None): #def train(train_loader, test_loader):
            assert c.train_model
            if c.debug:
                torch.autograd.set_detect_anomaly(True)
    
            if c.verbose: 
                print("Training run using {} train samples and {} valid samples...".format(str(len(train_loader)*int(train_loader.batch_size)),str(len(val_loader)*int(val_loader.batch_size))))
                print("Using device: {}".format(c.device))
                     
            now = datetime.now() 
            
            parts = [c.schema,
                     os.uname().nodename,
                    "BS"+str(train_loader.batch_size),
                    "LR_I"+str(lr_i),
                    "NC"+str(c.n_coupling_blocks),
                    "E"+str(c.meta_epochs*c.sub_epochs),
                    "FE",str(c.feat_extractor),
                    "DIM"+str(c.density_map_h)]
            
            #"LRS",str(c.scheduler), # only one LR scheduler currently
            if a.args.dlr_acd:
                parts.append('DLRACD')
            
            if c.mnist:
                parts.append('MNIST')
            
            if c.joint_optim:
                parts.append('JO')
                
            if c.pretrained:
                parts.append('PT')
                
            if c.pyramid:
                parts.append('PY_{}'.format(c.n_pyramid_blocks))
                
            if c.counts and not c.mnist:
                parts.append('CT')
            
            if c.fixed1x1conv:
                parts.append('1x1')
                
            if c.scale != 1:
                parts.append('SC_{}'.format(c.scale))
                
            if c.dropout_p != 0:
                parts.append('DP_{}'.format(c.dropout_p))
                
            if c.weight_decay != 1e-5:
                parts.extend(["WD",str(c.weight_decay)])
                
            if c.train_feat_extractor or c.load_feat_extractor_str != '':
                parts.append('FT')
                
            if c.sigma != 4 and not c.mnist:
                parts.extend(["FSG",str(c.sigma)])
                
            if c.clamp_alpha != 1.9 and not c.mnist:
                parts.extend(["CLA",str(c.clamp_alpha)])
                
            if c.test_train_split != 70 and not c.mnist:
                parts.extend(["SPLIT",str(c.test_train_split)])
                
            if c.balanced and not c.mnist:
                parts.append('BL')
                   
            parts.append(str(now.strftime("%d_%m_%Y_%H_%M_%S")))
            
            modelname = "_".join(parts)
    
            print("Training Model: ",modelname)
    
            model_hparam_dict = {'schema':c.schema,
                                 'learning rate init.':lr_i[0],
                                'batch size':val_loader.batch_size,
                                'image height':c.density_map_h,
                                'image width':c.density_map_w,
                                'joint optimisation?':c.joint_optim,
                                'global average pooling?':c.gap,
                                'scale:':c.scale,
                                'annotations only?':c.annotations_only,
                                'pretrained?':c.pretrained,
                                'feature pyramid?':c.pyramid,
                                'feature extractor?':c.feat_extractor,
                                '1x1convs?':c.fixed1x1conv,
                                'conv filters':c.filters,
                                'fc_width':c.width,
                                'finetuned?':c.train_feat_extractor,
                                'mnist?':c.mnist,
                                'counts?':c.counts,
                                'n pyramid blocks?':c.n_pyramid_blocks,
                                'subnet_type?':c.subnet_type,
                                'prop. of data':c.data_prop,
                                'clamp alpha':c.clamp_alpha,
                                'weight decay':c.weight_decay,
                                'epochs':c.meta_epochs*c.sub_epochs,
                                'no. of coupling blocks':c.n_coupling_blocks,
                                'filter sigma':c.sigma,
                                'feat vec length':c.n_feat}
            
            run_start = time.perf_counter()
            print("Starting run: ",str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    
            if c.verbose:
                print("Training using {} train samples and {} val samples...".format(len(train_loader)*train_loader.batch_size,
                                                                                            len(val_loader)*val_loader.batch_size))
    
            if not battery and c.tb:
                writer = SummaryWriter(log_dir='runs/'+c.schema+'/'+modelname)
            else:
                writer = writer
            
            feat_extractor = model.select_feat_extractor(c.feat_extractor,train_loader,val_loader)
            
            if c.mnist:
                mdl = model.MNISTFlow(modelname=modelname,feat_extractor = feat_extractor)    
            else:
                mdl = model.CowFlow(modelname=modelname,feat_extractor = feat_extractor)
                    
            if c.joint_optim:
                optimizer = torch.optim.Adam([
                            {'params': mdl.nf.parameters()},
                            {'params': mdl.feat_extractor.parameters(), 'lr_init': 1e-3,'betas':(0.9,0.999),'eps':1e-08, 'weight_decay':0}
                        ], lr=lr_i[0], betas=(0.8, 0.8), eps=1e-04, weight_decay=c.weight_decay )
            else:
                optimizer = torch.optim.Adam(mdl.nf.parameters(), lr=lr_i[0], betas=(0.9, 0.999), eps=1e-04, weight_decay=c.weight_decay)
            
            
            # add scheduler to improve stability further into training
            if c.scheduler == "exponential":
                scheduler = ExponentialLR(optimizer, gamma=0.9)
        
            mdl.to(c.device)   
            
            k = 0 # track total mini batches
            val_mb_iter = 0
            train_mb_iter = 0
            j = 0 # track total sub epochs
            l = 0 # track meta epochs
            
            for meta_epoch in range(c.meta_epochs):
                
                mdl.train()
                
                for sub_epoch in range(c.sub_epochs):
                    
                    if c.verbose:
                        t_e1 = time.perf_counter()
                    
                    train_loss = list()
                    
                    if c.debug and c.scheduler != 'none':
                        print('Initial Scheduler Learning Rate: ',scheduler.get_lr()[0])
                    
                    for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                        
                        t1 = time.perf_counter()
                        
                        optimizer.zero_grad()
                        
                        # TODO - can this section
                        if a.args.dlr_acd:
                            images,dmaps,counts,point_maps = data
                        elif not c.mnist and not c.counts and not train_loader.dataset.classification:
                            images,dmaps,labels, _ = data # _ annotations
                        elif not c.mnist and not c.counts:
                            images,dmaps,labels, _, _ = data # _ = annotations
                        elif not c.mnist:
                            images,dmaps,labels,counts = data
                        else:
                            images,labels = data
                        
                        if c.debug and not a.args.dlr_acd:
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
                            
                        z, log_det_jac = mdl(*input_data)
                            
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
                        # note: probably going to get an error with mnist or counts # TODO
                        dims = tuple(range(1, len(z.size())))
                        loss = get_loss(z, log_det_jac,dims) # mdl.nf.jacobian(run_forward=False)
                        k += 1
                        train_mb_iter += 1
                        
                        loss_t = t2np(loss)
                        
                        if c.debug:
                            print('loss/minibatch_train',train_mb_iter)
                        
                        if writer != None:
                            writer.add_scalar('loss/minibatch_train',loss, train_mb_iter)
                        
                        train_loss.append(loss_t)
                        loss.backward()
                        clip_grad_value_(mdl.parameters(), c.clip_value)
                        optimizer.step()
                        
                        if c.scheduler != "none":
                            scheduler.step()
                        
                        t2 = time.perf_counter()
                        
                        total_iter = len(train_loader) * c.sub_epochs * c.meta_epochs + (len(val_loader) * c.meta_epochs)
                        
                        if k % c.report_freq == 0 and c.verbose and k != 0 and c.report_freq != -1:
                                print('\nTrain | Mini-Batch Time: {:.1f}, Mini-Batch: {:d}, Mini-Batch loss: {:.4f}'.format(t2-t1,i+1, loss_t))
                                print('{:d} Mini-Batches in sub-epoch remaining'.format(len(train_loader)-i))
                                print('Total Iterations: ',total_iter,'| Passed Iterations: ',k)
                                
                        if c.debug and not c.mnist:
                            print("number of elements in density maps list:")
                            print(len(dmaps)) # images
                            print("number of images in image tensor:")
                            print(len(images)) # features                
                    
                    mean_train_loss = np.mean(train_loss)
                    
                    if c.verbose:
                        t_e2 = time.perf_counter()
                        print("\nTrain | Sub Epoch Time (s): {:f}, Epoch loss: {:.4f}".format(t_e2-t_e1,mean_train_loss))
                        print('Meta Epoch: {:d}, Sub Epoch: {:d}, | Epoch {:d} out of {:d} Total Epochs'.format(meta_epoch, sub_epoch,meta_epoch*c.sub_epochs + sub_epoch+1,c.meta_epochs*c.sub_epochs))
                    
                    if c.debug:
                        print("loss/epoch_train:",j)
                    
                    if writer != None:
                        writer.add_scalar('loss/epoch_train',mean_train_loss, j)
                    
                    if c.validation and not a.args.dlr_acd: 
                        val_loss = list()
                        val_z = list()
                        # val_coords = list() # todo
                        
                        with torch.no_grad():
                            
                            for i, data in enumerate(tqdm(val_loader, disable=c.hide_tqdm_bar)):
                                
                                # Validation
                                # TODO - DRY violation...
                                if not c.mnist and not c.counts:
                                    input_data = (images,dmaps) # inputs features,dmaps
                                elif not c.mnist:
                                    input_data = (images,counts) 
                                else:
                                    input_data = (images,labels)
                            
                                z, log_det_jac = mdl(*input_data)
                                    
                                val_z.append(z)
                                
                                if i % c.report_freq == 0 and c.debug and not c.mnist and not c.counts:
                                        print('val true count: {:f}'.format(dmaps.sum()))
                                    
                                dims = tuple(range(1, len(z.size())))
                                loss = get_loss(z, log_det_jac,dims)
                                k += 1
                                val_mb_iter += 1
                                val_loss.append(t2np(loss))
                                
                                if c.debug:
                                    print('loss/minibatch_val',val_mb_iter)
                                
                                if writer != None:
                                    writer.add_scalar('loss/minibatch_val',loss, val_mb_iter)
                             
                        mean_val_loss = np.mean(np.array(val_loss))
                         
                        if c.verbose:
                                print('Validation | Sub Epoch: {:d} \t Epoch loss: {:4f}'.format(sub_epoch,mean_val_loss))
                        
                        if c.debug:
                            print('loss/epoch_val: ',j)
                        
                        if writer != None:
                            writer.add_scalar('loss/epoch_val',mean_val_loss, j)
                            
                        j += 1
                
                #### Meta epoch code (metrics) ------
                if mdl.mnist:
                    val_acc, train_acc = eval_mnist(mdl,val_loader,train_loader)
                    print("\n")
                    print("Training Accuracy: ", train_acc,"| Epoch: ",j)
                    print("val Accuracy: ",val_acc,"| Epoch: ",j)
                
                if (c.save_model or battery) and c.checkpoints:
                    mdl.to('cpu')
                    model.save_model(mdl,"checkpoint_"+str(l)+"_"+modelname)
                    #save_weights(model,"checkpoint_"+str(l)+"_"+modelname) # currently have no use for saving weights
                    mdl.to(c.device)
                
                model_metric_dict = {}
                
                if writer != None:
                    # DMAP Count Metrics - y,y_n,y_hat_n,y_hat_n_dists,y_hat_coords
                    # add images to TB writer
                    if c.viz:
                        plot_preds(mdl,train_loader,writer=writer,writer_epoch=j,writer_mode='train')
                        
                    train_metric_dict = dmap_metrics(mdl, train_loader,n=1,mode='train')
                    model_metric_dict.update(train_metric_dict)
                    
                    if c.viz and mdl.dlr_acd:
                        plot_preds(mdl,train_loader,writer=writer,writer_epoch=j,writer_mode='train')
                    
                    if c.validation:
                        if c.viz:
                            plot_preds(mdl,val_loader,writer=writer,writer_epoch=j,writer_mode='val')
                            
                        val_metric_dict = dmap_metrics(mdl, val_loader,n=1,mode='val')
                        model_metric_dict.update(val_metric_dict)

                # MNIST Model metrics
                if writer != None and mdl.mnist:
                    writer.add_scalar('acc/meta_epoch_train',train_acc, j)
                    model_metric_dict['acc/meta_epoch_train'] = train_acc
                    
                    writer.add_scalar('acc/meta_epoch_val',val_acc, j)
                    model_metric_dict['acc/meta_epoch_val'] = val_acc
                
                # Count Model Metrics
                if writer != None and mdl.count:
                    train_R2 = torch_r2(mdl,train_loader)
                    writer.add_scalar('R2/meta_epoch_train',train_R2, j)
                    model_metric_dict['R2/meta_epoch_train'] = train_R2
                    
                    if c.verbose:
                        print("Train R2: ",train_R2)
                    
                    # TODO (MAPE or RMSE)
                    #writer.add_scalar('acc/meta_epoch_val',val_acc, l)
                    #model_metric_dict['acc/meta_epoch_val'] = val_acc
                    
                    if c.validation:
                        val_R2 = torch_r2(mdl,val_loader)
                        writer.add_scalar('R2/meta_epoch_val',val_R2, j)
                        model_metric_dict['R2/meta_epoch_val'] = val_R2
                        
                        if c.verbose:
                            print("Val R2: ",val_R2)
                
                # train classification head to filter out null patches
                if not a.args.dlr_acd and not mdl.mnist:
                    mdl.classification_head = train_classification_head(mdl,head_train_loader,head_val_loader)
                
                # add param tensorboard scalars
                if writer != None:
                    
                    for name,value in model_metric_dict.items():
                        writer.add_scalar(tag=name, scalar_value=value,global_step=j)
                    
                    #writer.add_hparams(
                    #           hparam_dict = model_hparam_dict,
                    #           metric_dict = model_metric_dict,
                    #           run_name = "epoch_{}".format(j)
                    #           )
                    
                    writer.flush()
                
                l += 1
            
            ### Post-Training ---
            
            # visualise a random reconstruction
            if c.viz:
                
                if mdl.dlr_acd:
                    preds_loader = train_loader
                    dmap_pr_mode = 'train'
                else:
                    preds_loader = val_loader
                    dmap_pr_mode = 'val'
                    
                plot_preds(mdl, preds_loader, plot = True)
                # TODO skip PR curve, as this funciton takes 20 minutes
                # dmap_pr_curve(mdl, preds_loader,n = 10,mode = dmap_pr_mode)
                
                if c.counts:
                    print("Plotting Train R2")
                    counts_preds_vs_actual(mdl,train_loader,plot=c.viz)
            
            # save final model, unless models are being saved at end of every meta peoch
            if c.save_model and not c.checkpoints:
                
                filename = "./models/"+"final_"+modelname+".txt"
                
                # could switch to using json and print params on model reload
                with open(filename, 'w') as f:
                    print(model_hparam_dict, file=f)
                
                mdl.to('cpu')
                model.save_model(mdl,"final_"+modelname)
                #model.save_weights(model, modelname) # currently have no use for saving weights
                mdl.to(c.device)
            
            
            if not a.args.dlr_acd:
                print("Performing final evaluation with trained null classifier...")
                final_metrics = dmap_metrics(mdl, train_loader,n=1,mode='train',null_filter = True)
                print(final_metrics)
            
            run_end = time.perf_counter()
            print("Finished Model: ",modelname)
            print("Run finished. Time Elapsed (mins): ",round((run_end-run_start)/60,2),"| Datetime:",str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
            if not battery:
                return mdl

# train classification head to predict empty patches
def train_classification_head(mdl,full_trainloader,full_valloader,criterion = nn.CrossEntropyLoss()):
    assert c.train_model
    if c.verbose:
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
    
    if not c.debug:
        writer = SummaryWriter(log_dir='runs/feat_pretrained/'+filename)
    else:
        writer = None
    
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
            
            if c.verbose:
                print('Classification Head {} Epoch {}/{}'.format(phase,epoch, c.feat_extractor_epochs)) 
                
            running_loss = 0.0; running_corrects = 0
            
            for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
                
                optimizer.zero_grad()
                
                images = data[0].to(c.device)
                binary_labels = data[3]
                
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
                best_model_wts = copy.deepcopy(mdl.classification_head.state_dict())
                
    if c.verbose:
        print("Finetuning finished.")
        
    mdl.classification_head.load_state_dict(best_model_wts)
    model.save_model(mdl.classification_head,filename=filename,loc=g.FEAT_MOD_DIR)
    
    return mdl.classification_head
        
# from pytorch tutorial...           
def train_feat_extractor(feat_extractor,trainloader,valloader,criterion = nn.CrossEntropyLoss()):
    assert c.train_model
    if c.verbose:
        print("Finetuning feature extractor...")
    
    now = datetime.now() 
    filename = "_".join([
                c.feat_extractor,
                'FTE',str(c.feat_extractor_epochs),
                str(now.strftime("%d_%m_%Y_%H_%M_%S")),
                "PT",str(c.pretrained),
                'BS',str(trainloader.batch_size)
            ])
    
    if not c.debug:
        writer = SummaryWriter(log_dir='runs/feat_pretrained/'+filename)
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
                
                optimizer.zero_grad()
                
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
    model.save_model(feat_extractor,filename=filename,loc=g.FEAT_MOD_DIR)
    return feat_extractor