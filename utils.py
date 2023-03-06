# MIT License Matthew Skiffington 2022
# External
import torch
import dill # solve error when trying to pickle lambda function in FrEIA
import shutil
import os
import numpy as np
import pandas as pd
import random
import cv2
import rasterio

from torch import randn
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime 
from prettytable import PrettyTable
from pathlib import Path
from PIL import Image
from rasterio.plot import reshape_as_raster
from sklearn.preprocessing import StandardScaler
from lcfcn import lcfcn_loss
from skimage.feature import peak_local_max # 
from skimage import morphology as morph
from pathlib import Path
import scipy

import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl

# Internal
import config as c
import gvars as g
import arguments as a
import data_loader 
import model
import baselines

# TODO - shift below 5 util functions to utils
def save_cstate(cdir,modelname,config_file):
    ''' saves a snapshot of the config file before running and saving model '''
    if not os.path.exists(g.C_DIR):
        os.makedirs(g.C_DIR)
        
    base, extension = os.path.splitext(config_file)
    
    if c.verbose:
        'Config file copied to {}'.format(g.C_DIR)
    
    new_name = os.path.join(cdir, base+"_"+modelname+".txt")
    
    shutil.copy(config_file, new_name)

def save_model(mdl,filename,loc=g.MODEL_DIR):
    if not os.path.exists(loc):
        os.makedirs(loc)
    torch.save(mdl, os.path.join(loc,filename), pickle_module=dill)
    print("model {} saved to {}".format(filename,loc))
    save_cstate(cdir=g.C_DIR,modelname="",config_file="config.py")
    
def load_model(filename,loc=g.MODEL_DIR):
    
    path = os.path.join(loc, filename)
    mdl = torch.load(path, pickle_module=dill,map_location=torch.device(c.device))
    
    bc_path = os.path.join(g.FEAT_MOD_DIR, a.args.bin_classifier_path)
    
    if a.args.bin_classifier_path != '': # and a.args.model_name == 'NF':
        mdl.classification_head = torch.load(bc_path, pickle_module=dill)
    
    print("model {} loaded from {}".format(filename,loc))
    mdl.eval()
    
    return mdl
    
def save_weights(mdl,filename,loc=g.WEIGHT_DIR):
    if not os.path.exists(loc):
        os.makedirs(loc)
    torch.save(mdl.state_dict(),os.path.join(loc,'weights_'+filename), pickle_module=dill)
        
def load_weights(mdl, filename,loc=g.WEIGHT_DIR):
    path = os.path.join(loc, filename)
    mdl.load_state_dict(torch.load(path,pickle_module=dill)) 
    return mdl

def is_baseline(mdl):
    
    if str(type(mdl))=="<class 'baselines.UNet'>":
        return True
    if str(type(mdl))=="<class 'baselines.CSRNet'>":
        return True
    if str(type(mdl))=="<class 'baselines.LCFCN'>":
        return True
    if str(type(mdl))=="<class 'baselines.FCRN_A'>":
        return True
    if str(type(mdl))=="<class 'baselines.MCNN'>":
        return True
    if str(type(mdl))=="<class 'baselines.Res50'>":
        return True
    if str(type(mdl))=="<class 'baselines.VGG_density'>":
        return True
        
    return False

def n_feats_select(fe=a.args.feat_extractor):
    
    # "condition dim"
    if fe == '':
        n_feat = -99

    if fe == "alexnet":
        n_feat = 256 
    elif fe == "vgg16_bn":
        n_feat = 512 
    elif fe in ["resnet18","resnet50","resnet9"]:
        n_feat = 512
    elif fe == "none":
        if a.args.data == 'mnist':
            n_feat = 1 
        else:
            # conditioning on raw RGB image
             n_feat = 3
             
    return n_feat

def ft_dims_select(mdl=None,config={}):
    # bug here if invoked from nf_head()
       
    if config == {}:
       config['feat_extractor'] = a.args.feat_extractor 
    
    if mdl == None:
        fe = config['feat_extractor']
    else:
        fe = mdl.feat_extractor.__class__.__name__
    
    if c.downsampling:
        
        if a.args.data == 'dlr':
            ft_dims = 10,10#(mdl.density_map_h // 2**5,mdl.density_map_w // 2**5) # 10, 10
        elif fe in ['resnet18','ResNetPyramid',"VGGPyramid",'resnet9','resnet50'] or fe == 'Sequential': 
            if a.args.resize or a.args.rrc:
                ft_dims = (8,8)
            else:
                ft_dims = (19, 25) # (mdl.density_map_h // 2**5,mdl.density_map_w // 2**5)
        elif fe == 'vgg16_bn' or fe == 'VGG':
            ft_dims = (19,25) #(8,8) #(18,25)
        elif fe == 'alexnet' or fe == 'AlexNet':
            ft_dims = (8,8) #(17,24)
        elif fe == 'none' or fe == 'NothingNet':
            ft_dims = (600,800)
            
    else:
        ft_dims = (600,800)#(mdl.density_map_h//2,mdl.density_map_w//2) 
    
    # for i in range(100):
    #     print(ft_dims)
    
    return ft_dims

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None

# split numpy matrix into submatrices
# credit SO user Dat
# link: https://stackoverflow.com/questions/11105375/how-to-split-a-matrix-into-4-blocks-using-numpy/11105569
def np_split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

def get_loss(z, jac,dims):
    # in differnet, exponentiate over 'channel' dim (n_feat)  
    # here, we exponentiate over channel, height, width to produce single norm val per density map
    
    return torch.mean(0.5 * torch.sum(z ** 2, dim=dims) - jac) / z.shape[1]

# change default initialisation
# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)  

def add_plot_tb(writer,fig,writer_mode,writer_epoch):
    
    if writer != None and writer_mode != None and writer_epoch != None:
        writer.add_figure('{} Dmap Pred: epoch {}'.format(writer_mode,writer_epoch), fig)
        writer.close() 

# TODO - edit to include UNet seg and LCFCN, MCNN?
@torch.no_grad()
def plot_preds_multi(mode,loader,loader_86,model_path_dict=g.BEST_MODEL_PATH_DICT,sample_n=5,anno=a.args.sampler=='anno',n=10): 
    
    # https://matplotlib.org/3.5.0/tutorials/provisional/mosaic.html
    
    UNet = load_model(model_path_dict['UNet'])
    UNet_seg = load_model(model_path_dict['UNet_seg'])
    CSRNet = load_model(model_path_dict['CSRNet'])
    FCRN = load_model(model_path_dict['FCRN'])
    
    LCFCN = load_model(model_path_dict['LCFCN'])
    MCNN = load_model(model_path_dict['MCNN'])
    Res50 = load_model(model_path_dict['Res50'])
    VGG = load_model(model_path_dict['VGG'])
    NF = load_model(model_path_dict['NF'])
    NF_FRZ = load_model(model_path_dict['NF_FRZ'])

    titles256 = ['UNet(d)','FCRN','ResNet50','CSRNet','MCNN']
    titles86 = ['VGG','NF','NF Frozen']
    titlesSeg = ['UNet(s)','Seg. Map (256x256)','LC-FCN','Seg. Map (800x600)']
    titleCond = 'Conditioning Aerial Image'
    
    tuplesSeg = [(0,0),(0,1),(1,0),(1,1)]
    
    assert mode in ['train','val']
    
    lb_idx = random.randint(0,loader.batch_size-1)
    
    z = 0 # track whether plot is occuring as annotations present past n
    for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            
            if z==1:
                break
            
            images,dmaps,labels,binary_labels,annotations,point_maps = data_loader.preprocess_batch(data)
            
            if len(annotations[lb_idx]) == 0 or i<n:
                print('pass')
                continue
            
            if n in [350,275,250,125]:
                text_x, text_y = 0.95, 0.83
            else:
                text_x, text_y = 0.95, 0.01

            # create resized seg map for labels quad plot
            seg_map_rs = scipy.ndimage.filters.maximum_filter(point_maps[lb_idx].cpu().numpy(),size = (4,4))
            
            idx256 = 0 
            idx86 = 0  
            idxSeg = 0 
            
            fig256, ax256 = plt.subplots(1,6)
            fig256.set_size_inches(24*1,4*1)
            fig256.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,hspace=0.15,wspace=0.05)
            fig256.set_dpi(100)
            [axi.set_axis_off() for axi in ax256.ravel()] # turn off subplot axes

            cmap = mpl.cm.get_cmap("viridis")
            naxes = len(fig256.axes)
            
            for i, ax in enumerate(fig256.axes):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor(cmap(float(i)/(naxes-1)))
        
            fig86, ax86 = plt.subplots(1, 4)
            fig86.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,hspace=0.15,wspace=0.05)
            fig86.set_size_inches(16*1,4*1)
            fig86.set_dpi(100)
            [axi.set_axis_off() for axi in ax86.ravel()] # turn off subplot axes
            
            figSeg, axSeg = plt.subplots(2, 2)
            figSeg.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,hspace=0.15,wspace=0.05)
            figSeg.set_size_inches(8*1,8*1)
            figSeg.set_dpi(100)
            [axi.set_axis_off() for axi in axSeg.ravel()] # turn off subplot axes
            
            z = z+1

            for mdl in [UNet,FCRN,Res50,CSRNet,MCNN]:
                
                plot_dmap = dmaps[lb_idx]/mdl.dmap_scaling
                
                mdl.to(c.device)
                preds = mdl(images)
                preds = preds[lb_idx].permute(1,2,0).cpu().numpy()
                # todo - retrain models with noise attr and uncomment below
                constant = ((mdl.noise)/2)*plot_dmap.shape[0]*plot_dmap.shape[1]
                
                count = np.round(((preds.sum()/mdl.dmap_scaling)-constant),1)
                
                ax256[idx256].title.set_text('{}'.format(titles256[idx256]))#str(type(mdl))))
                ax256[idx256].text(text_x,text_y, 'Count: {:.1f}'.format(count),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax256[idx256].transAxes,
                    color='green', fontsize=15)
                ax256[idx256].imshow(preds)
                idx256+=1
        
            for mdl in [UNet_seg]:
                
                plot_dmap = dmaps[lb_idx]/mdl.dmap_scaling
                
                mdl.to(c.device)
                preds = mdl(images)
                preds = preds[lb_idx].permute(1,2,0).cpu().numpy()
                # todo - retrain models with noise attr and uncomment below
                constant = ((mdl.noise)/2)*plot_dmap.shape[0]*plot_dmap.shape[1]
                
                count = np.round(((preds.sum()/mdl.dmap_scaling)-constant),1)
                
                #unet(s) #if titles[j] == 'UNet(s)':
                count = len(peak_local_max(preds.squeeze(),min_distance=1,num_peaks=np.inf,threshold_abs=0,threshold_rel=0.5))
                
                axSeg[tuplesSeg[idxSeg]].title.set_text('{}'.format(titlesSeg[idxSeg]))#str(type(mdl))))
                axSeg[tuplesSeg[idxSeg]].text(text_x,text_y, 'Count: {:.1f}'.format(count),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=axSeg[tuplesSeg[idxSeg]].transAxes,
                    color='green', fontsize=15)
                axSeg[tuplesSeg[idxSeg]].imshow(preds)
                idxSeg+=1
            
            axSeg[tuplesSeg[idxSeg]].imshow(seg_map_rs)
            axSeg[tuplesSeg[idxSeg]].title.set_text(titlesSeg[idxSeg]); idxSeg+=1
                
    dmap_rs = dmaps[lb_idx]
    
    z = 0
    for i, data in enumerate(tqdm(loader_86, disable=c.hide_tqdm_bar)):        
    
        if z==1:
            break    
        
        images,dmaps,labels,binary_labels,annotations,point_maps = data_loader.preprocess_batch(data)

        if len(annotations[lb_idx]) == 0 or i<n:
            continue
        
        # create full sized seg map for labels quad plot
        seg_map_frs = scipy.ndimage.filters.maximum_filter(point_maps[lb_idx].cpu().numpy(),size = (4,4))
        
        dmap_frs = dmaps[lb_idx]
        z = z+1
        
        for mdl in [VGG]:
            
            mdl.to(c.device)
            preds = mdl(images)
            preds = preds[lb_idx].permute(1,2,0).cpu().numpy()
            
            # todo - retrain models with noise attr and uncomment below
            constant = ((mdl.noise)/2)*dmaps[lb_idx].shape[0]*dmaps[lb_idx].shape[1]
            
            count = np.round(((preds.sum()/mdl.dmap_scaling)-constant),1)
            
            ax86[idx86].title.set_text('{}'.format(titles86[idx86]))#str(type(mdl))))
            ax86[idx86].text(text_x,text_y, 'Count: {:.1f}'.format(count),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax86[idx86].transAxes,
                color='green', fontsize=15)
            ax86[idx86].imshow(preds); idx86+=1         
             
         
        # conditioning aerial image
        unnorm = data_loader.UnNormalize(mean=tuple(c.norm_mean),
                             std=tuple(c.norm_std))
        im = unnorm(images[lb_idx])
        im = im.permute(1,2,0).cpu().numpy()
                
        figCond, axCond = plt.subplots(1)
        figCond.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,hspace=0.15,wspace=0.05)
        figCond.set_size_inches(8*1,8*1)
        axCond.title.set_text(titleCond)
        axCond.imshow((im * 255).astype(np.uint8))
        figCond.set_dpi(100)
        axCond.set_axis_off()
        
        mdl = NF; mdl.to(c.device)
        
        for mdl in [NF,NF_FRZ]:
        
            x_list = []
            print(a.args.resize)
            
            for i in range(sample_n):
                                    
                in_channels = c.channels*4**5 # n_ds=5
                a.args.resize = False
                ft_dims = ft_dims_select(mdl)
                dummy_z = (randn(images.size()[0], in_channels,ft_dims[0],ft_dims[1])).to(c.device)
                dummy_z = dummy_z.float().to(c.device)
                x, log_det_jac = mdl(images,dummy_z,rev=True)
                
                # x_list.append(x)
                        
                # x_agg = torch.stack(x_list,dim=1)
                # x = x_agg.mean(dim=1) # take average of samples from models for mean reconstruction
                         
            dmap_rev_np = x[lb_idx].squeeze().cpu().detach().numpy()
            constant = ((mdl.noise)/2)*plot_dmap.shape[0]*plot_dmap.shape[1]
            count = np.round((dmap_rev_np.sum()-constant),1)
            
            ax86[idx86].title.set_text('{}'.format(titles86[idx86]))#str(type(mdl))))
            ax86[idx86].text(text_x,text_y, 'Count: {:.1f}'.format(count),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax86[idx86].transAxes,
                color='green', fontsize=15)
            ax86[idx86].imshow(dmap_rev_np); idx86+=1
                   
        #  DENSITY 256
        ax256[idx256].imshow(dmap_rs.cpu().numpy())
        ax256[idx256].title.set_text('Density (256x256)')
        
        count = np.round(dmap_rs.cpu().numpy().sum(),1)
        ax256[idx256].text(text_x,text_y, 'Count: {:.1f}'.format(count),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax256[idx256].transAxes,
            color='green', fontsize=15)
        
        #  DENSITY 800x600
        ax86[idx86].imshow(dmap_frs.cpu().numpy())
        ax86[idx86].title.set_text('Density (800x600)')

        count = np.round((dmap_frs.cpu().numpy().sum()),1)
        ax86[idx86].text(text_x,text_y, 'Count: {:.1f}'.format(count),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax86[idx86].transAxes,
            color='green', fontsize=15)
    
        for mdl in [LCFCN]:
            
                mdl.to(c.device)
                preds = mdl(images)
                probs = preds.sigmoid().cpu().numpy()
                pred_blobs = lcfcn_loss.get_blobs(probs=probs).squeeze()
                pred_points = lcfcn_loss.blobs2points(pred_blobs).squeeze()
                y_list, x_list = np.where(pred_points.squeeze())
                count = np.round((np.unique(pred_blobs)!=0).sum(),1)
                preds = preds.squeeze(0).sigmoid().permute(1,2,0).cpu().numpy() 
                axSeg[tuplesSeg[idxSeg]].title.set_text('{}'.format(titlesSeg[idxSeg]))#str(type(mdl))))
                axSeg[tuplesSeg[idxSeg]].text(text_x,text_y, 'Count: {:.1f}'.format(count),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=axSeg[tuplesSeg[idxSeg]].transAxes,
                    color='green', fontsize=15)
                axSeg[tuplesSeg[idxSeg]].imshow(preds)
                idxSeg+=1
                
        axSeg[tuplesSeg[idxSeg]].imshow(seg_map_frs)
        axSeg[tuplesSeg[idxSeg]].title.set_text(titlesSeg[idxSeg])
        
        # 256 (MCNN, CSRNet, UNet, FCRN, Res50, GT)
        fig256.savefig('/home/mks29/Desktop/multi_preds/four_fig/figure256_{}.png'.format(n))  
        plt.close(fig256) 
        
        # 86 models + GT (NF, NF FRZ, VGG, GT)
        fig86.savefig('/home/mks29/Desktop/multi_preds/four_fig/fig86_{}.png'.format(n))   
        plt.close(fig86) 
        
        # seg 256, 86 (LCFCN, UNet Seg) + GTs (2x2)
        figSeg.savefig('/home/mks29/Desktop/multi_preds/four_fig/figSeg_{}.png'.format(n))   
        plt.close(figSeg) 
        
        # conditioning aerial image (1)
        figCond.savefig('/home/mks29/Desktop/multi_preds/four_fig/figCond_{}.png'.format(n))   
        plt.close(figCond) 
            
        break
        
    return
    # TODO - show pred counts

@torch.no_grad()
def predict_image(mdl_path,nf=False,geo=True,nf_n=500,mdl_type='',plot_single=False,
                  image_path='/home/mks29/clones/cow_flow/data/obj/930WT-92NMT_800_2400.jpg',dlr=False):
    
    #'/home/mks29/6-6-2018_Ortho_ColorBalance.tif'
    #'/home/mks29/Desktop/BB34_5000_1006.jpg'
    
    assert mdl_type in g.BASELINE_MODEL_NAMES + ['NF']
    print('Patch sizes hardcoded to 800x600 (cow dataset)')
    
    mdl = load_model(mdl_path).to(c.device)
    
    # split image into 256x256 chunks
    im = cv2.imread(image_path)
    
    if plot_single:
        im_patches = [im]
    else:
        im_patches = split_image(im,save = False, overlap = 0,dlr=dlr)
    
    for i in tqdm(range(len(im_patches)),desc="Predicting patches"):
        
        # normalise image patches
        #raw_patch = im_patches[i] # debug
        patch = torch.from_numpy(im_patches[i]).float().to(c.device)
        
        # double check processing lines up w/data loader
        patch = patch.permute(2,0,1)
        patch = patch.float().div(255).to(c.device)
        patch = TF.normalize(patch,mean = c.norm_mean,std= c.norm_std)
        
        resize = T.Resize(size=(a.args.image_size,a.args.image_size))
        patch = resize(patch.unsqueeze(0))
        patch = patch.squeeze()
        
        if nf:
            
            x_list = []
        
            for j in range(nf_n):     
                
                in_channels = c.channels*4**5 # n_ds=5
                ft_dims = ft_dims_select(mdl)
                                
                dummy_z = (randn(1, in_channels,ft_dims[0],ft_dims[1])).to(c.device)
                dummy_z = dummy_z.float()
                
                x, log_det_jac = mdl(patch.unsqueeze(0),dummy_z,rev=True)
                
                if not dlr:
                    x = TF.resize(x, (mdl.density_map_h,mdl.density_map_w))
                    
                x_list.append(x)
                        
            x_agg = torch.stack(x_list,dim=1)
            x = x_agg.mean(dim=1) # take average of samples from models for mean reconstruction
                    
            # replace predicted densities with null predictions if not +ve pred from feature extractor
            # subtract constant for uniform noise
            
            if a.args.bin_classifier_path != '' :
                outputs = mdl.classification_head(patch.unsqueeze(0))
                _, preds = torch.max(outputs, 1)
                x[(preds == 1).bool(),:,:] = torch.zeros(1,mdl.density_map_h,mdl.density_map_w).to(c.device)
            
            # constant = ((mdl.noise)/2)*patch_size*patch_size
            im_patches[i] = x.squeeze(0) #.permute(1,2,0)
            #DEBUG - model seems to be working?
            # if i>0:
            #     fig, ax = plt.subplots(2)
            #     plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            #     fig.set_size_inches(35*1,7*1)
            #     fig.set_dpi(100)
            #     im_patches[i] = im_patches[i].permute(1,2,0)
            #     ax[0].imshow(im_patches[i].cpu().numpy()) 
            #     ax[1].imshow(raw_patch) 
            # if i == 300:
            #     pass
            
        else:
            im_patches[i] = mdl(patch.unsqueeze(0)).squeeze(0) / mdl.dmap_scaling #.astype(np.uint8)
            #DEBUG - model seems to be working?
    #         if i>0:
    #             fig, ax = plt.subplots(2)
    #             plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    #             fig.set_size_inches(35*1,7*1)
    #             fig.set_dpi(100)
    #             im_patches[i] = im_patches[i].permute(1,2,0)
    #             ax[0].imshow(im_patches[i].cpu().numpy()) 
    #             ax[1].imshow(raw_patch) 
    #         if i == 300:
    #             pass
    # 1/0
        
    for i in tqdm(range(len(im_patches)),desc="Interpolating"):
        
       if not dlr:
           im_patches[i] = F.interpolate(im_patches[i].unsqueeze(0),(608,800), mode='bicubic', align_corners=False)
           im_patches[i] = im_patches[i].squeeze(0).permute(1,2,0).cpu().numpy() # 
       else:
           im_patches[i] = F.interpolate(im_patches[i].unsqueeze(0),(320,320), mode='bicubic', align_corners=False)
           im_patches[i] = im_patches[i].squeeze(0).permute(1,2,0).cpu().numpy()
    
    shape = im.shape
    
    if not plot_single:
        del im
    
    path='/home/mks29/' ; frmt = 'tif' ; name = Path(image_path).stem
    
    if plot_single:
        
        fig, ax = plt.subplots(2)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        fig.set_size_inches(4*1,8*1)
        fig.set_dpi(100)
        ax[0].imshow(im) 
        ax[1].imshow(im_patches[i]) 
        
        print(np.sum(im_patches))
        
        plt.show()
        
        return
    
    predicted = stich_image(shape, im_patches, name=name,save=True,path=path,
                            frmt=frmt,geo=geo,mdl_type=mdl_type,dlr=dlr)
    
    if geo:
        
        x_scaler = StandardScaler()
        predicted[0,:,:] = x_scaler.fit_transform(predicted[0,:,:])
        
        del im_patches
        
        print('whole pred image max')
        print(np.max(predicted))
        
        src = rasterio.open(image_path)
        predicted = reshape_as_raster(predicted)

        # Register GDAL format drivers and configuration options with a
        # context manager.
        with rasterio.Env():

            # Write an array as a raster band to a new 8-bit file. For
            # the new file's profile, we start with the profile of the source
            profile = src.profile
            # And then change the band count to 1, set the
            # dtype to uint8, and specify LZW compression.
            profile.update(
                width=predicted.shape[2],
                height=predicted.shape[1],
                dtype=rasterio.float32,
                count=1,
                compress='lzw')
    
        with rasterio.open(path+mdl_type+name+"."+frmt, 'w', **profile) as dst:
            dst.write(predicted.astype(rasterio.float32))
            dst.close()
            print('{}{}{}"."{} saved to file.'.format(path,mdl_type,name,frmt))
    # else:
    #     cv2.imwrite(format(path,mdl_type,name,'jpg'),img=predicted)
        
    

@torch.no_grad()
def plot_preds_baselines(mdl, loader,mode="",mdl_type='',writer=None,writer_epoch=None,writer_mode=None):
        
        is_unet_seg = a.args.model_name == "UNet_seg"
        
        assert mode in ['train','val']
        assert mdl_type in g.BASELINE_MODEL_NAMES
        
        lb_idx = random.randint(0,loader.batch_size-1)
        
        # todo - fix this mdl.seg stuff
        for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            
                # if i < 300:
                #     continue
            
                mdl.to(c.device)
                images,dmaps,labels,binary_labels , annotations, point_maps  = data_loader.preprocess_batch(data)
                preds = mdl(images)
                
                dmap_np = preds[lb_idx].squeeze().cpu().detach().numpy()
                dmap_np = dmap_np/mdl.dmap_scaling
                pred_count = dmap_np.sum() 
                
                # if len(labels[0]) == 0:
                #     continue
                
                if str(type(mdl)) == "<class 'baselines.LCFCN'>":#  or mdl_type == 'UNet_seg':
                    
                        probs = preds.sigmoid().cpu().numpy()
                        
                        pred_blobs = lcfcn_loss.get_blobs(probs=probs).squeeze()

                        pred_points = lcfcn_loss.blobs2points(pred_blobs).squeeze()
                        y_list, x_list = np.where(pred_points.squeeze())
                        pred_counts = (np.unique(pred_blobs)!=0).sum()
                        preds = preds.squeeze(0).sigmoid().permute(1,2,0).cpu().numpy() # 
                else:
                    preds = preds[lb_idx].permute(1,2,0).cpu().numpy()
                    
                    if a.args.noise != 0:
                        ground_truth_point_map = point_maps[lb_idx].squeeze().cpu().detach().numpy()
                        gt_count = ground_truth_point_map.sum() #.round()
                        constant = ((mdl.noise)/2)*dmaps[lb_idx].shape[0]*dmaps[lb_idx].shape[1] 
                        loader_noise = ((a.args.noise)/2)*dmaps[lb_idx].shape[0]*dmaps[lb_idx].shape[1] 
                        pred_count -= constant
                        gt_count -= loader_noise
                
                n_peaks=max(1,int(pred_count))
                
                if str(type(mdl)) == "<class 'baselines.LCFCN'>": # or is_unet_seg:
                    coordinates = np.argwhere(pred_points != 0)
                
                elif is_unet_seg:
                    # threshold absolute = 0 only works if passed through sigmoid transform
                    # in which case may as well use relative arg
                    coordinates = peak_local_max(preds,min_distance=int(mdl.sigma//2),threshold_rel=0.5)
                    
                else:
                    coordinates = peak_local_max(preds,min_distance=int(mdl.sigma//2),num_peaks=n_peaks)
                    
                
                unnorm = data_loader.UnNormalize(mean=tuple(c.norm_mean),
                                     std=tuple(c.norm_std))
                
                plot_dmap = dmaps[lb_idx] #/mdl.dmap_scaling
                
                im = unnorm(images[lb_idx])
                im = im.permute(1,2,0).cpu().numpy()

                fig, ax = plt.subplots(1,5)
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
                fig.set_size_inches(24*1,7*1)
                fig.set_dpi(100)
                fig.suptitle('Baseline ({}) Test - {}'.format(mdl_type,mode),y=1.0,fontsize=16) # :.2f
                
                [axi.set_axis_off() for axi in ax.ravel()] # turn off subplot axes

                ax[0].title.set_text(a.args.title) # 'Density Map Prediction'
                ax[0].imshow(preds)
                ax[1].title.set_text('Conditioning Aerial Image')
                ax[1].imshow((im * 255).astype(np.uint8))
                ax[2].title.set_text('Ground Truth Density Map')
                ax[2].imshow(plot_dmap.cpu().numpy())
                
                if mdl.density_map_h == 608 and not a.args.rrc:
                    width = 800
                    height = 608
                else:
                    width = 256
                    height = 256
                
                gt_coords = annotations[lb_idx]
                
                if gt_coords.nelement() != 0:
                    gt_coords = torch.stack([gt_coords[:,2]*width,gt_coords[:,1]*height]).cpu().detach().numpy()
                    gt_coords = np.swapaxes(gt_coords,1,0)
                else:
                    gt_coords = None
                
                
                ax[3].title.set_text('Ground Truth Annotations')
                if gt_coords is not None:
                    ax[3].scatter(gt_coords[:,0], gt_coords[:,1],c='red',marker='1',label='Ground truth coordinates')
                else:
                    ax[3].scatter([],[],label='Ground truth coordinates')
                
                ax[4].title.set_text('Predicted Points')
                ax[4].scatter(coordinates[:,0], coordinates[:,1],label='Predicted coordinates')
                
                add_plot_tb(writer,fig,writer_mode,writer_epoch)
                
                print("\n Sum Pred Density Map: {} ".format(preds.sum()))
                print("Sum GT Density Map: {} ".format(plot_dmap.sum()))
                print("Label Count: {}".format(len(labels[lb_idx])))
                
                if str(type(mdl)) =="<class 'baselines.LCFCN'>":# or mdl_type == 'UNet_seg':
                        print("%s predicted (LCFCN)" % (pred_counts))
                
                if mdl_type == 'UNet_seg':
                        print("%s predicted (UNet_seg)" % (len(coordinates)))
                
                if hasattr(mdl, 'dmap_scaling'):
                    if mdl.dmap_scaling != 1:
                        print('mdl dmap scaling: {}'.format(mdl.dmap_scaling))
                
                plt.show()    
                
                return #preds, preds.sum(),plot_dmap.sum(),len(labels[lb_idx])
                
# TODO split into minst and non mnist funcs
@torch.no_grad()
def plot_preds(mdl, loader, plot = True, save=True,title = "",digit=None,
               hist=False,sampling="randn",plot_n=None,writer=None,writer_epoch=None,
               writer_mode=None,include_empty=False,sample_n=50,null_filter=False):
    
    assert type(loader) == torch.utils.data.dataloader.DataLoader
    
    """

    Parameters
    ----------
    model
        A saved model, either MNISTFlow or CowFlow
    loader
        A dataloader of samples.
    plot : Bool, optional
        Whether to return a plot or not. If not MNISt, will cycle through data loader until an item with an annotation is found.
    save : TYPE, optional
        DESCRIPTION. The default is True.
    title : TYPE, optional
        Plot title. The default is "".
    digit : TYPE, optional
        Predict a specific digit from MNIST. The default is None.
    hist : TYPE, optional
        Plot histogram of predictions. The default is True.
    sampling : TYPE, optional
        Three options: zeros, ones or randn (i.e. ~N(0,1) ). The default is "randn".
    digit: 
        If 'None' and plot_n == 'None', a random digits will be plotted, else the digit specified will be plotted
    plot_n:
        Plots n plots from the loader sequentially. If not MNIST, plots null-labelled data.
    include_empty:
        (True) Skip check for images with no annotations. Default: False.
    sample_n:
        Number of samples to average across 

    Returns
    -------
    dmap_rev_np:
        the samples used to generate the mean reconstruction.
    
    mean_pred:
        The mean reconstruction

    """
      
    if not mdl.mnist and digit != None:
        print('Digit argument ignored for non-MNIST models')
        
    if not mdl.mnist :
        if mdl.count != loader.dataset.count:
            raise ValueError("model and loader count properties do not match!")
    
    if mdl.mnist:
        if plot_n != None and digit != None:
            print("Warning: plot_n argument will be ignored")
        elif plot_n == None and digit == None:
            print("Plotting a random digit")
        elif plot_n != None:
            print("Plotting first {} digits".format(plot_n))
    
    def inner_func(mdl=mdl,hist=hist):
        
        if a.args.mode == 'train':
            plt.switch_backend('agg') # for tensorboardx
            
        idx = random.randint(0,len(loader)-1)
        
        k = 0
        # TODO: better var name
        z = 0 
        
        for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            
            # if i < 300:
            #     continue
            
            # TODO - broken - filename doesn't match image
    #            if not mdl.mnist:
    #                sample_fname = loader.dataset.train_im_paths[i]
            
            if mdl.mnist:
                images,labels = data
            elif mdl.dlr_acd:
                images,dmaps,counts,point_maps = data_loader.preprocess_batch(data,dlr=True)
            elif loader.dataset.count: # mdl.dataset.count
                images,dmaps,labels,counts, point_maps = data
            elif loader.dataset.classification:
                images,dmaps,labels,binary_labels , annotations, point_maps  = data_loader.preprocess_batch(data)
            else:
                images,dmaps,labels,_ = data
            
            # TODO - intermittent bug here
            dmaps = dmaps.cpu()
            
            lb_idx = random.randint(0,loader.batch_size-1)
                
            if mdl.count: #  or # TODO
    
                # check annotations in batch aren't empty
                lb_idx = None
                 
                for count in counts:
                    j = 0
                    z = z+1
                    if z >= len(loader)*loader.batch_size:
                        print("Loader has no labelled data!")
                    if count != 0:
                        lb_idx = j
                        break
                    else:
                        j = j + 1
                    
                if lb_idx == None:
                    continue  
            
            elif not mdl.mnist and not mdl.dlr_acd and not include_empty:
                
                 # check annotations in batch aren't empty
                lb_idx = None
                 
                j = 0
                for label in labels:
                    
                    z = z+1
                    if z >= len(loader)*loader.batch_size:
                        print("Loader has no labelled data!")
                    if len(label) != 0:
                        lb_idx = j
                        break
                    else:
                        j = j + 1
    
                if lb_idx == None:
                    continue  
            
            if i != idx and plot_n == None and digit == None and mdl.mnist:
                continue
           
            images = images.float().to(c.device)
            mdl = mdl.to(c.device)
            x_list = []
            
            for i in range(sample_n):
                
                ## create noise vector ---
                if (mdl.mnist and labels.size) or not mdl.mnist: # triggers only if there is at least one annotation
                    # Z shape: torch.Size([2, 4, 300, 400]) (batch size = 2)
                    
                    if not mdl.mnist:
                        # TODO - remove hard coding dims here
                        if (loader.dataset.count and not mdl.gap) or (not mdl.downsampling and mdl.subnet_type == 'conv'):
                            dummy_z = (randn(images.size()[0],mdl.channels*4,(mdl.density_map_h) // 2,(mdl.density_map_w) // 2)).to(c.device) 
                        elif loader.dataset.count and mdl.gap:
                            dummy_z = (randn(images.size()[0],1)).to(c.device) 
                        elif mdl.downsampling: # self.downsampling
                            
                            if mdl.scale == 1:
                                n_ds = 5
                            elif mdl.scale == 2:
                                n_ds = 4
                            elif mdl.scale == 4:
                                n_ds = 3
                                
                            in_channels = c.channels*4**n_ds
                            ft_dims = ft_dims_select(mdl)
                            
                            if mdl.feat_extractor.__class__.__name__ == 'NothingNet':
                                in_channels = 2                      
                            
                            dummy_z = (randn(images.size()[0], in_channels,ft_dims[0],ft_dims[1])).to(c.device)
                    else:
                            if mdl.subnet_type == 'conv':
                               
                                if sampling == 'ones':
                                    dummy_z = (torch.ones(images.size()[0], mdl.channels*4, mdl.density_map_h // 2,mdl.density_map_w  // 2).to(c.device))
                                elif sampling == "zeros":
                                    dummy_z = torch.zeros(images.size()[0], mdl.channels*4, mdl.density_map_h // 2,mdl.density_map_w  // 2).to(c.device)
                                elif sampling == "randn":
                                    dummy_z = (randn(images.size()[0], mdl.channels*4, mdl.density_map_h // 2,mdl.density_map_w  // 2)).to(c.device)
                                else:
                                    ValueError("Invalid function arg (sampling). Try 'randn', 'zeros' or 'ones'.")
                            else:
                                dummy_z = (randn(images.size()[0], c.channels)).to(c.device)
                
                ## sample from model -
                dummy_z = dummy_z.float().to(c.device)
                
                x, log_det_jac = mdl(images,dummy_z,rev=True)
                x_list.append(x)
                    
            x_agg = torch.stack(x_list,dim=1)
            x = x_agg.mean(dim=1) # take average of samples from models for mean reconstruction
                
            # replace predicted densities with null predictions if not +ve pred from feature extractor
            constant = ((mdl.noise)/2)*dmaps[lb_idx].shape[0]*dmaps[lb_idx].shape[1]
            loader_noise = ((a.args.noise)/2)*dmaps[lb_idx].shape[0]*dmaps[lb_idx].shape[1]

            if not mdl.mnist and not mdl.count and null_filter==True:
                # subtract constrant for uniform noise
                if not mdl.dlr_acd:
                    # subtract constrant for uniform noise
                    print('replacing predicted densities with empty predictions from feature extractor')
                    outputs = mdl.classification_head(images)  
                    _, preds = torch.max(outputs, 1)  
                    x[(preds == 1).bool(),:,:,:] = torch.zeros(1,mdl.density_map_h,mdl.density_map_w).to(c.device)
            
            if lb_idx == None:
                lb_idx = 0
            
            if c.one_hot:
                
                if a.args.subnet_type in g.SUBNETS:
                    x = x.argmax(-3).to(torch.float)
                else:
                    x = x.argmax(-1).to(torch.float)
            
            if  plot_n != None:
                lb_idx = range(images.size()[0])
            else:
                lb_idx = [lb_idx]
            
            for lb_idx in lb_idx:
                dmap_rev_np = x[lb_idx].squeeze().cpu().detach().numpy()
                mean_pred = x[lb_idx].mean()
                
                # TODO
                if (mdl.count or mdl.mnist) and not mdl.gap:
                    x_flat = torch.reshape(x,(images.size()[0],mdl.density_map_h * mdl.density_map_w)) # torch.Size([200, 12, 12])
                    mode = torch.mode(x_flat,dim = 1).values.cpu().detach().numpy()
                elif mdl.count and mdl.gap:
                    x_flat = x
                                        
                if mdl.mnist and mdl.subnet_type in g.SUBNETS:
                    x = torch.mode(x,dim = 1).values.cpu().detach().numpy() #print(x.shape) (200,)
                else:
                    x = torch.mode(x,dim = 0).values.cpu().detach().numpy()
                
                # TODO!
                if mdl.mnist or mdl.subnet_type in g.SUBNETS:
                    sum_pred = dmap_rev_np.sum()-constant #[lb_idx]
                    true_dmap_count = dmaps[lb_idx].sum()-loader_noise
                    if not mdl.dlr_acd:
                        label_count = len(annotations[lb_idx])
                    else:
                        label_count = counts[lb_idx]
                elif mdl.subnet_type == 'fc' and mdl.gap:  
                    mode = x
                    sum_pred = x
                
                if plot:
                    
                    if mdl.unconditional: 
                        hist = False
                    
                    n_plots = 3+hist-mdl.mnist-mdl.count
                           
                    fig, ax = plt.subplots(1,n_plots)
                    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
                    [axi.set_axis_off() for axi in ax.ravel()] # turn off subplot axes
                    
                    if mdl.mnist:
                        fig.suptitle('{} \n mode of reconstruction: {}'.format(title,str(mode)),y=1.0,fontsize=16) # :.2f
                    if mdl.count:   
                        fig.suptitle('{} \n Predicted count: {:.2f}\n True Labelled Count: {:.1f}'.format(title,mean_pred),y=1.0,fontsize=16)
                    else:
                        fig.suptitle('{} \n Predicted count: {:.2f} (-{} from noise adj.)'.format(title,sum_pred,constant),y=1.05,fontsize=16)
                        
                    fig.set_size_inches(24*1,7*1)
                    fig.set_dpi(100)
                    
                    if mdl.mnist:
                        im = images[lb_idx].squeeze().cpu().numpy()
                    else:
                        unnorm = data_loader.UnNormalize(mean =tuple(c.norm_mean),
                                             std=tuple(c.norm_std))
                        
                        if mdl.dlr_acd:
                            im = images[lb_idx]
                        else:
                            im = unnorm(images[lb_idx])
                        
                        im = im.permute(1,2,0).cpu().numpy()
                     
                    if mdl.feat_extractor.__class__.__name__ == 'NothingNet':
                        dmap_rev_np = dmap_rev_np[1,:,:] # select only data from first duplicated channel
                    
                    if not (mdl.count and mdl.gap) and not (mdl.mnist and a.args.subnet_type == 'fc'):
                        ax[0].title.set_text(a.args.title) # 'Density Map Prediction'
                        ax[0].imshow(dmap_rev_np)#, cmap='viridis', interpolation='nearest')
                    else:
                        fig.delaxes(ax[0])
                    
                    if mdl.count:
                        # no dmaps provided by dataloader if mdl.count
                        ax[1].title.set_text('Conditioning Aerial Image')
                        ax[1].imshow((im * 255).astype(np.uint8))
                        if hist:
                            ax[2].hist(dmap_rev_np.flatten(),bins = 30)
                    else:
                    
                        if mdl.mnist:
                            ax[1].title.set_text('Conditioning Digit')
                            ax[1].imshow(im)
                            if hist:
                                ax[2].title.set_text('Histogram of Reconstruction Values')
                                ax[2].hist(dmap_rev_np.flatten(),bins = 30)
                        else:
                            ax[1].title.set_text('Conditioning Aerial Image')
                            ax[1].imshow((im * 255).astype(np.uint8))
                            ax[2].title.set_text('Density Map Ground Truth: N={:.2f}\nLabel Count: N={:.2f}'.format(true_dmap_count,label_count))
                            ax[2].imshow(dmaps[lb_idx])
                            if hist:
                                ax[3].title.set_text('Histogram of Reconstruction Values')
                                ax[3].hist(dmap_rev_np.flatten(),bins = 30)
                                
                    add_plot_tb(writer,fig,writer_mode,writer_epoch)
                    
                    # saving and outs
                    plt.show()
                    
                    if save:
                        if not os.path.exists(g.VIZ_DIR):
                            os.makedirs(g.VIZ_DIR)
                        plt.savefig("{}/{}.jpg".format(g.VIZ_DIR,mdl.modelname), bbox_inches='tight', pad_inches = 0)
                    
                    if mdl.mnist:
                        out = labels[lb_idx],dmap_rev_np, mean_pred 
                    else:
                        if mdl.count:
                            out = counts[lb_idx],dmap_rev_np, mean_pred
                        else:
                            out = {'True Density Map count':true_dmap_count,
                                   'Predicted Density Map':dmap_rev_np, 
                                   'Predicted Density Map count':sum_pred,
                                   'Label Count':label_count}
                    
                    if digit != None and labels[idx] == digit:
                        return out 
                    
                    if plot_n != None:
                        k = k+1
                        if k >= plot_n:
                            lb_idx = lb_idx
                            return out
                    else:
                        return out
                                
    out = inner_func()       

    return out



def torch_r2(mdl,loader):
    """calcs r2 (on torch only)"""
    assert mdl.count
    assert loader.dataset.count
    assert not mdl.mnist
    assert mdl.modelname 
    
    mdl = mdl.to(c.device)
    actuals = []
    means = []
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar,desc="Calculating R2")):
            
            images,dmaps,labels,counts,point_maps  = data
            
            if not mdl.gap:  
                num = 0
                dummy_z = (randn(images.size()[0],mdl.channels*4,mdl.density_map_h // 2,mdl.density_map_w // 2, requires_grad=False,device=c.device))
            else:
                num = 1
                dummy_z = (randn(images.size()[0],mdl.channels,requires_grad=False,device=c.device))
            
            dummy_z = dummy_z.float()
            x, _ = mdl(images,dummy_z,rev=True)
            
            if mdl.subnet_type == 'conv':
                mean_preds = x.mean(dim = tuple(range(1,len(x.shape)-num)))
                means.append(mean_preds)
            else:
                means.append(x)
                 
            actuals.append(counts)
        
        y_hat = torch.cat(means,dim=0).squeeze()
        y = torch.cat(actuals)
        y_bar = y.mean()
        
        ss_res = torch.sum((y-y_hat)**2)
        ss_tot = torch.sum((y-y_bar)**2)
        
        r2 = 1 - (ss_res / ss_tot)
             
    return r2   

def counts_preds_vs_actual(mdl,loader,plot=False,ignore_zeros=False):
    """Plots predicted versus actual counts from the data and returns the R^2 value. Required: count dataloader and count model."""
    
    assert mdl.count
    assert loader.dataset.count
    assert not mdl.mnist
    assert mdl.modelname 
    
    if ignore_zeros:
        assert loader.batch_size == 1
    
    means = []
    actuals = []
    
    if plot:
        desc = "Plotting Counts vs Actuals"
    else:
        desc = "Calculating R2"
        
    for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar,desc=desc)):
        
        images,dmaps,labels,counts,point_maps  = data

        
        if not mdl.gap:  
            num = 0
            dummy_z = (randn(images.size()[0],mdl.channels*4,mdl.density_map_h // 2,mdl.density_map_w // 2, requires_grad=False)).to(c.device)
        else:
            num = 1
            dummy_z = (randn(images.size()[0],mdl.channels,requires_grad=False)).to(c.device)
        
        images = images.float().to(c.device)
        dummy_z = dummy_z.float().to(c.device)
        mdl = mdl.to(c.device)
        x, _ = mdl(images,dummy_z,rev=True)
        
        if mdl.subnet_type == 'conv':
            mean_preds = x.mean(dim = tuple(range(1,len(x.shape)-num))).detach().cpu().numpy()
            means.append(mean_preds)
        else:
             mean_preds = x.squeeze().detach().cpu().numpy().tolist()
             
             if not isinstance(mean_preds, list):
                means.append(mean_preds) 
             else:
                means.extend(mean_preds)
    
        actuals.append(counts.detach().cpu().numpy())
    
    if mdl.gap:
        means = [means]
    
    means = np.concatenate(means, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    correlation_matrix = np.corrcoef(means, actuals)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    
    # TODO - add model info to plot
    if plot:
        ident = [np.min(np.concatenate([means,actuals], axis=0)), np.max(np.concatenate([means,actuals], axis=0))]
        plt.scatter(means, actuals, alpha=0.5)
        plt.plot(ident,ident)
        plt.title('Predicted versus Actual Counts: R2 = {}'.format(str(round(r_squared,2))))
        plt.xlabel("Predicted Counts")
        plt.ylabel("Actual Counts")
        plt.show()
    
    return means,actuals,r_squared

def get_likelihood(mdl, loader, plot = True,save=False,digit=None,ood=False,empty_dmaps=False):
    # TODO - more sophisticated ood test of likelihood
    """"'ood' is simply magnifying the value of the image by 100 and checking the loglikelihood is lower"""
    
    anamoly_scores = []; errors = []
    n_cows = []; ldj_l = []
    min_anamoly = 1e99; max_anamoly = 0
    
    for i, data in enumerate(tqdm(loader, disable=False)): # c.hide_tqdm_bar
    
        if a.args.data == 'mnist':
            images,labels = data
        else:
            images,dmaps,labels,binary_labels , annotations, point_maps  = data_loader.preprocess_batch(data)
            
        if labels[0] != digit and a.args.data == 'mnist' and digit is not None:
            continue
        
        unnorm = data_loader.UnNormalize(mean=tuple(c.norm_mean),
                             std=tuple(c.norm_std))
        #dmaps[:,:,:] = torch.zeros(1,mdl.density_map_h,mdl.density_map_w).to(c.device)
        
        for i in range(len(labels)):
            
            mdl = mdl.to(c.device)
            
            images_list = []
            
            # supply blank density map to model and see what conditional likelihood
            # images_list.append(torch.flip(images[i].unsqueeze(0).unsqueeze(0),(3,)).squeeze(0).squeeze(0))
            # images_list.append(torch.flip(images[i].unsqueeze(0).unsqueeze(0),(2,)).squeeze(0).squeeze(0))
            images_list.append(images[i])
            
            # transformed_imgs = torch.stack(images_list).to(c.device)
            # empty_dmaps = torch.zeros(len(transformed_imgs),mdl.density_map_h,mdl.density_map_w).to(c.device)
            
            if ood:
                images[i] = images[i] * 100 # images_ood
                # x_ood, log_det_jac_ood = mdl(images_ood,labels[i],rev=False)
                # dmap_rev_np_ood = x_ood[0].squeeze().cpu().detach().numpy()
                # mean_ll_ood = x_ood[0].mean()
            
            z_l = []; ldj = []
            
            for j in range(len(images_list)):
                
                if empty_dmaps:
                    z, log_det_jac = mdl(images_list[j].unsqueeze(0),
                                         torch.zeros(1,mdl.density_map_h,mdl.density_map_w).to(c.device),rev=False,jac=True)
                else:
                    z, log_det_jac = mdl(images_list[j].unsqueeze(0),dmaps[i].unsqueeze(0),rev=False,jac=True)
                
                z = z[0]#.squeeze().cpu().detach().numpy()
                
                # mean_ll = z[0].mean()
                # sum_ll = np.sum(z)
                z_l.append(z)
                ldj.append(log_det_jac)
                
            z = torch.stack(z_l).cpu().detach().numpy()
            anamoly_score = np.mean(z ** 2)
            anamoly_scores.append(anamoly_score)
            
            im = unnorm(images[i])
            im = im.permute(1,2,0).cpu().numpy()
            
            if anamoly_score > max_anamoly:
                max_anamoly_score_im = im
                max_anamoly = anamoly_score
            
            if anamoly_score < min_anamoly:
                min_anamoly_score_im = im
                min_anamoly = anamoly_score
            
            pred = dmaps[i].sum() - ((a.args.noise)/2)*dmaps[i].shape[0]*dmaps[i].shape[1]
            
            ldj_l.append(ldj[0].cpu().detach().numpy())
            n_cows.append(len(labels[i]))
            errors.append(np.abs(len(labels[i])-pred.cpu().detach().numpy()))
            
            # print('mean_ll')
            # print(mean_ll)
            # print('sum_ll')
            # print(sum_ll)
            # print('anamoly_score')
            # print(anamoly_score)
            # print('annoations n')
            # print(len(labels[i]))
            # print('----')
            
            if plot:
                
                if ood: 
                    n_plots = 3 
                else: 
                    n_plots = 2
                
                fig, ax = plt.subplots(n_plots,1)
                plt.ioff()
                #fig.suptitle('{} \n mean reconstruction: {}'.format(title,mean_pred),y=1.0,fontsize=24)
                fig.set_size_inches(8*1,12*1)
                fig.set_dpi(100)
                
                if a.args.data =='mnist':
                    im = images[0].squeeze().cpu().numpy()
                else:
                    im = images[0].permute(1,2,0).cpu().numpy()
                
                if a.args.data =='mnist':
                    ax[0].imshow(im)
                else:
                    ax[0].imshow((im * 255).astype(np.uint8))
                    #ax[1].imshow(dmap_rev_np)
                
                ax[1].hist(z.flatten(),bins = 30)
                
                # if ood:
                #     ax[2].hist(dmap_rev_np_ood.flatten(),bins = 30)
                
                if save:
                    if not os.path.exists(g.VIZ_DIR):
                        os.makedirs(g.VIZ_DIR)
                    plt.savefig("{}/{}.jpg".format(g.VIZ_DIR,mdl.modelname), bbox_inches='tight', pad_inches = 0)
            
            break
    
    mean_ll = -99    
    
    # if ood:
    #     out = mean_ll,mean_ll_ood # z[0],x[0], images_ood[0],x_ood[0]
    # else:
    #     out = mean_ll #mean_ll # z[0],x[0],
    
    fig, ax = plt.subplots(2,4)
    plt.ioff()
    fig.set_size_inches(8*6,8*1)
    fig.set_dpi(100) 
    
    ax[0,0].scatter(ldj_l,n_cows)
    ax[0,0].set_xlabel('Log det jac')
    ax[0,0].set_ylabel('n cows')
    
    ax[0,1].scatter(anamoly_scores,n_cows)
    ax[0,1].set_xlabel('anamoly scores')
    ax[0,1].set_ylabel('n cows')

    ax[0,2].scatter(anamoly_scores,ldj_l)
    ax[0,2].set_xlabel('anamoly scores')
    ax[0,2].set_ylabel('log det jac')

    ax[0,3].scatter(anamoly_scores,errors)
    ax[0,3].set_xlabel('anamoly_scores')
    ax[0,3].set_ylabel('errors')
    
    ax[1,0].scatter(ldj_l,errors)
    ax[1,0].set_xlabel('log det jac')
    ax[1,0].set_ylabel('errors')
    
    ax[1,1].imshow((min_anamoly_score_im * 255).astype(np.uint8), aspect="auto")
    ax[1,1].title.set_text('Min anamoly score')
    
    ax[1,2].imshow((max_anamoly_score_im * 255).astype(np.uint8), aspect="auto")
    ax[1,2].title.set_text('Max anamoly score')
    
    ax[1,3].scatter(n_cows,errors)
    ax[1,3].set_xlabel('n cows')
    ax[1,3].set_ylabel('errors')
    
    fig.suptitle('LDJ / Anamoly / Error pair plots - Anno data',y=1.0,fontsize=16) 
    
    print(np.where(np.array(anamoly_scores) == np.amax(np.array(anamoly_scores))))
    
    plt.show()
    
    return #out # x.size() #

# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def count_parameters(mdl):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in mdl.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def slice_directory(dir_path,frmt="jpg"):
    img_names = os.listdir(dir_path)
    
    if not os.path.exists(dir_path+'/out/'):
        os.makedirs(dir_path+'/out/')
    
    for img in img_names:
        if img.endswith(frmt):   
            im = cv2.imread(dir_path+img)
            split_image(img=im,save=True,overlap=0,name=img[:-4],path=dir_path+'/out/',frmt=frmt,dlr=False)
        
# https://github.com/Devyanshu/image-split-with-overlap/blob/master/split_image_with_overlap.py
# minor edits to function, remove non-square opt, grey scale
def split_image(img,save=True,overlap=0,name=None,path=None,frmt=None,dlr=False):
    
    if save:
        assert name and path and frmt
        
    assert 0 <= overlap < 1
    
    splits = []
    
    # insert single channel dim for greyscale
    if len(img.shape) == 2:
        img_h, img_w = img.shape
    else:
        img_h, img_w, _ = img.shape
    
    if dlr:
        split_width = 320
        split_height = 320  
    else:
        split_width = 800
        split_height = 608
    
    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)
    
    count = 0
    
    for i in Y_points:
        for j in X_points:
            split = img[i:i+split_height, j:j+split_width]
            
            #if overlap==0:
            splits.append(split) # appending with large image overlaps causes memory issues
            
            if save:
                patch_name = '{}_{}.{}'.format(name, count, frmt)
                cv2.imwrite(path+patch_name, split)
                
            count += 1
    
    return splits

def stich_image(img_size,image_patches,name,save=True,overlap=0,path=None,frmt=None,geo=False,mdl_type='',dlr=False):
    ''' img_size = tuple of image size'''
    
    if save:
        assert name and path and frmt
        
    assert 0 <= overlap < 1
    
    # insert single channel dim for greyscale
    if len(img_size) == 2:
        img_h, img_w = img_size
    else:
        img_h, img_w, _ = img_size
        
    predicted = np.zeros(shape=((img_size)[:2]+(1,)),dtype=np.float32)
    
    if dlr:
        split_width = 320
        split_height = 320
    else:
        split_width = 800
        split_height = 608
    
    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)
    
    count = 0
    
    for i in tqdm(Y_points,desc='Filling in rows of image'):
        for j in X_points: 
            predicted[i:i+split_height, j:j+split_width] = image_patches[count] # .astype(np.uint8)
            count += 1
    
    if save and not geo:
        print('Saving image...')
        img_name = 'pred_{}_{}.{}'.format(mdl_type,name,'png') #'jpg'
        predicted = predicted.transpose(2,0,1)
        
        x_scaler = StandardScaler()
        predicted[0,:,:] = x_scaler.fit_transform(predicted[0,:,:])
        
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=predicted[0].min(), vmax=predicted[0].max())
        image = cmap(norm(predicted[0]))
        plt.imsave('{}'.format(img_name), image)
                
        #im = Image.fromarray((predicted[0]).astype(np.uint8))
        #im.save(path+img_name)
        print("Done.")
        #cv2.imwrite(path+img_name,predicted.astype(np.uint8))
       
    if geo:
        out = predicted
    else:
        out = None
        
    del predicted
    
    return out

def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

# def stich_image(img_dir=None,img_prefix=None,raw_img_path=None,patch_width=None):
    
#     img_ls = os.listdir(img_dir)
#     patch_name_ls = [[{'name':'','path':''}]]
    
#     for img in img_ls:
    
#       match = re.match(pattern=img_prefix,string=img)
#       if match != None:
#           if match.group() == img_prefix:
#               patch_name_ls.append(img.removeprefix(img_prefix)) # Python 3.9+
              
#     sorted(patch_name_ls, key=lambda x: int(x.split('.', 1)[0]))
    
#     im_width = cv2.imread(raw_img_path).shape[1]
#     row_split = im_width - patch_width
#     counter = 0
#     for patch_name in patch_name_ls:
#         patch = cv2.imread(os.path.join(img_dir,img_prefix,patch_name))

     #if re.match(pattern=img_name,string=img): 
    return None

# For MNIST
# from: https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
# @ptrblck
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
class AddUniformNoise(object):
    def __init__(self, r1=0., r2=1e-2):
        self.r1 = r1
        self.r2 = r2
        
    def __call__(self, tensor):
        # uniform tensor in pytorch: 
        # https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
        return tensor + torch.FloatTensor(tensor.size()).uniform_(self.r1, self.r2)
    
    def __repr__(self):
        return self.__class__.__name__ + '(r1={0}, r2={1})'.format(self.r1, self.r2)


def make_model_name(train_loader=None):
     now = datetime.now() 
     
     if train_loader == None:
         bs = 'search'
     else:
         bs = train_loader.batch_size
            
     parts = [a.args.schema,
              a.args.model_name,
              os.uname().nodename,
             "BS"+str(bs),
             "LR_I"+str(a.args.learning_rate),
             "E"+str(a.args.meta_epochs*a.args.sub_epochs),
             "DIM"+str(c.density_map_h),
             "OPTIM"+str(a.args.optim)]
     
     if a.args.model_name not in g.BASELINE_MODEL_NAMES:
         parts.append("FE_"+str(a.args.feat_extractor))    
     
     if a.args.dmap_scaling != 1:
         parts.append('SC{}'.format(a.args.dmap_scaling))
     
     if a.args.model_name == 'NF':
         parts.append('NC'+str(c.n_coupling_blocks))
         parts.append(a.args.subnet_type)
         
         if a.args.subnet_bn:
             parts.append('BN')
             
     if a.args.jac:
         parts.append('JC')
    
     if a.args.all_in_one:
         parts.append('IN1')
         
     if a.args.split_dimensions:
         parts.append('SPLIT')

     parts.append(a.args.sampler)
     
     if a.args.model_name in ['UNet_seg','LCFCN']:
         parts.append('MX_SZ_'+str(a.args.max_filter_size))
        
     parts.append(str(a.args.scheduler))
     
     if a.args.data == 'dlr':
         parts.append('DLRACD')
     
     if a.args.data == 'mnist':
         parts.append('MNIST')
     
     if a.args.model_name == 'NF' and a.args.joint_optim:
         parts.append('JO')
         
     if a.args.model_name == 'NF' and a.args.pretrained:
         parts.append('PT')
         
     if a.args.model_name == 'NF' and a.args.pyramid:
         parts.append('PY_{}'.format(a.args.n_pyramid_blocks))
         
     if c.counts and not a.args.data == 'mnist':
         parts.append('CT')
     
     if a.args.model_name == 'NF' and a.args.fixed1x1conv:
         parts.append('1x1')
         
     if c.scale != 1:
         parts.append('SC_{}'.format(c.scale))
         
     if a.args.model_name == 'NF' and c.dropout_p != 0:
         parts.append('DP_{}'.format(c.dropout_p))
         
     parts.extend(["WD",str(a.args.weight_decay)])
         
     if c.train_feat_extractor or a.args.load_feat_extractor_str != '':
         parts.append('FT')
         
     if a.args.sigma != 4 and not a.args.data == 'mnist':
         parts.extend(["FSG",str(a.args.sigma)])
         
     if a.args.model_name == 'NF' and c.clamp_alpha != 1.9 and not a.args.data == 'mnist':
         parts.extend(["CLA",str(c.clamp_alpha)])
         
     if c.test_train_split != 70 and not a.args.data == 'mnist':
         parts.extend(["SPLIT",str(c.test_train_split)])
         
            
     parts.append(str(now.strftime("%d_%m_%Y_%H_%M_%S")))
     
     modelname = "_".join(parts)
     
     print("Training Model: ",modelname)
     
     if a.args.mode == 'search':
         modelname = 'ray_'+modelname
     return modelname
 
def make_hparam_dict(val_loader):
    
    hparam_dict = {'schema':a.args.schema,
                        'arch':a.args.model_name,
                        'learning rate init.':a.args.learning_rate,
                        'weight decay':a.args.weight_decay,
                        'batch size':val_loader.batch_size,
                        'image height':c.density_map_h,
                        'image width':c.density_map_w,
                        'joint optimisation?':a.args.joint_optim,
                        'global average pooling?':c.gap,
                        'scale:':c.scale,
                        'jac?:':a.args.jac,
                        'annotations only?':a.args.sampler == 'anno',
                        'pretrained?':a.args.pretrained,
                        'feature pyramid?':a.args.pyramid,
                        'feature extractor?':a.args.feat_extractor,
                        '1x1convs?':a.args.fixed1x1conv,
                        'conv filters':a.args.filters,
                        'mcnn':a.args.subnet_type == 'MCNN',
                        'fc_width':c.width,
                        'finetuned?':c.train_feat_extractor,
                        'mnist?':a.args.data == 'mnist',
                        'split_dims?':a.args.split_dimensions,
                        'counts?':c.counts,
                        'all_in_one?':a.args.all_in_one,
                        'n pyramid blocks?':a.args.n_pyramid_blocks,
                        'subnet_type?':a.args.subnet_type,
                        'prop. of data':c.data_prop,
                        'clamp alpha':c.clamp_alpha,
                        'epochs':a.args.meta_epochs*a.args.sub_epochs,
                        'no. of coupling blocks':c.n_coupling_blocks,
                        'dmap sigma':a.args.sigma,
                        'dmap scaling':a.args.dmap_scaling,
                        'feat vec length':n_feats_select()}
    
    return hparam_dict

def create_point_map(mdl,annotations,pred=False,resize=False):
    # add points onto basemap
    
    if mdl == None:
        # error introduced here as float position annotation centre converted to int
        if resize:
            base_map = np.zeros((256,256), dtype=np.float32) # c.raw_img_size
        else:
            
            if pred:
                pd = 8
                base_map = np.zeros((c.raw_img_size[1]+pd,c.raw_img_size[0]), dtype=np.float32)
            else:
                base_map = np.zeros((c.raw_img_size[1], c.raw_img_size[0]), dtype=np.float32)

    else:
        base_map = np.zeros((mdl.density_map_h,mdl.density_map_w), dtype=np.float32)    

    point_flags = []
    
    for point in annotations:
        
        if pred:
            point = np.insert(point, 0, 0., axis=0)
        
        offset = 1
        
        if not pred:
            point_flags.append(point[0])
        
        # subtract 1 to account for 0 indexing
        # NOTE: this overrides duplicate annotation points (4 out of 22k)
        if pred:
            
            if resize:
                base_map[int(round((point[2]))-offset),int(round((point[1]))-offset)] = 1 # +=1
            else:
                #base_map[int(round((point[2])*c.raw_img_size[1])-offset),int(round((point[1])*c.raw_img_size[0])-offset)] = 1 # +=1
                base_map[int(round((point[1]))-offset),int(round((point[2]))-offset)] = 1 # +=1
        else:
        
            # subtract 1 to account for 0 indexing
            # NOTE: this overrides duplicate annotation points (4 out of 22k)
            if resize:
                base_map[int(round((point[2])*256)-offset),int(round((point[1])*256)-offset)] = 1 # +=1
            else:
                base_map[int(round((point[2])*c.raw_img_size[1])-offset),int(round((point[1])*c.raw_img_size[0])-offset)] = 1 # +=1
            
    return base_map, point_flags

def loader_check(mdl,loader):
    
    assert mdl.sigma == a.args.sigma
    assert mdl.dmap_scaling == a.args.dmap_scaling, 'model scaling is {}'.format(mdl.dmap_scaling)
    
    if not a.args.mode == 'search' and not a.args.mode ==  'eval': # zambingo
        assert mdl.noise == a.args.noise, 'model noise is {}'.format(mdl.noise)
    
    if str(type(mdl))=="<class 'baselines.UNet'>":
        if mdl.seg:
            assert a.args.model_name == 'UNet_seg'
        else:
            assert a.args.model_name == 'UNet'
    if str(type(mdl))=="<class 'baselines.CSRNet'>":
        assert a.args.model_name == 'CSRNet'
    if str(type(mdl))=="<class 'baselines.LCFCN'>":
        assert a.args.model_name == 'LCFCN'
    if str(type(mdl))=="<class 'baselines.FCRN_A'>":
        assert a.args.model_name == 'FCRN'
    if str(type(mdl))=="<class 'baselines.MCNN'>":
        assert a.args.model_name == 'MCNN'
    if str(type(mdl))=="<class 'baselines.Res50'>":
        assert a.args.model_name == 'Res50'
    if str(type(mdl))=="<class 'baselines.VGG_density'>":
        assert a.args.model_name == 'VGG'
        
    if mdl.density_map_h == 608:
        assert not a.args.resize
        
    if mdl.density_map_h == 256:
        assert a.args.resize or a.args.rrc
   
      
# working
def plot_errors(error_files_list,prediction_interval_file_name):
    
    mod = g.BASELINE_TITLE_NAMES
    
    # want to plot width of prediction interval from NF (as measure of uncertainty)
    # against actual error from baseline model
    # high correspondence shows the NF is correctly assigning uncertainty to predictions
    # https://stackoverflow.com/questions/17411940/matplotlib-scatter-plot-legend
    
    fig, ax = plt.subplots(1)
    ax.set_title("NF Prediction Interval Calibration",size=16)
    fig.set_size_inches(8,8,forward=True)    
    prediction_intervals = []
    prediction_interval_file_path = os.path.join(g.VIZ_DATA_DIR,prediction_interval_file_name)

    with open(prediction_interval_file_path) as f:
        prediction_interval_input = f.readlines()
        
        for line in prediction_interval_input:
            line = np.array(line[1:-2].split(),dtype=np.float32)
            prediction_intervals.append(line) 
    
    i = 0; colors = ['b', 'c', 'y', 'm', 'r','k','g','0.5']; axs = []
    
    for error_file_name in error_files_list:
    
        errors = []
        error_file_path = os.path.join(g.VIZ_DATA_DIR,error_file_name)
        
        with open(error_file_path) as f:
            errors_input = f.readlines()
        
            for line in errors_input:
                errors.append(line)     
        
        intervals_df = pd.DataFrame(np.array(prediction_intervals), columns=['lb', 'ub'])
        pred_width = np.array(np.abs(intervals_df['lb']-intervals_df['ub']))
        errors = np.array(errors,dtype=np.float32)+random.randint(0, 20) # todo - delete when final error paths populated
        
        axs.append(ax.scatter(errors,pred_width,color = colors[i],alpha=0.2))
        axs[i]
        i += 1
    
    #ax.vlines(errors,intervals_df['lb'],intervals_df['ub'])        
    ax.set_xlabel('Baseline error size',size=14)
    ax.set_ylabel('NF prediction interval width',size=14)
    
    plt.legend((axs[0],axs[1],axs[2],axs[3],axs[4],axs[5],axs[6],axs[7]),
               (mod[0],mod[1],mod[2],mod[3],mod[4],mod[5],mod[6],mod[7]),
               scatterpoints = 1,
               loc = 'lower left',
               ncol = 3,
               fontsize = 12)
    
    plt.subplots_adjust(left=0.1, right=0.87, top=0.9, bottom=0.1)
    plt.savefig("/home/mks29/Desktop/diagrams/temp_plot_errors_comparison", bbox_inches='tight', pad_inches = 0.3)
    plt.show()
    
def init_model(feat_extractor=None,config=None):
    
    name = make_model_name()
    
    if a.args.model_name == 'NF':
         mdl = model.CowFlow(modelname=name,feat_extractor=feat_extractor,config=config)
    elif a.args.model_name == 'UNet':
         mdl = baselines.UNet(modelname=name)
    elif a.args.model_name == 'UNet_seg':
         mdl = baselines.UNet(modelname=name,seg=True)
    elif a.args.model_name == 'CSRNet': 
         mdl = baselines.CSRNet(modelname=name)
    elif a.args.model_name ==  'MCNN':
         mdl = baselines.MCNN(modelname=name)
    elif a.args.model_name ==  'FCRN':
         mdl = baselines.FCRN_A(modelname=name)
    elif a.args.model_name ==  'VGG':
         mdl = baselines.VGG_density(modelname=name)
    elif a.args.model_name ==  'LCFCN':
         mdl = baselines.LCFCN(modelname=name)
    elif a.args.model_name ==  'Res50':
         mdl = baselines.Res50(modelname=name)
    if a.args.model_name == 'NF':
         mdl = model.CowFlow(modelname='best_mdl_NF',feat_extractor=feat_extractor,config=config)
    elif a.args.model_name == 'UNet':
         mdl = baselines.UNet(modelname='best_mdl_UNet')
    elif a.args.model_name == 'UNet_seg':
         mdl = baselines.UNet(modelname='best_mdl_UNet_seg',seg=True)
    elif a.args.model_name == 'CSRNet': 
         mdl = baselines.CSRNet(modelname='best_mdl_CSRNet')
    elif a.args.model_name ==  'MCNN':
         mdl = baselines.MCNN(modelname='best_mdl_MCNN')
    elif a.args.model_name ==  'FCRN':
         mdl = baselines.FCRN_A(modelname='best_mdl_FCRN')
    elif a.args.model_name ==  'VGG':
         mdl = baselines.VGG_density(modelname='best_mdl_VGG')
    elif a.args.model_name ==  'LCFCN':
         mdl = baselines.LCFCN(modelname='best_mdl_LCFCN')
    elif a.args.model_name ==  'Res50':
         mdl = baselines.Res50(modelname='best_mdl_Res50')
         
    return mdl
    
