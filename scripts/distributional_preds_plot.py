# plot set of distributional plots showing NF outputs
# executable script

import sys

sys.path.append("/home/mks29/clones/cow_flow/")

# add script args
# import argparse
# parser = argparse.ArgumentParser(description='Create predicted versus actuals plot for supplied model path, dataloader args and title')
# parser.add_argument('-title',"--plot_title",help="Specify plot title",default='Plot title')
# plot_args = parser.parse_args()

# external
import torch
import numpy as np
import random

from tqdm import tqdm
from torch import randn
from torch.utils.data import DataLoader # Dataset 
from matplotlib import pyplot as plt      
from skimage.feature import peak_local_max #                                                                                                                                                              

# internal
import arguments as a
import gvars as g
import config as c
from data_loader import prep_transformed_dataset, UnNormalize
from utils import load_model, ft_dims_select

assert a.args.holdout
assert a.args.mode == 'plot'
assert a.args.model_name == 'NF'
assert a.args.batch_size == 1

# TODO - add plot full set and save opt
@torch.no_grad()
def plot_peaks(n=1,min_annotations=60): # 50
    '''Uses NF model and dataloader to create a set of plots showing distributional outputs '''
    
    mdl = load_model(a.args.mdl_path)
    transformed_dataset = prep_transformed_dataset(is_eval=False)
    loader = DataLoader(transformed_dataset, batch_size=a.args.batch_size,shuffle=False, 
                        num_workers=4,collate_fn=transformed_dataset.custom_collate_aerial,
                        pin_memory=False)
    
    mdl = mdl.to(c.device)
    
    for i, data in enumerate(tqdm(loader, disable=False)):
        
        if mdl.dlr_acd:
            images,dmaps,counts,point_maps = data
        elif loader.dataset.classification:
            images,dmaps,labels,binary_labels,annotations,point_maps = data
        else: 
            images,dmaps,labels,annotations,point_maps = data
             
        images = images.float().to(c.device)
         
        if mdl.scale == 1:
            n_ds = 5
        elif mdl.scale == 2:
            n_ds = 4
        elif mdl.scale == 4:
            n_ds = 3
        
        if torch.numel(labels[0]) < min_annotations:
            continue 
                    
        in_channels = c.channels*4**n_ds
        ft_dims = ft_dims_select(mdl) 

        x_list = []
                    
        ## sample from model ---
        for i in range(n):
            dummy_z = (randn(images.size()[0], in_channels,ft_dims[0],ft_dims[1])).to(c.device)
            x, _ = mdl(images,dummy_z,rev=True)
            x_list.append(x)
        
        # get back to correct image data
        unnorm = UnNormalize(mean=tuple(c.norm_mean),
                             std=tuple(c.norm_std))
        
        x_agg = torch.stack(x_list,dim=1)
        
        # sum across channel, spatial dims for counts
        x = x_agg.mean(dim=1)
        x_std = x_agg.std(dim=1)
        #x_std_norm = x_std #x_std-x #x_std.div(x) # unused

        for idx in range(images.size()[0]):               
            
            idx = random.randint(0,images.size()[0]-1)
            
            ground_truth_dmap = dmaps[idx].squeeze().cpu().detach().numpy()
            
            if mdl.dlr_acd:
                point_map = point_maps[idx].squeeze().cpu().detach().numpy()
                label_count = counts[idx]
            else:
                label_count = len(annotations[idx])
            
            dmap_rev_np = x[idx].squeeze().cpu().detach().numpy()
            #dmap_uncertainty = x_std[idx].squeeze().cpu().detach().numpy()
            sum_count = dmap_rev_np.sum()
            dist_counts  = x_agg[idx].sum((1,2,3))
            gt_count = ground_truth_dmap.sum().round()
            
            # if sum_count > 0:
            #     thres = int(sum_count)
              
            dmap_uncertainty = dmap_rev_np-ground_truth_dmap #x_std_norm[idx].squeeze().cpu().detach().numpy()
            
            # subtract constrant for uniform noise
            constant = ((mdl.noise)/2)*ground_truth_dmap.shape[0]*ground_truth_dmap.shape[1]
            loader_noise = ((a.args.noise)/2)*ground_truth_dmap.shape[0]*ground_truth_dmap.shape[1]
            
            sum_count -= constant
            dist_counts -= constant 
            gt_count -= loader_noise
            
            if mdl.dlr_acd:
                # https://stackoverflow.com/questions/60782965/extract-x-y-coordinates-of-each-pixel-from-an-image-in-python
                gt_coords = np.argwhere(point_map == 1)
            else:
                gt_coords = annotations[idx].cpu().detach().numpy()
                
            coordinates = peak_local_max(dmap_rev_np,min_distance=4,num_peaks=max(1,int(sum_count)))

            if mdl.dlr_acd:
                im = images[idx]
            else:
                im = unnorm(images[idx])
                                            
            im = images[idx].permute(1,2,0).cpu().numpy()
            fig, ax = plt.subplots(2,3, figsize=(18, 10))
            
            ax[0,0].imshow(dmap_rev_np, aspect="auto")
            ax[0,1].imshow(dmap_uncertainty, aspect="auto")
            ax[0,2].hist(dist_counts.cpu().numpy())
            ax[0,2].axvline(dist_counts.cpu().numpy().mean(), color='red', linestyle='dashed', linewidth=1)
            ax[1,0].imshow(ground_truth_dmap, aspect="auto")
            ax[1,1].imshow((im * 255).astype(np.uint8), aspect="auto")
            
            if len(coordinates) != 0:
                x_coords, y_coords = zip(*coordinates)
                ax[1,2].scatter(y_coords, x_coords,label='Predicted coordinates')
                if mdl.dlr_acd:
                   ax[1,2].scatter(gt_coords[:,1], gt_coords[:,0],c='red',marker='1',label='Ground truth coordinates') 
                else:
                    ax[1,2].scatter(gt_coords[:,1]*mdl.density_map_w, gt_coords[:,2]*mdl.density_map_h,c='red',marker='1',label='Ground truth coordinates')
                ax[1,2].legend()
                ax[1,2].set_ylim(ax[1,0].get_ylim())
                ax[1,2].set_xlim(ax[1,0].get_xlim())
                ax[1,2].title.set_text('Localizations via Peak Finding on\n Predicted Density Map\nN = {}'.format(len(coordinates)))   
               
            ax[0,0].title.set_text('Mean (N={}) Predicted Density Map\nSum Density Map = {:.2f} '.format(n,sum_count)) # (-{} manual noise adj.) # ,constant
            #ax[1,0].title.set_text('Pixel-wise Uncertainty Estimate (normalised std. dev)')
            ax[0,1].title.set_text('Image Differencing\n (Predicted, Ground Truth Density Map)')
            ax[0,2].title.set_text('Prediction Distribution (N = {})'.format(n))
            ax[1,0].title.set_text('Ground Truth Density Map\nLabel Count = {} | Sum = {:.2f}'.format(label_count,gt_count))
            ax[1,1].title.set_text('Conditioning Aerial Image')
            
            for ax in [ax[0,0],ax[0,1],ax[1,0],ax[1,1]]:
                ax.tick_params(
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    left=False,
                    right=False,
                    top=False,         # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False) #
            
            fig.subplots_adjust(hspace=0.4)
            plt.savefig("{}/NF_dist_plot_{}.jpg".format(g.VIZ_DIR,mdl.modelname), pad_inches = 1) # plot_args.plot_title
            
            return
        
    return     
    
plot_peaks() # plot_args.plot_title
    

