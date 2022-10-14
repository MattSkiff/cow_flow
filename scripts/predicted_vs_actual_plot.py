# plot predicted versus actual for cows dataset for models from both val/train splits of holdout dataset
# executable script

import sys

sys.path.append("/home/mks29/clones/cow_flow/")

# add script args
# import argparse
# parser = argparse.ArgumentParser(description='Create predicted versus actuals plot for supplied model path, dataloader args and title')
# parser.add_argument('-title',"--plot_title",help="Specify plot title",default='Plot title')
# plot_args = parser.parse_args()

# external
from torch.utils.data import DataLoader # Dataset 
from matplotlib import pyplot as plt                                                                                                                                                                   

# internal
import arguments as a
import gvars as g
from data_loader import prep_transformed_dataset
from eval import eval_baselines, dmap_metrics
from utils import load_model

assert a.args.holdout
assert a.args.title != ''
assert a.args.mode == 'plot'
assert a.args.batch_size == 1

def plot_pred_vs_actual(title=''):
    
    mdl = load_model(a.args.mdl_path)
    transformed_dataset = prep_transformed_dataset(is_eval=False)
    val_loader = DataLoader(transformed_dataset, batch_size=a.args.batch_size,shuffle=False, 
                        num_workers=4,collate_fn=transformed_dataset.custom_collate_aerial,
                        pin_memory=False)

    if a.args.model_name == 'NF':
        val_y_n, val_y_hat_n = dmap_metrics(mdl, val_loader,n=1,mode='val',null_filter=False,write=False,write_errors_only=False,qq=True)
    else:
        val_y_n, val_y_hat_n = eval_baselines(mdl,val_loader,mode='val',is_unet_seg=a.args.model_name == "UNet_seg",write=False,null_filter=False,write_errors_only=False,qq=True)

    fig, axis = plt.subplots(figsize =(10, 10))
    #axis.plot([0, 1], [0, 1], transform=axis.transAxes)
    
    plt.scatter(val_y_n, val_y_hat_n, c='crimson',alpha=0.4)
    
    p1 = max(max(val_y_hat_n), max(val_y_n))
    p2 = min(min(val_y_hat_n), min(val_y_n))
    
    plt.plot([p1, p2], [p1, p2], '--') # 'b-'
    plt.xlabel('True Values', fontsize=28)
    plt.ylabel('Predictions', fontsize=28)
    plt.title(title, fontsize=36)
    plt.xticks(fontsize=24) # plt.xticks(fontsize=14, rotation=90)
    plt.yticks(fontsize=24)
    plt.axis('equal')
    #plt.show()
    plt.savefig("{}/pred_vs_actual_{}.jpg".format(g.VIZ_DIR,mdl.modelname), pad_inches = 1) # plot_args.plot_title
    
plot_pred_vs_actual(title=a.args.title) # plot_args.plot_title
