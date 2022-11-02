# plot predicted versus actual for cows dataset for models from both val/train splits of holdout dataset
# executable script

import sys
import numpy as np

sys.path.append("/home/mks29/clones/cow_flow/")


# external
from torch.utils.data import DataLoader # Dataset 
from matplotlib import pyplot as plt                                                                                                                                                                   

# internal
import arguments as a
import gvars as g
from data_loader import prep_transformed_dataset
from eval import eval_baselines, dmap_metrics
from utils import load_model

#assert a.args.holdout
assert a.args.mode == 'eval'
assert a.args.batch_size == 1

def print_errors(ub=20,lb=10):
    
    mdl = load_model(a.args.mdl_path)
    transformed_dataset = prep_transformed_dataset(is_eval=False)
    im_paths = transformed_dataset.im_paths
    val_loader = DataLoader(transformed_dataset, batch_size=a.args.batch_size,shuffle=False, 
                        num_workers=4,collate_fn=transformed_dataset.custom_collate_aerial,
                        pin_memory=False)

    if a.args.model_name == 'NF':
        val_y_n, val_y_hat_n = dmap_metrics(mdl, val_loader,n=1,mode='val',null_filter=False,write=False,write_errors_only=False,qq=True)
    else:
        val_y_n, val_y_hat_n = eval_baselines(mdl,val_loader,mode='val',is_unet_seg=a.args.model_name == "UNet_seg",write=False,null_filter=False,write_errors_only=False,qq=True)

    
    diffs = np.array(val_y_hat_n)-np.array(val_y_n)
    print(np.array(im_paths)[np.logical_and(diffs < ub,diffs > lb)])
    
print_errors() 
