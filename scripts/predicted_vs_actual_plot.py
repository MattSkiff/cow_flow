# plot predicted versus actual for cows dataset for models from both val/train splits of holdout dataset

import os
import sys

sys.path.append("/home/mks29/clones/cow_flow/")

# external
import torch
from torch.utils.data.sampler import SubsetRandomSampler # RandomSampling
from torch.utils.data import DataLoader # Dataset 
from matplotlib import pyplot as plt                                                                                                                                                                   

# internal
import  config as c
import arguments as a
from data_loader import prep_transformed_dataset, train_val_split
from eval import eval_baselines, dmap_metrics
from utils import load_model

assert a.args.holdout

mdl_path = '/home/mks29/clones/cow_flow/models/best_86_widereal_w_vgg1NF_lowNSE_C5_NF_quatern_BS16_LR_I0.0001_E1000_DIM608_OPTIMadam_FE_vgg16_bn_NC5_conv_JC_weighted_none_JO_PY_1_WD_1e-05_19_07_2022_19_20_51'
mdl = load_model(mdl_path)

transformed_dataset = prep_transformed_dataset(is_eval=False)

f_t_indices, f_t_weights, f_v_indices, f_v_weights  = train_val_split(dataset = transformed_dataset,
                                                  train_percent = c.test_train_split,
                                                  annotations_only = False,
                                                  seed = c.seed,
                                                  oversample=False)
  
full_train_sampler = SubsetRandomSampler(f_t_indices,generator=torch.Generator().manual_seed(c.seed))
full_val_sampler = SubsetRandomSampler(f_v_indices,generator=torch.Generator().manual_seed(c.seed)) 

full_train_loader = DataLoader(transformed_dataset, batch_size=1,shuffle=False, 
                    num_workers=1,collate_fn=transformed_dataset.custom_collate_aerial,
                    pin_memory=False,sampler=full_train_sampler)

full_val_loader = DataLoader(transformed_dataset, batch_size=1,shuffle=False, 
                    num_workers=1,collate_fn=transformed_dataset.custom_collate_aerial,
                    pin_memory=False,sampler=full_val_sampler)


#y_n, y_hat_n = eval_baselines(mdl,loader,mode,is_unet_seg=False,write=True,null_filter=(a.args.bin_classifier_path != ''),write_errors_only=False,qq=False)

train_y_n, train_y_hat_n = dmap_metrics(mdl, full_train_loader,n=50,mode='val',null_filter=False,write=False,write_errors_only=False,qq=False)
val_y_n, val_y_hat_n = dmap_metrics(mdl, full_val_loader,n=50,mode='val',null_filter=False,write=False,write_errors_only=False,qq=False)


full_y_n = train_y_n + val_y_n
full_y_hat_n = train_y_hat_n + val_y_hat_n
    

plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

fig, axis = plt.subplots(figsize =(5, 5))
plt.scatter(true_value, predicted_value, c='crimson')

p1 = max(max(y_hat_n), max(y_n))
p2 = min(min(y_hat_n), min(y_n))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()