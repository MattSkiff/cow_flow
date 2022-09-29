# create histogram of dataset for training/validation and holdout datasets

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
from data_loader import prep_transformed_dataset, train_val_split, preprocess_batch

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

train_counts = []
val_counts = []

for i, data in enumerate(full_train_loader):
    
    images,dmaps,labels, binary_labels , annotations, point_maps  = preprocess_batch(data)
    train_counts.append(len(annotations[0]))
    
    
for i, data in enumerate(full_val_loader):
    
    images,dmaps,labels,binary_labels , annotations, point_maps  = preprocess_batch(data)
    val_counts.append(len(annotations[0]))
    
train_val_set = train_counts + val_counts

train_val_set_nonzero = [i for i in train_val_set if i != 0]

plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

fig, axis = plt.subplots(figsize =(5, 5))
axis.set_xlabel('No of Livestock Annotated', fontsize = 14)
axis.set_ylabel('No. of Image Patches', fontsize = 14)
#fig.suptitle('Histogram of Object Counts{}'.format(title,str(mode)),y=1.0,fontsize=16) # :.2f
axis.hist(train_val_set_nonzero,edgecolor='black') # , bins = [0, 20, 40, 80, 160]
plt.show()