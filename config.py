'''This file configures the training procedure (infrequently changed options)'''

# device settings
import torch
import arguments as a
import os

# this file configures opts that are changed infrequnetly or were used for development
if os.uname().nodename == 'weka-13':
    proj_dir = "/home/matthew/Desktop/laptop_desktop/clones/cow_flow/data"
else:
    proj_dir = "/home/mks29/clones/cow_flow/data"

gpu = True
seed = 101 # important to keep this constant between model and servers for evaluaton

## Dataset Options ------
load_stored_dmaps = False # speeds up precomputation (with RAM = True)
store_dmaps = False # this will save dmap objects (numpy arrays) to file
ram = False # load aerial imagery and precompute dmaps and load both into ram before training
counts = False # must be off for pretraining feature extractor (#TODO)

## Training Options ------
validation = True # whether to run validation data per meta epoch
eval_n = 10
data_prop = 0.1 # proportion of the full dataset to use (ignored in DLR ACD)
test_train_split = 70 # percentage of data to allocate to train set

## Density Map Options ------
sigma = 4.0 # "   -----    "  ignored for DLR ACD which uses gsd correspondence

if a.args.model_name == "CSRNet":
    sigma = sigma/8

## Feature Extractor Options ------
pretrained = True
feat_extractor = "resnet18" # alexnet, vgg16_bn,resnet18, none 
feat_extractor_epochs = 100
train_feat_extractor = False # whether to finetune or load finetuned model # redundent
load_feat_extractor_str = 'resnet18_FTE_5_16_11_2021_13_47_53_PT_True_BS_100' # '' to train from scratch, loads FE  # 
# nb: pretraining FE saves regardless of save flag

## Architecture Options ------
fixed1x1conv = True 
freq_1x1 = 2 # 1 for always | how many x coupling blocks to have a 1x1 conv permutation layer
n_splits = 4 # number of splits
gap = False # global average pooling
n_coupling_blocks = 5 # if pyramid, total blocks will be n_pyramid_blocks x 5

## Subnet Architecture Options
subnet_type = 'conv' # options = fc, conv
batchnorm = True # conv
width = 400 # fc ('128' recommended min)
dropout_p = 0.0 # fc only param - 0 for no dropout

# Hyper Params and Optimisation ------
joint_optim = True # jointly optimse feature extractor and flow
clip_value = 1 # gradient clipping
clamp_alpha = 1.9 

## Output Settings ----
report_freq = -1 # nth minibatch to report minibatch loss on (1 = always,-1 = turn off)
viz = False # visualise outputs and stats
hide_tqdm_bar = False
save_model = True # also saves a copy of the config file with the name of the model
checkpoints = False # saves after every meta epoch

# nb: same as the defaults specified for the pretrained pytorch model zoo
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] 

# Logic # TODO - move out of config.py -----
if not gpu:
    device = 'cpu' 
else: 
    device = 'cuda' 
    
torch.cuda.set_device(0)

# if dmap is scaled down outside of flow
# less downsample 'levels' are needed'
levels = 5

# "condition dim"
if feat_extractor == "alexnet":
    n_feat = 256 
elif feat_extractor == "vgg16_bn":
    n_feat = 512 
elif feat_extractor == "resnet18":
    n_feat = 512
elif feat_extractor == "none":
         n_feat = 3

channels = 1

if a.args.data == 'dlr':
    img_size = (a.args.image_size,a.args.image_size)
else:
    img_size = (a.args.image_size, a.args.image_size) # width, height (x-y)

raw_img_size = (800, 600)

img_dims = [3] + list(img_size) # RGB + x-y

# TODO: rename this parameter
# this effects the padding applied to the density maps
if a.args.data == 'dlr_acd':
    density_map_h,density_map_w = a.args.image_size,a.args.image_size
else:
    # size of count expanded to map spatial feature dimensions
    density_map_h = 19 * 2 # need at least 2 channels, expand x2, then downsample (haar)
    density_map_w = 25 * 2 # TODO - rework so this ties to ft_dims (vgg,alexnet, resnet, etc)

if not a.args.resize: #and not a.args.model_name == 'CSRNet':
    img_size = (800,600)
    density_map_w,density_map_h = (800,608)

# Checks ------ 

# TODO

assert not (feat_extractor == 'none' and gap == True)
assert n_splits >= 0 and n_splits < 6
assert subnet_type in ['conv','fc']
assert feat_extractor in ['none' ,'alexnet','vgg16_bn','resnet18']

if subnet_type == 'fc':
    assert not fixed1x1conv

assert not (load_stored_dmaps and store_dmaps)

if load_stored_dmaps or store_dmaps:
    assert ram

if subnet_type == 'conv':
    assert dropout_p == 0

assert freq_1x1 != 0