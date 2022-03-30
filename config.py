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
seed = 101

## Dataset Options ------
load_stored_dmaps = True # speeds up precomputation (with RAM = True)
store_dmaps = False # this will save dmap objects (numpy arrays) to file
ram = True # load aerial imagery and precompute dmaps and load both into ram before training
counts = False # must be off for pretraining feature extractor (#TODO)

## Training Options ------
train_model = True # (if false, will only prep dataset,dataloaders)
validation = True # whether to run validation data per meta epoch
eval_n = 1
data_prop = 1 # proportion of the full dataset to use (ignored in DLR ACD,MNIST)
test_train_split = 70 # percentage of data to allocate to train set

## Density Map Options ------
sigma = 4.0 # "   -----    "  ignored for DLR ACD which uses gsd correspondence
scale = 1 # 4, 2 = downscale dmaps four/two fold, 1 = unchanged

## Feature Extractor Options ------
pretrained = True
feat_extractor = "vgg16_bn" # alexnet, vgg16_bn,resnet18, none # TODO mnist_resnet, efficient net
feat_extractor_epochs = 10
train_feat_extractor = False # whether to finetune or load finetuned model # redundent
load_feat_extractor_str = '' # '' to train from scratch, loads FE  # final_eval_test_weka-13_BS2_LR_I[0.002]_NC5_E1_FE_resnet18_DIM608_JO_PT_PY_1_1x1_WD_0.001_FSZ_16_14_12_2021_21_49_14
# nb: pretraining FE saves regardless of save flag

## Architecture Options ------
fixed1x1conv = False 
freq_1x1 = 1 # 1 for always | how many x coupling blocks to have a 1x1 conv permutation layer
pyramid = False # only implemented for resnet18
n_splits = 4 # number of splits
gap = False # global average pooling
downsampling = True # whether to downsample (5 ds layers) dmaps by converting spatial dims to channel dims
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
debug = False # report loads of info/debug info
verbose = True # report stats per sub epoch and other info
report_freq = -1 # nth minibatch to report minibatch loss on (1 = always,-1 = turn off)
viz = False # visualise outputs and stats
hide_tqdm_bar = False
save_model = True # also saves a copy of the config file with the name of the model
checkpoints = False # saves after every meta epoch

# debug opts
debug_dataloader = False
debug_utils = False

# nb: same as the defaults specified for the pretrained pytorch model zoo
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] 

# Logic # TODO - move out of config.py ------
if not gpu:
    device = 'cpu' 
else: 
    device = 'cuda' 
    torch.cuda.set_device(0)

# if dmap is scaled down outside of flow
# less downsample 'levels' are needed'
levels = 5
# if scale == 1:
#     levels = 5
# elif scale == 2:
#     levels = 4
# elif scale == 4:
#     levels = 3

# "condition dim"
if feat_extractor == "alexnet":
    n_feat = 256 
elif feat_extractor == "vgg16_bn":
    n_feat = 512 
elif feat_extractor == "resnet18":
    n_feat = 512
elif feat_extractor == "none":
    if a.args.mnist:
        n_feat = 1 
    else:
        # conditioning on raw RGB image
         n_feat = 3

if a.args.mnist:
    one_hot = True # only for MNIST
else:
    one_hot = False

if one_hot:
    channels = 10 # onehot 1 num -> 0,0,0,1 etc
elif a.args.mnist or feat_extractor != "none":
    channels = 1 # greyscale mnist, density maps
elif counts:
    channels = 1 # duplication not needed (linear subnets)
else:
    # TODO - this is a massive hack to test feature extractor-less  NF
    channels = 2 # duplicate dmap over channel dimension (1->2)

if debug:
    a.args.schema = 'debug' # aka ignore debugs

if a.args.dlr_acd:
    img_size = (a.args.image_size,a.args.image_size)
elif a.args.mnist and feat_extractor == "none":
    img_size = (28,28)
elif a.args.mnist:
    img_size = (228,228) # (28,28)
else:
    img_size = (a.args.image_size, a.args.image_size) # width, height (x-y)

img_dims = [3] + list(img_size) # RGB + x-y

# TODO: rename this parameter
# this effects the padding applied to the density maps
if a.args.dlr_acd:
    density_map_h,density_map_w = 256,256
elif not a.args.mnist and not counts and downsampling:
    density_map_w = 256//scale #img_size[0]
    if feat_extractor == 'resnet18':
        density_map_h = 256//scale #img_size[1]
    elif feat_extractor == 'alexnet':
         density_map_h = 544//scale #img_size[1]
         density_map_w = 768//scale
    elif feat_extractor == 'vgg16_bn':
        density_map_h = 576//scale #img_size[1]
    elif feat_extractor == 'none':
        density_map_h = 600//scale
elif not a.args.mnist and not counts and not downsampling:
    density_map_w = 256//scale
    density_map_h = 256//scale
elif not a.args.mnist:
    # size of count expanded to map spatial feature dimensions
    density_map_h = 19 * 2 # need at least 2 channels, expand x2, then downsample (haar)
    density_map_w = 25 * 2 # TODO - rework so this ties to ft_dims (vgg,alexnet, resnet, etc)
elif a.args.mnist and feat_extractor != "none":
    if gap:
        density_map_h = img_size[1] * 2
        density_map_w = img_size[0] * 2
    else:
        density_map_h = 6 * 2
        density_map_w = 6 * 2 # feature size x2 (account for downsampling)
elif a.args.mnist and feat_extractor == "none":
    # minimum possible dimensionality of flow possible with coupling layers
    density_map_h = 4
    density_map_w = 4

if gpu:
    device = torch.device("cuda:{}".format(a.args.gpu_number) if torch.cuda.is_available() else "cpu") # select gpu

if a.args.gpu_number != 0:
    assert gpu

# Checks ------ 

# TODO
# assert not (pyramid and fixed1x1conv)
assert not (feat_extractor == 'none' and gap == True)
assert gap != downsampling
assert n_splits >= 0 and n_splits < 6
assert subnet_type in ['conv','fc']
assert feat_extractor in ['none' ,'alexnet','vgg16_bn','resnet18']

if subnet_type == 'fc':
    assert gap
    assert a.args.mnist or counts
    assert not fixed1x1conv

assert not (load_stored_dmaps and store_dmaps)

if load_stored_dmaps or store_dmaps:
    assert ram
 
if store_dmaps:
    assert not train_model 

if subnet_type == 'conv':
    assert dropout_p == 0
    
if a.args.mnist:
    assert not counts

if counts:
    assert subnet_type == 'fc' and gap or subnet_type == 'conv' and not gap

if pyramid:
    assert n_coupling_blocks == 5 # for recording purposes
    assert (pyramid and feat_extractor == 'resnet18')
    assert (pyramid and downsampling) # pyramid nf head has  downsmapling
    #assert (pyramid and not train_feat_extractor) # TODO
    # TODO - get pyramid working with other scales!
    assert scale == 1

assert scale in (1,2,4)
assert freq_1x1 != 0
