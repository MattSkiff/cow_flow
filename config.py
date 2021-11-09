'''This file configures the training procedure'''
proj_dir = "/home/matthew/Desktop/laptop_desktop/clones/cow_flow/data"

# device settings
import torch
import arguments as a

gpu = True

## Data Options ------
mnist = False 
counts = False # must be off for pretraining feature extractor (#TODO)
balanced = True # whether to have a 1:1 mixture of empty:annotated images
weighted = True # whether to weight minibatch samples
annotations_only = False # whether to only use image patches that have annotations
test_run = False # use only a small fraction of data to check everything works
validation = True # whether to run validation per meta epoch
data_prop = 0.1 # proportion of the full dataset to use     
test_train_split = 70 # percentage of data to allocate to train set

## Density Map Options ------
filter_size = 45 # as per single image mcnn paper
sigma = 12.0 # "   -----    " 
scale = 1 # 4, 2 = downscale dmaps four/two fold, 1 = unchanged

## Feature Extractor Options ------
pretrained = False
feat_extractor = "resnet18" # alexnet, vgg16_bn,resnet18, none # TODO mnist_resnet, efficient net
feat_extractor_epochs = 50
train_feat_extractor = False # whether to finetune or load finetuned model 
load_feat_extractor_str = '' # '' to train from scratch, loads FE 
# nb: pretraining FE saves regardless of save flag

## Architecture Options ------
fixed1x1conv = True 
pyramid = False # only implemented for resnet18
gap = True # global average pooling
downsampling = False # whether to downsample (5 ds layers) dmaps by converting spatial dims to channel dims
n_coupling_blocks = 2

## Subnet Architecture Options
subnet_type = 'conv' # options = fc, conv
filters = 32 # conv ('64' recommended min)
batchnorm = False # conv
width = 400 # fc ('128' recommended min)
dropout_p = 0.0 # fc - 0 for no dropout

# Hyper Params and Optimisation ------
joint_optim = False # jointly optimse feature extractor and flow
scheduler = 'none' # exponential, none
weight_decay = 1e-4 # differnet: 1e-5
clip_value = 1 # gradient clipping
clamp_alpha = 1.9 

# vectorised params must always be passed as lists
lr_init = [2e-3]
batch_size = [1] # actual batch size is this value multiplied by n_transforms(_test)

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 1
sub_epochs = 1

## Output Settings ----
schema = '1x1_test' # if debug, ignored
debug = True # report loads of info/debug info
tb = False # write metrics, hyper params to tb files
verbose = True # report stats per sub epoch and other info
report_freq = -1 # nth minibatch to report minibatch loss on (1 = always,-1 = turn off)
viz = False # visualise outputs and stats
hide_tqdm_bar = True
save_model = True # also saves a copy of the config file with the name of the model
checkpoints = False # saves after every meta epoch

# nb: same as the defaults specified for the pretrained pytorch model zoo
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] 

# Logic # TODO - move out of config.py ------
if not gpu:
    device = 'cpu' 
else: 
    device = 'cuda' 
    torch.cuda.set_device(0)

if annotations_only:
    fixed_indices = False # must be off for annotations only runs
    assert not fixed_indices
else:
    fixed_indices = False # turn this off for actual experiments, on to speed up code
    # save/load is now redundant

# "condition dim"
if feat_extractor == "alexnet":
    n_feat = 256 
elif feat_extractor == "vgg16_bn":
    n_feat = 512 
elif feat_extractor == "resnet18":
    n_feat = 512
elif feat_extractor == "none":
    if mnist:
        n_feat = 1 
    else:
        # conditioning on raw RGB image
         n_feat = 3

if mnist:
    one_hot = True # only for MNIST
else:
    one_hot = False

if one_hot:
    channels = 10 # onehot 1 num -> 0,0,0,1 etc
elif mnist or feat_extractor != "none":
    channels = 1 # greyscale mnist, density maps
elif counts:
    channels = 1 # duplication not needed (linear subnets)
else:
    # TODO - this is a massive hack to test feature extractor-less  NF
    channels = 2 # duplicate dmap over channel dimension (1->2)

if debug and tb:
    schema = 'schema/debug' # aka ignore debugs
elif debug:
    schema = 'debug' # aka ignore debugs

if mnist and feat_extractor == "none":
    img_size = (28,28)
elif mnist:
    img_size = (228,228) # (28,28)
else:
    img_size = (800, 600) # width, height (x-y)

img_dims = [3] + list(img_size) # RGB + x-y

# TODO: rename this parameter
# this effects the padding applied to the density maps
if not mnist and not counts and downsampling:
    density_map_w = 800//scale #img_size[0]
    if feat_extractor == 'resnet18':
        density_map_h = 608//scale #img_size[1]
    elif feat_extractor == 'alexnet':
         density_map_h = 544//scale #img_size[1]
         density_map_w = 768//scale
    elif feat_extractor == 'vgg16_bn':
        density_map_h = 576//scale #img_size[1]
    elif feat_extractor == 'none':
        density_map_h = 600//scale
elif not mnist and not counts and not downsampling:
    density_map_w = 800//scale
    density_map_h = 600//scale
elif not mnist:
    # size of count expanded to map spatial feature dimensions
    density_map_h = 19 * 2 # need at least 2 channels, expand x2, then downsample (haar)
    density_map_w = 25 * 2 # TODO - rework so this ties to ft_dims (vgg,alexnet, resnet, etc)
elif mnist and feat_extractor != "none":
    if gap:
        density_map_h = img_size[1] * 2
        density_map_w = img_size[0] * 2
    else:
        density_map_h = 6 * 2
        density_map_w = 6 * 2 # feature size x2 (account for downsampling)
elif mnist and feat_extractor == "none":
    # minimum possible dimensionality of flow possible with coupling layers
    density_map_h = 4
    density_map_w = 4

if gpu:
    device = torch.device("cuda:{}".format(a.args.gpu_number) if torch.cuda.is_available() else "cpu") # select gpu

if a.args.gpu_number != 0:
    assert gpu

# Checks ------ 
assert not (pyramid and fixed1x1conv)
assert not (feat_extractor == 'none' and gap == True)
assert gap != downsampling
assert subnet_type in ['conv','fc']
assert feat_extractor in ['none' ,'alexnet','vgg16_bn','resnet18']
assert scheduler in ['exponential','none']

if subnet_type == 'fc':
    assert gap
    assert mnist or counts
    
if mnist:
    assert not counts

if counts:
    assert subnet_type == 'fc' and gap or subnet_type == 'conv' and not gap

if pyramid:
    n_coupling_blocks = 5 # for recording purposes
    assert (pyramid and feat_extractor == 'resnet18')
    assert (pyramid and downsampling) # pyramid nf head has  downsmapling
    assert (pyramid and not train_feat_extractor) # TODO
    assert scale == 1

assert scale in (1,2,4)
