'''This file configures the training procedure'''
proj_dir = "/home/matthew/Desktop/laptop_desktop/clones/cow_flow/data"

# device settings
import torch
import arguments as a

gpu = False

## Data Options ------
mnist = False 
counts = False # must be off for pretraining feature extractor (#TODO)
balanced = True # whether to have a 1:1 mixture of empty:annotated images
annotations_only = False # whether to only use image patches that have annotations
data_prop = 0.1 # proportion of the full dataset to use     
test_train_split = 70 # percentage of data to allocate to train set
scale = 1 # 4, 2 = downscale four/two fold, 1 = unchanged

## Density Map Options ------
filter_size = 45 # as per single image mcnn paper
sigma = 12.0 # "   -----    " 

test_run = False # use only a small fraction of data to check everything works
validation = True # whether to run validation per meta epoch

## Feature Extractor Options ------
joint_optim = False
pretrained = True
feat_extractor = "resnet18" # alexnet, vgg16_bn,resnet18, none # TODO mnist_resnet, efficient net
feat_extractor_epochs = 50
train_feat_extractor = False # whether to finetune or load finetuned model 
load_feat_extractor_str = 'resnet18_FTE_50_21_10_2021_10_27_59_PT_True' # 'resnet18_FTE_50_21_10_2021_10_27_59_PT_True' to train from scratch, loads FE 
# nb: pretraining FE saves regardless of save flag

## Architecture Options ------
pyramid = False # only implemented for resnet18
gap = True # global average pooling
downsampling = False # whether to downsample (5 ds layers) dmaps by converting spatial dims to channel dims
n_coupling_blocks = 1

## Subnet Architecture Options
batchnorm = False
filters = 32
width = 800
subnet_type = 'conv' # options = fc, conv

# Hyper Params and Optimisation ------
scheduler = 'none' # exponential, none
weight_decay = 1e-5 # differnet: 1e-5
clip_value = 1 # gradient clipping
clamp_alpha = 1.9 

# vectorised params must always be passed as lists
lr_init = [2e-3]
batch_size = [1] # actual batch size is this value multiplied by n_transforms(_test)

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 2
sub_epochs = 1

## Output Settings ----
schema = 'tb_test' # if debug, ignored
debug = False # report loads of info/debug info
tb = True # write metrics, hyper params to tb files
verbose = True # report stats per sub epoch and other info
report_freq = -1 # nth minibatch to report minibatch loss on (1 = always,-1 = turn off)
dmap_viz = True
hide_tqdm_bar = True
save_model = False # also saves a copy of the config file with the name of the model
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
if not mnist and not counts:
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

# Checks ------
assert not (feat_extractor == 'none' and gap == True)
assert gap != downsampling
assert subnet_type in ['conv','fc']
assert feat_extractor in ['none' ,'alexnet','vgg16_bn','resnet18']
assert scheduler in ['exponential','none']

if pyramid:
    n_coupling_blocks = 5 # for recording purposes
    assert (pyramid and feat_extractor == 'resnet18')
    assert (pyramid and downsampling) # pyramid nf head has  downsmapling
    assert (pyramid and not train_feat_extractor) # TODO
    assert scale == 1

assert scale in (1,2,4)
