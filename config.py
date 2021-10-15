'''This file configures the training procedure'''
import os

proj_dir = "/home/matthew/Desktop/laptop_desktop/clones/cow_flow/data"
data_prop = 1 # proportion of the full dataset to use 
test_train_split = 70 # percentage of data to allocate to train set
balanced = True # whether to have a 1:1 mixture of empty:annotated images
annotations_only = False # whether to only use image patches that have annotations

if annotations_only:
    fixed_indices = False # must be off for annotations only runs
    assert not fixed_indices
else:
    fixed_indices = False # turn this off for actual experiments, on to speed up code
    
counts = False
mnist = False 

if mnist:
    one_hot = True # only for MNIST
else:
    one_hot = False

if one_hot:
    channels = 10 # onehot 1 num -> 0,0,0,1 etc
else:
    channels = 1 # greyscale mnist, density maps

test_run = False # use only a small fraction of data to check everything works
validation = False
joint_optim = False
pretrained = True
feat_extractor = "resnet18" # alexnet, vgg16_bn,resnet18, none
feat_extractor_epochs = 50
train_feat_extractor = True
gap = False # global average pooling
clip_value = 1 # gradient clipping
scheduler = 'exponential' # exponential, none
# TODO resnet18, mnist_resnet,


# unused: n_scales = 1 #3 # number of scales at which features are extracted, img_size is the highest - others are //2, //4,...
if feat_extractor == "alexnet":
    n_feat = 256 #* n_scales # do not change except you change the feature extractor
elif feat_extractor == "vgg16_bn":
    n_feat = 512 #* n_scales # do not change except you change the feature extractor
elif feat_extractor == "resnet18":
    n_feat = 512

elif feat_extractor == "none":
    n_feat = 1 

# core hyper params
weight_decay = 1e-5 # differnet: 1e-5
n_coupling_blocks = 1

# vectorised params must always be passed as lists
lr_init = [2e-4]
batch_size = [4] # actual batch size is this value multiplied by n_transforms(_test)

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 1
sub_epochs = 1

# data settings
#dataset_path = "mnist_toy"
#class_name = "dummy_class"

if mnist and feat_extractor == "none":
    img_size = (28,28)
elif mnist:
    img_size = (228,228) # (28,28)
else:
    img_size = (800, 600) # width, height (x-y)
    
img_dims = [3] + list(img_size)

# density map ground truth generation
filter_size = 15 # as per single image mcnn paper
sigma = 4.0 # "   -----    "

# TODO: rename this parameter
if not mnist and not counts:
    
    density_map_w = 800 #img_size[0]
    
    if feat_extractor == 'resnet18':
        density_map_h = 608 #img_size[1]
    elif feat_extractor == 'alexnet':
         density_map_h = 544 #img_size[1]
         density_map_w = 768
    elif feat_extractor == 'vgg16_bn':
        density_map_h = 576 #img_size[1]
        

elif not mnist:
    # size of count expanded to map spatial feature dimensions
    density_map_h = 18 * 2 # need at least 2 channels, expand x2, then downsample (haar)
    density_map_w = 24 * 2
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

# differ net config settings

# device settings
import torch

gpu = False

if not gpu:
    device = 'cpu' 
else: 
    device = 'cuda' 
    torch.cuda.set_device(0)

# unused 
## transformation settings
#transf_rotations = True
#transf_brightness = 0.0
#transf_contrast = 0.0
#transf_saturation = 0.0
    
# nb: these are the same as the defaults specified for the pretrained pytorch
# model zoo
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] 

# network hyperparameters
# edited: cows counting - only one scale for now
clamp_alpha = 1.9 # see paper equation 2 for explanation

#fc_internal = 2048/2 # number of neurons in hidden layers of s-t-networks
#dropout = 0.0 # dropout in s-t-networks

# dataloader parameters
n_transforms = 4 # number of transformations per sample in training
n_transforms_test = 64 # number of transformations per sample in testing
# batch_size_test = batch_size * n_transforms // n_transforms_test

# output settings
debug = False
tb = False
verbose = True
report_freq = 50 # nth minibatch to report on (1 = always)
dmap_viz = False
hide_tqdm_bar = False
save_model = False # also saves a copy of the config file with the name of the model
checkpoints = False

if debug:
    schema = 'schema/debug'
else:
    schema = 'saving_models_test'
  
pc = os.uname().nodename

    