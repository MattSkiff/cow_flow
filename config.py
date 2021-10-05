'''This file configures the training procedure'''
import os

# custom config settings
proj_dir = "/home/matthew/Desktop/laptop_desktop/clones/cow_flow/data"
data_prop = 1 # proportion of the full dataset to use 
fixed_indices = True # turn this off for actual experiments
annotations_only = False # whether to only use image patches that have annotations
mnist = True 

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
feat_extractor = "alexnet" # alexnet, vgg16_bn, none
gap = False # global average pooling
clip_value = 1 # gradient clipping
scheduler = 'exponential' # exponential, none
# TODO resnet18, mnist_resnet,

# unused: n_scales = 1 #3 # number of scales at which features are extracted, img_size is the highest - others are //2, //4,...
if feat_extractor == "alexnet":
    n_feat = 256 #* n_scales # do not change except you change the feature extractor
elif feat_extractor == "vgg16_bn":
    n_feat = 512 #* n_scales # do not change except you change the feature extractor
elif feat_extractor == "none":
    n_feat = 1 

# core hyper params
weight_decay = 1e-5 # differnet: 1e-5
lr_init = [2e-2,2e-3,2e-4,2e-5,2e-6,2e-7]
n_coupling_blocks = 8
batch_size = 200 # actual batch size is this value multiplied by n_transforms(_test)

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
if not mnist:
    density_map_h = 576 #img_size[1]
    density_map_w = 800 #img_size[0]
elif mnist and feat_extractor != "none":
    if gap:
        density_map_h = img_size[1] * 2
        density_map_w = img_size[0] * 2
    else:
        density_map_h = 6 * 2
        density_map_w = 6 * 2 # feature size x2 (account for downsampling)
elif mnist and feat_extractor == "none":
    density_map_h = 4
    density_map_w = 4

# differ net config settings

# device settings
import torch

gpu = True

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
#norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] 

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
verbose = False
report_freq = 200 # nth minibatch to report on (1 = always)
dmap_viz = False
hide_tqdm_bar = False
save_model = True # also saves a copy of the config file with the name of the model
checkpoints = False

if debug:
    schema = 'schema/debug'
else:
    schema = 'schema/lri_battery'
    
pc = os.uname().nodename
modelname = "_".join(["LOC",pc,
                      "J-O",str(joint_optim),
                      "Pre-T",str(pretrained),
                      "BS",str(batch_size),
                      "NC",str(n_coupling_blocks),
                      "EPOCHS",str(meta_epochs*sub_epochs),
                      "DIM",str(density_map_h),
                      "LR_I",str(lr_init),
                      "MNIST",str(mnist),
                      "WD",str(weight_decay),
                      "FE",str(feat_extractor),
                      "LRS",str(scheduler)])