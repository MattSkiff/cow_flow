'''This file configures the training procedure because handling arguments in every single function'''

# custom config settings
proj_dir = "/home/matthew/Desktop/laptop_desktop/clones/cow_flow/data"
data_prop = 0.1 # proportion of the full dataset to use 
annotations_only = False # whether to only use image patches that have annotations
mnist = True 
test_run = False # use only a small fraction of data to check everything works
validation = False
feat_extractor = "resnet18"
weight_decay = 1e-4 # differnet: 1e-5

# data settings
#dataset_path = "mnist_toy"
#class_name = "dummy_class"
modelname = "fg-mnist_test_smalldims_deep_30e"

if mnist:
    img_size = (228,228) # (28,28)
else:
    img_size = (800, 600) # width, height (x-y)
    
img_dims = [3] + list(img_size)

# density map ground truth generation
filter_size = 15 # as per single image mcnn paper
sigma = 4.0 # "   -----    "

if not mnist:
    density_map_h = img_size[1]
    density_map_w = img_size[0]
else:
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
#norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] 

# network hyperparameters
# edited: cows counting - only one scale for now
n_scales = 1 #3 # number of scales at which features are extracted, img_size is the highest - others are //2, //4,...
clamp_alpha = 3 # see paper equation 2 for explanation
n_coupling_blocks = 8
fc_internal = 2048/2 # number of neurons in hidden layers of s-t-networks
#dropout = 0.0 # dropout in s-t-networks
lr_init = 2e-5

n_feat = 256 * n_scales # do not change except you change the feature extractor

# dataloader parameters
n_transforms = 4 # number of transformations per sample in training
n_transforms_test = 64 # number of transformations per sample in testing
batch_size = 10 # actual batch size is this value multiplied by n_transforms(_test)
# batch_size_test = batch_size * n_transforms // n_transforms_test

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 30
sub_epochs = 1

# output settings
debug = False
verbose = False
report_freq = 200 # nth minibatch to report on (1 = always)
dmap_viz = False
hide_tqdm_bar = False
save_model = True # also saves a copy of the config file with the name of the model