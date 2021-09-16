'''This file configures the training procedure because handling arguments in every single function'''

# custom config settings
proj_dir = "/home/matthew/Desktop/clones/cow_flow/data"
data_prop = 1 # proportion of the full dataset to use 
annotations_only = False # whether to only use image patches that have annotations

# data settings
dataset_path = "dummy_dataset"
class_name = "dummy_class"
modelname = "debugging_del"

img_size = (800, 600) # width, height (x-y)
img_dims = [3] + list(img_size)

# density map ground truth generation
filter_size = 15 # as per single image mcnn paper
sigma = 4.0 # "   -----    "
density_map_h = img_size[1]
density_map_w = img_size[0]

# differ net config settings

# device settings
device = 'cuda' # or 'cpu'
import torch
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
n_coupling_blocks = 4
#fc_internal = 2048 # number of neurons in hidden layers of s-t-networks
#dropout = 0.0 # dropout in s-t-networks
lr_init = 2e-4
n_feat = 256 * n_scales # do not change except you change the feature extractor

# dataloader parameters
n_transforms = 4 # number of transformations per sample in training
n_transforms_test = 64 # number of transformations per sample in testing
batch_size = 1 # actual batch size is this value multiplied by n_transforms(_test)
# batch_size_test = batch_size * n_transforms // n_transforms_test

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 12
sub_epochs = 6

# output settings
debug = False
verbose = True
report_freq = 200 # nth minibatch to report on (1 = always)
dmap_viz = True
hide_tqdm_bar = False
save_model = True
