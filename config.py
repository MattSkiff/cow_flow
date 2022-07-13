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
seed = 101  #101 # important to keep this constant between model and servers for evaluaton

## Dataset Options ------
counts = False # must be off for pretraining feature extractor (#TODO)

## Training Options ------
validation = True # whether to run validation data per meta epoch
eval_n = 10
data_prop = 1 # proportion of the full dataset to use (ignored in DLR ACD,MNIST)
test_train_split = 70 # percentage of data to allocate to train set

## Density Map Options ------
scale = 1 # 4, 2 = downscale dmaps four/two fold, 1 = unchanged

## Feature Extractor Options ------
pretrained = False
feat_extractor_epochs = 100
train_feat_extractor = False # whether to finetune or load finetuned model # redundent
load_feat_extractor_str = '' # 'resnet18_FTE_100_02_05_2022_17_34_03_PT_True_BS_64_classification_head' - to train from scratch, loads FE  # 
# nb: pretraining FE saves regardless of save flag

## Architecture Options ------
n_splits = 4 # number of splits
gap = False # global average pooling
downsampling = True # whether to downsample (5 ds layers) dmaps by converting spatial dims to channel dims
n_coupling_blocks = 5 # if pyramid, total blocks will be n_pyramid_blocks x 5

## Subnet Architecture Options
batchnorm = False # conv
width = 400 # fc ('128' recommended min)
dropout_p = 0.0 # fc only param - 0 for no dropout

# Hyper Params and Optimisation ------
joint_optim = True # jointly optimse feature extractor and flow
clip_value = 1  # gradient clipping
clamp_alpha = 1.9

## Output Settings ----
debug = False # report loads of info/debug info
verbose = True # report stats per sub epoch and other info
report_freq = -1 # nth minibatch to report minibatch loss on (1 = always,-1 = turn off)
hide_tqdm_bar = False
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
if a.args.feat_extractor == '':
    n_feat = -99

if a.args.feat_extractor == "alexnet":
    n_feat = 256 
elif a.args.feat_extractor == "vgg16_bn":
    n_feat = 512 
elif a.args.feat_extractor in ["resnet18","resnet50"]:
    n_feat = 512
elif a.args.feat_extractor == "none":
    if a.args.data == 'mnist':
        n_feat = 1 
    else:
        # conditioning on raw RGB image
         n_feat = 3

if a.args.data == 'mnist':
    one_hot = True # only for MNIST
else:
    one_hot = False

if one_hot:
    channels = 10 # onehot 1 num -> 0,0,0,1 etc
elif a.args.data == 'mnist' or a.args.feat_extractor != "none":
    channels = 1 # greyscale mnist, density maps
elif counts:
    channels = 1 # duplication not needed (linear subnets)
else:
    # TODO - this is a massive hack to test feature extractor-less  NF
    channels = 2 # duplicate dmap over channel dimension (1->2)

if debug:
    a.args.schema = 'debug' # aka ignore debugs

if a.args.data == 'dlr':
    img_size = (a.args.image_size,a.args.image_size)
elif a.args.data == 'mnist' and a.args.feat_extractor == "none":
    img_size = (28,28)
elif a.args.data == 'mnist':
    img_size = (228,228) # (28,28)
else:
    img_size = (a.args.image_size, a.args.image_size) # width, height (x-y)

raw_img_size = (800, 600)

img_dims = [3] + list(img_size) # RGB + x-y

# TODO: rename this parameter
# this effects the padding applied to the density maps
if a.args.data == 'dlr_acd':
    density_map_h,density_map_w = a.args.image_size,a.args.image_size
elif not a.args.data == 'mnist' and not counts and downsampling:
    density_map_w = a.args.image_size//scale #img_size[0]
    density_map_h = a.args.image_size//scale
    # if feat_extractor == 'resnet18':
    #     density_map_h = 256//scale #img_size[1]
    # elif feat_extractor == 'alexnet':
    #      density_map_h = 544//scale #img_size[1]
    #      density_map_w = 768//scale
    # elif feat_extractor == 'vgg16_bn':
    #     density_map_h = 576//scale #img_size[1]
    # elif feat_extractor == 'none':
    #     density_map_h = 600//scale
elif not a.args.data == 'mnist' and not counts and not downsampling:
    density_map_w = a.args.image_size//scale
    density_map_h = a.args.image_size//scale
elif not a.args.data == 'mnist':
    # size of count expanded to map spatial feature dimensions
    density_map_h = 19 * 2 # need at least 2 channels, expand x2, then downsample (haar)
    density_map_w = 25 * 2 # TODO - rework so this ties to ft_dims (vgg,alexnet, resnet, etc)
elif a.args.data == 'mnist' and a.args.feat_extractor != "none":
    if gap:
        density_map_h = img_size[1] * 2
        density_map_w = img_size[0] * 2
    else:
        density_map_h = 6 * 2
        density_map_w = 6 * 2 # feature size x2 (account for downsampling)
elif a.args.data == 'mnist' and a.args.feat_extractor == "none":
    # minimum possible dimensionality of flow possible with coupling layers
    density_map_h = 4
    density_map_w = 4

if gpu:
    device = torch.device("cuda:{}".format(a.args.gpu_number) if torch.cuda.is_available() else "cpu") # select gpu

if a.args.gpu_number != 0:
    assert gpu

if not a.args.resize: 
    img_size = (800,600)
    density_map_w,density_map_h = (800,608)

# Checks ------ 

# TODO
# assert not (pyramid and fixed1x1conv)
assert not (a.args.feat_extractor == 'none' and gap == True)
#assert gap != downsampling
assert n_splits >= 0 and n_splits < 6

if a.args.subnet_type == 'fc' and a.args.model_name == 'NF':
    assert gap
    assert a.args.data == 'mnist' or counts
    assert not a.args.fixed1x1conv

if a.args.subnet_type == 'conv':
    assert dropout_p == 0
    
if a.args.data == 'mnist':
    assert not counts

if counts and a.args.model_name == 'NF':
    assert a.args.subnet_type == 'fc' and gap or a.args.subnet_type == 'conv' and not gap

if a.args.pyramid:
    assert n_coupling_blocks == 5 # for recording purposes
    assert a.args.feat_extractor in ['resnet18','vgg16_bn','resnet50']
    assert downsampling # pyramid nf head has  downsmapling
    assert not train_feat_extractor # TODO
    # TODO - get pyramid working with other scales!
    assert scale == 1

assert scale in (1,2,4)
assert a.args.freq_1x1 != 0

# set config values for local testing
if any('SPYDER' in name for name in os.environ):
    train_model = True
    data_prop = 1
    feat_extractor_epochs = 1
