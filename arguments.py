import argparse
import socket
import os
# command line params
parser = argparse.ArgumentParser(description='Create dataloaders and train a model on MNIST, DLRACD or cow dataset.')

parser.add_argument('-mode',help="Specify mode (train,eval,store).",default='')
parser.add_argument('-get_likelihood',help='get and plot likelihoods / anamoly scores',action="store_true",default=False)
parser.add_argument('-get_grad_maps',help='dev',action="store_true",default=False)
parser.add_argument('-jac',help='enable the jacobian as part of training',action="store_true",default=False)
parser.add_argument('-freeze_bn',help='freeze batch norms',action="store_true",default=False)
parser.add_argument('-holdout',help="Use holdout data",action="store_true",default=False)

parser.add_argument('-mdl_path',help="Specify mdl for eval",default='')
parser.add_argument('-bin_classifier_path', default='')
parser.add_argument('-ram',help='Load images/dmaps/point maps into ram for faster trainig',action="store_true",default=False)

parser.add_argument('-mod',"--model_name",help="Specify model (NF,CSRNet, UNet, UCSRNetNet_seg, FCRN, LCFCN, MCNN).",default='')
parser.add_argument("-fe_only", help="Trains the feature extractor only.", action="store_true",default=False)
parser.add_argument("-bc_only", help="Trains the binary classifier only.", action="store_true",default=False)
parser.add_argument("-skip_final_eval", help="Skips final eval.", action="store_true",default=False)
parser.add_argument('-save_final_mod',help="Save final model",action="store_true",default=False)
parser.add_argument("-uc", "--unconditional", help="Trains the model without labels.", action="store_true")
parser.add_argument("-gn", "--gpu_number", help="Selects which GPU to train on.", type=int, default=0)

# Choose Dataset
parser.add_argument('-d','--data',help='Run the architecture on the [dlr,cows,mnist] dataset.',default='cows')

parser.add_argument('-sampler',help='type of sampler to use [anno,weighted,none]',default='')
parser.add_argument('-normalise',help='normalise aerial imagery supplied to model with img net mean & std dev',action='store_true',default=True)
parser.add_argument('-resize',help='resize image to the specified img size',action="store_true",default=False)
parser.add_argument('-rrc',help='perform random resize cropping',action="store_true",default=False)
parser.add_argument('-sigma',help='Variance of gaussian kernels used to create density maps',type=float,default=4.0)  # ignored for DLR ACD which uses gsd correspondence
parser.add_argument('-dmap_scaling',help='Scale up density map to ensure gaussianed density is not too close to zero per pixel',type=int,default=1)
parser.add_argument('-min_scaling',help='Minimum scaling bound (0-1) for random resized crop transform',type=float,default=-1)
parser.add_argument('-img_sz','--image_size',help='Size of the random crops taken from the original data patches [Cows 800x600, DLR 320x320] - must be divisble by 8 for CSRNet',type=int,default=256)
parser.add_argument('-max_filter_size',help='Size of max filters for unet seg and LCFCN',type=int,default=0)


parser.add_argument('-test','--test_run',help='use only a small fraction of data to check everything works',action='store_true')
# parser.add_argument("-cfile", "--config_file", help="Specify a config file that will determine training options.", type=int, default=0)
# parser.add_argument("-c", "--counts", help="Train a model that predicts only counts.", action="store_true")

parser.add_argument("-name","--schema",type=str,default='debug') # if debug, ignored
parser.add_argument("-tb","--tensorboard",help='calc and write metrics, hyper params to tb files',action="store_true",default=False)
parser.add_argument("-viz",help='visualise outputs and stats',action="store_true",default=False)
parser.add_argument("-viz_freq",help='how many epochs per viz',type=int,default=25)

# Key params
parser.add_argument("-se","--sub_epochs",help='evaluation is not performed in sub epochs',type=int,default=0)
parser.add_argument("-me","--meta_epochs",help='eval after every meta epoch. total epochs = meta*sub',type=int,default=0)
parser.add_argument("-bs","--batch_size",type=int,default=0)

# Optimiser options
parser.add_argument("-lr","--learning_rate",type=float,default=None)
parser.add_argument('-optim',help='optimizer [sgd,adam, adamw]',type=str,default='')
parser.add_argument('-adam_b1',help='adam beta1',type=float,default=0.9)
parser.add_argument('-adam_b2',help='adam beta2',type=float,default=0.999)
parser.add_argument('-adam_e',help='adam episilon',type=float,default=1e-8)
parser.add_argument('-sgd_mom',help='sgd momentum',type=float,default=0.9)
parser.add_argument("-scheduler",help="Learning rate scheduler (exponential,step,none)",default ='')
parser.add_argument("-step_size",help="step size of stepLR scheduler",type=int,default=10)
parser.add_argument("-step_gamma",help="gamma of stepLR scheduler",type=float,default=0.1)
parser.add_argument("-expon_gamma",help="gamma of expon scheduler",type=float,default=0.9)
parser.add_argument('-wd','--weight_decay',type=float,default=None) # differnet: 1e-5

# NF only options
parser.add_argument('-feat_extractor',help="type of feature extractor backbone to use in flow [resnet18, vgg16_bn,resnet50,resnet9]",default ='')
parser.add_argument("-train_classification_head",action='store_true',default=False) # whether to train classification model for filtering null patches
parser.add_argument("-all_in_one",action='store_true',default=False) # whether to use all in blocks (inc act norm)
parser.add_argument("-pyramid",action='store_true',default=False) # whether to a feature pyramid for conditioning - only implemented for resnet18
parser.add_argument("-fixed1x1conv",action='store_true',default=False) # whether to use 1x1 convs
parser.add_argument("-freq_1x1",type=int,default=1) # 1 for always | how many x coupling blocks to have a 1x1 conv permutation layer
parser.add_argument("-npb","--n_pyramid_blocks",type=int,default=0)
parser.add_argument('-nse',"--noise",help='amount of uniform noise (sample evenly from 0-x) | 0 for none',type=float,default=0)
parser.add_argument('-f','--filters',help='width of conv subnetworks',type=int,default=0)
parser.add_argument("-split", "--split_dimensions", help="split off half the dimensions after each block of coupling layers.", type=int, default=0)
parser.add_argument("-subnet_type",help="type of subnet to use in flow [fc,conv,MCNN,UNet,conv_shallow,conv_deep]",default ='')
parser.add_argument("-batch_norm",help="Add batchnorm to subnets",action="store_true",default=False)

parser.add_argument("-fe_load_imagenet_weights",help="load pt weights into FE",action='store_true',default=False)
parser.add_argument('-fe_b1',help='fe_adam beta1',type=float,default=0.9)
parser.add_argument('-fe_b2',help='fe adam beta2',type=float,default=0.999)
parser.add_argument("-fe_lr",help="fe LR",type=float,default=1e-3)
parser.add_argument("-fe_wd",help="fe wd",type=float,default=1e-5)

# weighted: weight minibatch samples such that sampling distribution is 50/50 null/annotated
# anno: use only annotated samples

# to implement functions
# parser.add_argument("-eval_image",'--img',help="Run the model on the image specified (requires model loaded)",type=str,default='')
# parser.add_argument("-load_model",'--load',help="load the model (from path) for evaluation, inference, or visualisation",type=str,default='')

global args 
args = parser.parse_args()
host = socket.gethostname()

# defaults for if running interactively
if any('SPYDER' in name for name in os.environ):
    args.model_name = "CSRNet"
    args.data = 'cows'
    args.optim = "adamw"
    args.scheduler = 'none'
    args.sampler = 'none'
    args.mode = 'train' #'eval'
    args.sub_epochs = 1
    args.meta_epochs = 1
    args.batch_size = 2
    args.learning_rate = 1e-3
    args.weight_decay = 1e-8
    args.tensorboard = False
    args.viz = True
    args.viz_freq = 100
    args.resize = True
    args.rrc = False
    args.dmap_scaling = 1
    args.max_filter_size = 4
    args.sigma = 4.0
    args.mdl_path = '' # 'final_9Z5_NF_quatern_BS64_LR_I0.0002_E10000_DIM256_OPTIMadam_FE_resnet18_NC5_anno_step_JO_PY_1_1x1_WD_0.001_10_05_2022_17_37_42'
    args.holdout = True
    args.all_in_one = False
    args.fixed1x1conv = False
    args.split_dimensions = 0
    
    args.subnet_type = ''
    args.noise = 0
    args.filters = 0
    args.n_pyramid_blocks = 0
    args.skip_final_eval = False
    args.feat_extractor = ''
    args.pyramid = False
    args.tb = False
    
    args.image_size = 256

    args.expon_gamma = 0.99
    args.adam_b1 = 0.9
    args.adam_b2 = 0.999
    args.ram = False
    args.save_final_mod = True
    
    args.fe_lr = 1e-3
    args.fe_b1 = 0.9
    args.fe_b2 = 0.999
    args.fe_wd = 1e-8
    
    args.bc_only = False
    
# checks
assert args.mode in ['train','eval','store','plot']
assert args.gpu_number > -1



if args.mode == 'store':
    assert args.ram

if args.split_dimensions:
    assert not args.all_in_one

if args.jac:
    assert args.model_name == "NF"

if args.mode == 'eval':
    assert args.mdl_path != ''
    #assert args.sampler=='none'

if args.holdout:
    assert args.data == 'cows'

assert args.sampler in ['weighted','anno','none']

if args.rrc:
    #assert args.min_scaling > 0 and args.min_scaling < 1
    assert not args.resize

if (args.step_size != 0 or args.step_gamma != 0) and args.scheduler != 'step':
    ValueError

if (args.adam_b1 != 0 or args.adam_b2 != 0 or args.adam_e != 0) and args.optim != 'adam':
    ValueError
    
if args.sgd_mom != 0 and args.optim != 'sgd':
    ValueError

if args.split_dimensions:
    assert args.model_name == 'NF'
    assert not args.fixed1x1conv

if args.all_in_one:
    assert args.model_name == 'NF'
    assert args.fixed1x1conv

if args.model_name != 'NF' and args.mode == 'train':
    assert args.bin_classifier_path == ''
    assert not args.all_in_one
    assert not args.train_classification_head
    assert args.noise == 0
    assert args.filters == 0
    assert args.n_pyramid_blocks == 0
    assert args.subnet_type == ""

if args.model_name == 'NF' and args.mode == 'train':
    assert args.noise != 0
    assert args.n_pyramid_blocks != 0
    assert args.subnet_type != ''
    
    if args.pyramid:
        assert args.feat_extractor in ['resnet18','vgg16_bn','resnet50','resnet9']
    else:
        assert args.feat_extractor in ['alexnet', 'vgg16_bn','resnet18', 'none']
elif args.model_name != 'NF':
    assert args.feat_extractor == ''
    
if args.subnet_type in ['conv','conv_shallow','conv_deep']:
    assert args.filters != 0 # this argument only applies to regular conv subnets
else:
    assert args.filters == 0

if args.fixed1x1conv and args.pyramid:
    assert args.freq_1x1 == 1

if args.feat_extractor == 'resnet50':
    assert args.pyramid

if args.model_name == 'LCFCN': #  in ['UNet_seg','LCFCN']:
    assert args.batch_size == 1 # https://github.com/ElementAI/LCFCN/issues/9
else:
    assert args.batch_size >= 1
    # LCFCN only supports batch size of 1

if args.model_name in ['UNet_seg','LCFCN']:
    assert args.max_filter_size >= 1
    assert args.max_filter_size == args.sigma

if args.mode == 'train':
    assert args.meta_epochs and args.sub_epochs >= 1
    assert args.scheduler in ['exponential','step','none']
    assert args.optim in ['sgd','adam','adamw']

# todo - find better way of checking NF only argument

assert args.data in ['cows','dlr','mnist']

if args.data == "dlr":
    assert args.image_size == 320

if host == 'hydra':
     assert args.gpu_number < 8
elif host == 'quartet':
    assert args.gpu_number < 4
elif host == 'quatern' or host == 'deuce':
    assert args.gpu_number < 2
else:
    assert args.gpu_number < 1 
    
