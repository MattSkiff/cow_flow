import argparse
import socket
import os
# command line params
parser = argparse.ArgumentParser(description='Create dataloaders and train a model on MNIST, DLRACD or cow dataset.')

parser.add_argument('-mod',"--model_name",help="Specify model to train (NF,CSRNet, UNet, UCSRNetNet_seg, FCRN, LCFCN, MCNN).",default='')
parser.add_argument("-fe_only", "--feat_extract_only", help="Trains the feature extractor component only.", action="store_true",default=False)
parser.add_argument("-uc", "--unconditional", help="Trains the model without labels.", action="store_true")
parser.add_argument("-gn", "--gpu_number", help="Selects which GPU to train on.", type=int, default=0)

# Choose Dataset
parser.add_argument('-d','--data',help='Run the architecture on the [dlr,cows,mnist] dataset.',default='cows')

parser.add_argument('-anno','--annotations_only',help='only use image patches that have annotations',action="store_true",default=False)
parser.add_argument('-weighted','--weighted_sampler',help='weight minibatch samples such that sampling distribution is 50/50 null/annotated', action="store_true",default=False)
parser.add_argument('-normalise',help='normalise aerial imagery supplied to model with img net mean & std dev',action='store_true',default=True)
parser.add_argument('-resize',help='resize image to the specified img size',action="store_true",default=False)
parser.add_argument('-rrc',help='perform random resize cropping',action="store_true",default=False)
parser.add_argument('-sigma',help='Variance of gaussian kernels used to create density maps',type=float,default=4)  # ignored for DLR ACD which uses gsd correspondence
parser.add_argument('-dmap_scaling',help='Scale up density map to ensure gaussianed density is not too close to zero per pixel',type=int,default=1)
parser.add_argument('-min_scaling',help='Minimum scaling bound (0-1) for random resized crop transform',type=float,default=-1)
parser.add_argument('-img_sz','--image_size',help='Size of the random crops taken from the original data patches [Cows 800x600, DLR 320x320] - must be divisble by 8 for CSRNet',type=int,default=256)
parser.add_argument('-max_filter_size',help='Size of max filters for unet seg and LCFCN',type=int,default=0)


parser.add_argument('-test','--test_run',help='use only a small fraction of data to check everything works',action='store_true')
parser.add_argument("-split", "--split_dimensions", help="split off half the dimensions after each block of coupling layers.", type=int, default=0)
# parser.add_argument("-cfile", "--config_file", help="Specify a config file that will determine training options.", type=int, default=0)
# parser.add_argument("-c", "--counts", help="Train a model that predicts only counts.", action="store_true")

parser.add_argument("-name","--schema",type=str,default='debug') # if debug, ignored
parser.add_argument("-tb","--tensorboard",help='calc and write metrics, hyper params to tb files',action="store_true",default=False)
parser.add_argument("-viz",help='visualise outputs and stats',action="store_true",default=False)
parser.add_argument("-viz_freq",help='how many epochs per viz',type=int,default=25)

# Key params
parser.add_argument("-se","--sub_epochs",help='evaluation is not performed in sub epochs',type=int,default=0)
parser.add_argument("-me","--meta_epochs",help='eval after every meta epoch. total epochs = meta*sub',type=int,default=0)
parser.add_argument("-scheduler",help="Learning rate scheduler (exponential,step,none)",default ='')
parser.add_argument("-step_size",help="step size of stepLR scheduler",type=int,default=10)
parser.add_argument("-step_gamma",help="gamma of stepLR scheduler",type=float,default=0.1)
parser.add_argument("-expon_gamma",help="gamma of expon scheduler",type=float,default=0.9)


parser.add_argument("-lr","--learning_rate",type=float,default=None)
parser.add_argument("-bs","--batch_size",type=int,default=0)
parser.add_argument('-optim',help='optimizer [sgd,adam]',type=str,default='')
parser.add_argument('-adam_b1',help='adam beta1',type=float,default=0.9)
parser.add_argument('-adam_b2',help='adam beta2',type=float,default=0.999)
parser.add_argument('-adam_e',help='adam episilon',type=float,default=1e-8)
parser.add_argument('-sgd_mom',help='sgd momentum',type=float,default=0.9)

parser.add_argument("-npb","--n_pyramid_blocks",type=int,default=3)
parser.add_argument('-nse',"--noise",help='amount of uniform noise (sample evenly from 0-x) | 0 for none',type=float,default=0)
parser.add_argument('-f','--filters',help='width of conv subnetworks',type=int,default=32)
parser.add_argument('-wd','--weight_decay',type=float,default=None) # differnet: 1e-5

# to implement functions
# parser.add_argument("-eval_image",'--img',help="Run the model on the image specified (requires model loaded)",type=str,default='')
# parser.add_argument("-load_model",'--load',help="load the model (from path) for evaluation, inference, or visualisation",type=str,default='')

global args 
args = parser.parse_args()
host = socket.gethostname()

# defaults for if running interactively
if any('SPYDER' in name for name in os.environ):
    args.model_name = "UNet"
    args.optim = "adam"
    args.scheduler = 'none'
    args.annotations_only = False
    args.weighted_sampler = True
    args.sub_epochs = 5
    args.meta_epochs = 1
    args.batch_size = 8
    args.learning_rate = 1e-3
    args.weight_decay = 1e-5
    args.tensorboard = True
    args.viz = True
    args.viz_freq = 1
    args.resize = True
    args.dmap_scaling = 1000
    args.max_filter_size = 4.0
    args.sigma = 4.0
    
# checks
assert args.gpu_number > -1

if args.rrc:
    assert args.min_scaling > 0 and args.min_scaling < 1

if (args.step_size != 0 or args.step_gamma != 0) and args.scheduler != 'step':
    ValueError

if (args.adam_b1 != 0 or args.adam_b2 != 0 or args.adam_e != 0) and args.optim != 'adam':
    ValueError
    
if args.sgd_mom != 0 and args.optim != 'sgd':
    ValueError

if args.model_name == 'LCFCN':
    assert args.batch_size == 1 # https://github.com/ElementAI/LCFCN/issues/9
    # LCFCN only supports batch size of 1

if args.model_name in ['UNet_seg','LCFCN']:
    assert args.max_filter_size >= 1
    assert args.max_filter_size == args.sigma

assert not (args.weighted_sampler and args.annotations_only)
assert args.weighted_sampler or args.annotations_only
assert args.scheduler in ['exponential','step','none']
assert args.optim in ['sgd','adam']

# todo - find better way of checking NF only argument

assert args.data in ['cows','dlr','mnist']

if host == 'hydra':
     assert args.gpu_number < 8
elif host == 'quartet':
    assert args.gpu_number < 4
elif host == 'quatern' or host == 'deuce':
    assert args.gpu_number < 2
else:
    assert args.gpu_number < 1 