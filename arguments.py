import argparse
import socket

# command line params
parser = argparse.ArgumentParser(description='Create dataloaders and train a model on MNIST, DLRACD or cow dataset.')

parser.add_argument('-mod',"--model_name",help="Specify model to train (NF, CSRNet, UNet, UNet_seg,FCRN, LCFCN).",default='NF')
parser.add_argument("-scheduler",help="Learning rate scheduler (exponential,step,none)",default ='exponential')
parser.add_argument('-optim',help='optimizer',type=str,default='adam')

parser.add_argument('-anno','--annotations_only',help='only use image patches that have annotations',action="store_true",default=True)
parser.add_argument('-weighted','--weighted_sampler',help='weight minibatch samples such that sampling distribution is 50/50 null/annotated', action="store_true",default=False)

####
parser.add_argument("-fe_only", "--feat_extract_only", help="Trains the feature extractor component only.", action="store_true",default=False)

# Choose Dataset
parser.add_argument('-d','--data',help='Run the architecture on the [dlr,cows] dataset.',default='cows')

parser.add_argument('-normalise',help='normalise aerial imagery supplied to model with img net mean & std dev',action='store_true',default=True)
parser.add_argument('-resize',help='resize image to the specified img size',action="store_true",default=False)
parser.add_argument('-rrc',help='perform random resize cropping',action="store_true",default=False)
parser.add_argument('-dmap_scaling',help='Scale up density map to ensure gaussianed density is not too close to zero per pixel',type=int,default=1)
parser.add_argument('-min_scaling',help='Minimum scaling bound (0-1) for random resized crop transform',type=float,default=-1)
parser.add_argument('-img_sz','--image_size',help='Size of the random crops taken from the original data patches [Cows 800x600, DLR 320x320] - must be divisble by 8 for CSRNet',type=int,default=256)

parser.add_argument("-name","--schema",type=str,default='debug') # if debug, ignored
parser.add_argument("-tb","--tensorboard",help='calc and write metrics, hyper params to tb files',action="store_true",default=False)

# Key params
parser.add_argument("-se","--sub_epochs",help='evaluation is not performed in sub epochs',type=int,default=1)
parser.add_argument("-me","--meta_epochs",help='eval after every meta epoch. total epochs = meta*sub',type=int,default=1)

# LR scheduler params
parser.add_argument("-step_size",help="step size of stepLR scheduler",type=int,default=10)
parser.add_argument("-step_gamma",help="gamma of stepLR scheduler",type=float,default=0.1)
parser.add_argument("-ex_gamma",help="gamma of expon scheduler",type=int,default=0.9)

# Optim params
parser.add_argument("-lr","--learning_rate",type=float,default=1e-3)
parser.add_argument("-bs","--batch_size",type=int,default=1)
parser.add_argument('-adam_b1',help='adam beta1',type=float,default=0.9)
parser.add_argument('-adam_b2',help='adam beta2',type=float,default=0.999)
parser.add_argument('-adam_e',help='adam episilon',type=float,default=1e-8)
parser.add_argument('-sgd_mom',help='sgd momentum',type=float,default=0.9)
parser.add_argument('-wd','--weight_decay',type=float,default=1e-3) # differnet: 1e-5

# NF params
parser.add_argument("-npb","--n_pyramid_blocks",type=int,default=3)
parser.add_argument('-nse',"--noise",help='amount of uniform noise (sample evenly from 0-x) | 0 for none',type=float,default=0)
parser.add_argument('-f','--filters',help='width of conv subnetworks',type=int,default=32)

global args 
args = parser.parse_args()
host = socket.gethostname()

if args.rrc:
    assert args.min_scaling > 0 and args.min_scaling < 1

if (args.step_size != 0 or args.step_gamma != 0) and args.scheduler != 'step':
    ValueError

if (args.adam_b1 != 0 or args.adam_b2 != 0 or args.adam_e != 0) and args.optim != 'adam':
    ValueError
    
if args.sgd_mom != 0 and args.optim != 'sgd':
    ValueError
  
assert args.model_name in ['NF','UNet','CSRNet','FCRN','LCFCN','UNet_seg']

if args.model_name == 'LCFCN':
    assert args.batch_size == 1 # https://github.com/ElementAI/LCFCN/issues/9
    # LCFCN only supports batch size of 1

assert not (args.weighted_sampler and args.annotations_only)
assert args.weighted_sampler or args.annotations_only
assert args.scheduler in ['exponential','step','none']
assert args.optim in ['sgd','adam']

# todo - find better way of checking NF only arguments
if args.model_name in ['UNet','CSRNet','FCRN','LCFCN','UNet_seg']:
    assert args.noise == 0


assert args.data in ['cows','dlr']
    
