import argparse
import socket

# command line params
parser = argparse.ArgumentParser(description='Create dataloaders and train a conditional NF.')

parser.add_argument('-mod',"--model_name",help="Specify model to train (NF, CSRNet, UNet, FCRN).",default='CSRNet')
parser.add_argument("-fe_only", "--feat_extract_only", help="Trains the feature extractor component only.", action="store_true")
parser.add_argument("-uc", "--unconditional", help="Trains the model without labels.", action="store_true")
parser.add_argument("-gn", "--gpu_number", help="Selects which GPU to train on.", type=int, default=0)

parser.add_argument("-dlr", "--dlr_acd", help="Run the architecture on the DLR ACD dataset.", action="store_true",default=False)
parser.add_argument("-cows", "--cows", help="Run the architecture on the aerial cows dataset.", action="store_true",default=True)
parser.add_argument('-mnist',help='Run the architecture on the DLR ACD dataset.', action="store_true",default=False)

parser.add_argument('-test','--test_run',help='use only a small fraction of data to check everything works',action='store_true')
parser.add_argument("-split", "--split_dimensions", help="Whether to split off half the dimensions after each block of coupling layers.", type=int, default=0)
# parser.add_argument("-cfile", "--config_file", help="Specify a config file that will determine training options.", type=int, default=0)
# parser.add_argument("-c", "--counts", help="Train a model that predicts only counts.", action="store_true")

parser.add_argument("-name","--schema",type=str,default='debug') # if debug, ignored
parser.add_argument("-tb","--tensorboard",help='calc and write metrics, hyper params to tb files',action="store_true")

# Key params
parser.add_argument("-se","--sub_epochs",help='evaluation is not performed in sub epochs',type=int,default=1)
parser.add_argument("-me","--meta_epochs",help='eval after every meta epoch. total epochs = meta*sub',type=int,default=1)
parser.add_argument("-lr","--learning_rate",type=float,default=2e-3)
parser.add_argument("-bs","--batch_size",type=int,default=2)
parser.add_argument("-npb","--n_pyramid_blocks",type=int,default=3)
parser.add_argument('-nse',"--noise",help='amount of uniform noise (sample evenly from 0-x) | 0 for none',type=float,default=1e-3)
parser.add_argument('-f','--filters',help='width of conv subnetworks',type=int,default=32)
parser.add_argument('-wd','--weight_decay',type=float,default=1e-3) # differnet: 1e-5

# to implement functions
# parser.add_argument("-eval_image",'--img',help="Run the model on the image specified (requires model loaded)",type=str,default='')
# parser.add_argument("-load_model",'--load',help="load the model (from path) for evaluation, inference, or visualisation",type=str,default='')

global args 
args = parser.parse_args()
host = socket.gethostname()

assert args.gpu_number > -1
assert args.model_name in ['NF','UNet','CSRNet',]
assert args.cows+args.dlr_acd+args.mnist == 1

if host == 'hydra':
     assert args.gpu_number < 8
elif host == 'quartet':
    assert args.gpu_number < 4
elif host == 'quatern' or host == 'deuce':
    assert args.gpu_number < 2
else:
    assert args.gpu_number < 1 
    