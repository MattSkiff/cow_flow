import argparse
import socket

# command line params
parser = argparse.ArgumentParser(description='Create dataloaders and train CowFlow or MNISTFlow conditional NF.')
parser.add_argument("-fe_only", "--feat_extract_only", help="Trains the feature extractor component only.", action="store_true")
parser.add_argument("-uc", "--unconditional", help="Trains the model without labels.", action="store_true")
parser.add_argument("-gn", "--gpu_number", help="Selects which GPU to train on.", type=int, default=0)
parser.add_argument("-dlr", "--dlr_acd", help="Run the architecture on the DLR ACD dataset.", type=int, default=1)
parser.add_argument("-split", "--split_dimensions", help="Whether to split off half the dimensions after each block of coupling layers.", type=int, default=1)
# parser.add_argument("-cfile", "--config_file", help="Specify a config file that will determine training options.", type=int, default=0)
# parser.add_argument("-c", "--counts", help="Train a model that predicts only counts.", action="store_true")

# to implement functions
parser.add_argument("-eval_image",'--img',help="Run the model on the image specified (requires model loaded)",type=str,default='')
parser.add_argument("-load_model",'--load',help="load the model (from path) for evaluation, inference, or visualisation",type=str,default='')

global args 
args = parser.parse_args()
host = socket.gethostname()

assert args.gpu_number > -1

if host == 'hydra':
     assert args.gpu_number < 8
elif host == 'quartet':
    assert args.gpu_number < 4
elif host == 'quatern' or host == 'deuce':
    assert args.gpu_number < 2
else:
    assert args.gpu_number < 1 
    