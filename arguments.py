# command line params
import argparse
parser = argparse.ArgumentParser(description='Create dataloaders and train CowFlow or MNISTFlow conditional NF.')
parser.add_argument("-fe_only", "--feat_extract_only", help="Trains the feature extractor component only.", action="store_true")
parser.add_argument("-uc", "--unconditional", help="Trains the feature extractor component only.", action="store_true")
#parser.add_argument("-c", "--counts", help="Train a model that predicts only counts.", action="store_true")

global args 
args = parser.parse_args()