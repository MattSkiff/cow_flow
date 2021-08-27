import config as c
import torch
from pyro import distributions as T

def preprocess_batch(data):
    '''move data to device and reshape image'''
    images,annotations,classes = data
    images,annotations,classes = images.to(c.device), torch.cat(annotations,dim = 0).to(c.device), classes #annotations.to(c.device),classes.to(c.device)
    images = images.view(-1, *images.shape[-3:])
    return images, annotations, classes

def gaussianise_annotations(annotations):
    
    for annotation in annotations:
        print(annotations)
    
