import config as c
import torch
import numpy as np
from pyro import distributions as T

def preprocess_batch(data):
    '''move data to device and reshape image'''
    images,annotations,classes = data
    images,annotations,classes = images.to(c.device), torch.cat(annotations,dim = 0).to(c.device), classes #annotations.to(c.device),classes.to(c.device)
    images = images.view(-1, *images.shape[-3:])
    return images, annotations, classes

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gaussianise_annotations(annotations):
    
    for annotation in annotations:
        print(annotations)
    
