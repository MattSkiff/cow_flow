from model import load_model
from os.path import join
from PIL import Image
import config as c
import torch
def eval(model_name,image_folder):
    
    model = load_model(model_name)
    print(model)
    
    return(model)
    
image_folder = 'test'
eval(c.modelname,image_folder)