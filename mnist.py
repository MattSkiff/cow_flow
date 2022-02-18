# This file starts the training on the MNIST dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from utils import AddUniformNoise #, AddGaussianNoise
from torch.cuda import empty_cache
from torch.utils.data import DataLoader # Dataset                                                                                                                                                                    
# from torchvision import transforms
import config as c
from train import train, train_battery

from torch.utils.data.sampler import SubsetRandomSampler # RandomSampling

empty_cache() # free up memory for cuda

def train_mnist(load_only=False):

    mnist_pre = Compose([
        ToTensor(),
        AddUniformNoise(),
        Resize((c.img_size[0], c.img_size[0])),
        Normalize((0.1307,), (0.3081,))
        ])
    
    mnist_train = MNIST(root='./data', train=True, download=True, transform=mnist_pre)
    mnist_test = MNIST(root='./data', train=False, download=True, transform=mnist_pre)
    
    if c.test_run:
        toy_sampler = SubsetRandomSampler(range(200))
    else:
        toy_sampler = None
    
    if len(c.batch_size) == 1:
        train_loader = DataLoader(mnist_train,batch_size = c.batch_size[0],pin_memory=True,
                                      shuffle=False,sampler=toy_sampler)
        val_loader = DataLoader(mnist_test,batch_size = c.batch_size[0],pin_memory=True,
                                      shuffle=False,sampler=toy_sampler)
        if len(c.lr_init) == 1:
            mdl = train(train_loader,val_loader,lr_i=c.lr_init)
        else:
            mdl = train_battery([train_loader],[val_loader],lr_i=c.lr_init)
                
    else:
        tls,vls = [],[]
        
        for bs in c.batch_size:
            tls.append(DataLoader(mnist_train,batch_size = bs,pin_memory=True,
                                      shuffle=False,sampler=toy_sampler))
            vls.append(DataLoader(mnist_test,batch_size = bs,pin_memory=True,
                                      shuffle=False,sampler=toy_sampler))
            
            if not load_only:
                mdl = train_battery(tls,vls,lr_i=c.lr_init)
            else:
                mdl = None
    
    return mdl, train_loader, val_loader
