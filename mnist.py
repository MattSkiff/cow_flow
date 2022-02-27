# This file starts the training on the MNIST dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from utils import AddUniformNoise #, AddGaussianNoise
from torch.cuda import empty_cache
from torch.utils.data import DataLoader # Dataset                                                                                                                                                                    
# from torchvision import transforms
import config as c
import arguments as a
from train import train

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
    
    train_loader = DataLoader(mnist_train,batch_size = a.args.batch_size,pin_memory=True,
                                      shuffle=False,sampler=toy_sampler)
    val_loader = DataLoader(mnist_test,batch_size = a.args.batch_size,pin_memory=True,
                                      shuffle=False,sampler=toy_sampler)
    mdl = train(train_loader,val_loader)
    
    return mdl, train_loader, val_loader
