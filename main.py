# This file starts the training
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from utils import AddUniformNoise #, AddGaussianNoise
from torch.cuda import empty_cache
from torch.utils.data import DataLoader # Dataset                                                                                                                                                                    
from torch.utils.data.sampler import SubsetRandomSampler # RandomSampling
# from torchvision import transforms
import config as c
import arguments as a
import model
import os
import pickle 
from train import train, train_battery
#from utils import load_datasets, make_dataloaders
from data_loader import CowObjectsDataset, CustToTensor,AerialNormalize, DmapAddUniformNoise, CustCrop, CustResize, train_valid_split

empty_cache() # free up memory for cuda

# torchivsion inputs are 3x227x227, mnist_resnet 1x227...
# 0.1307, 0.3081 = mean, std dev mnist
mnist_pre = Compose([
    ToTensor(),
    AddUniformNoise(),
    Resize((c.img_size[0], c.img_size[0])),
    Normalize((0.1307,), (0.3081,))
    ])

dmaps_pre = Compose([
            CustToTensor(),
            DmapAddUniformNoise(),
            AerialNormalize(),
            CustCrop(),
            CustResize()
        ])

if c.mnist:
    mnist_train = MNIST(root='./data', train=True, download=True, transform=mnist_pre)
    mnist_test = MNIST(root='./data', train=False, download=True, transform=mnist_pre)
    
    if c.test_run:
        toy_sampler = SubsetRandomSampler(range(200))
    else:
        toy_sampler = None
    
    if len(c.batch_size) == 1:
        train_loader = DataLoader(mnist_train,batch_size = c.batch_size[0],pin_memory=True,
                                      shuffle=False,sampler=toy_sampler)
        valid_loader = DataLoader(mnist_test,batch_size = c.batch_size[0],pin_memory=True,
                                      shuffle=False,sampler=toy_sampler)
        if len(c.lr_init) == 1:
                mdl = train(train_loader,valid_loader,lr_i=c.lr_init)
        else:
                mdl = train_battery([train_loader],[valid_loader],lr_i=c.lr_init)
                
    else:
        tls,vls = [],[]
        
        for bs in c.batch_size:
            tls.append(DataLoader(mnist_train,batch_size = bs,pin_memory=True,
                                      shuffle=False,sampler=toy_sampler))
            vls.append(DataLoader(mnist_test,batch_size = bs,pin_memory=True,
                                      shuffle=False,sampler=toy_sampler))
            
            mdl = train_battery(tls,vls,lr_i=c.lr_init)
    
   
else:
    # instantiate class
    transformed_dataset = CowObjectsDataset(root_dir=c.proj_dir,transform = dmaps_pre,
                                            convert_to_points=True,generate_density=True,
                                            count = c.counts,classification = c.train_feat_extractor)
    
    # check dataloader if running interactively
    if any('SPYDER' in name for name in os.environ):
        transformed_dataset.show_annotations(5809)
    
    # create test train split
    # https://stackoverflow.com/questions/27745500/how-to-save-a-list-to-a-file-and-read-it-as-a-list-type
    if not c.fixed_indices:
        train_indices, valid_indices = train_valid_split(dataset = transformed_dataset, 
                                                         train_percent = c.test_train_split,
                                                         annotations_only = c.annotations_only,
                                                         balanced = c.balanced)
        
        with open("train_indices.txt", "wb") as fp:   # Pickling
            pickle.dump(train_indices, fp)
        with open("valid_indices.txt", "wb") as fp:   # Pickling
            pickle.dump(valid_indices, fp)

    else:
        with open("train_indices.txt", "rb") as fp:   # Unpickling
            train_indices = pickle.load(fp)
        with open("valid_indices.txt", "rb") as fp:   # Unpickling
            valid_indices = pickle.load(fp)
    
    # Creating data samplers and loaders:
    # only train part for dev purposes 
    
    if not c.annotations_only:
        train_sampler = SubsetRandomSampler(train_indices[:round(c.data_prop*len(train_indices))])
        valid_sampler = SubsetRandomSampler(valid_indices[:round(c.data_prop*len(valid_indices))])
    
    if c.annotations_only:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        
    if len(c.batch_size) != 1 or len(c.lr_init) != 1 and a.args.feat_extract_only:
        ValueError('Training batteries not available for Feature Extractor only runs')
        
    
    if len(c.batch_size) == 1:
        train_loader = DataLoader(transformed_dataset, batch_size=c.batch_size[0],shuffle=False, 
                            num_workers=0,collate_fn=transformed_dataset.custom_collate_aerial,
                            pin_memory=True,sampler=train_sampler)
    
        valid_loader = DataLoader(transformed_dataset, batch_size=c.batch_size[0],shuffle=False, 
                            num_workers=0,collate_fn=transformed_dataset.custom_collate_aerial,
                            pin_memory=True,sampler=valid_sampler)
        
        if len(c.lr_init) == 1:
            if a.args.feat_extract_only:
                feat_extractor = model.select_feat_extractor(c.feat_extractor,train_loader,valid_loader)
            else:
                mdl = train(train_loader,valid_loader,lr_i=c.lr_init)
        else:
                mdl = train_battery([train_loader],[valid_loader],lr_i=c.lr_init)
                
    else:
        tls,vls = [],[]
        
        for bs in c.batch_size:
            tls.append(DataLoader(transformed_dataset, batch_size=bs,shuffle=False, 
                            num_workers=0,collate_fn=transformed_dataset.custom_collate_aerial,
                            pin_memory=True,sampler=train_sampler))
            
            vls.append(DataLoader(transformed_dataset, batch_size=bs,shuffle=False, 
                            num_workers=0,collate_fn=transformed_dataset.custom_collate_density,
                            pin_memory=True,sampler=valid_sampler))
            
            mdl = train_battery(tls,vls,lr_i=c.lr_init)