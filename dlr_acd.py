# This file starts the training on the DLR ACD dataset
from torch.cuda import empty_cache
from torchvision.transforms import Compose
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader # Dataset   
from train import train                                                                                                                                                                 
import config as c

#from utils import load_datasets, make_dataloaders
from data_loader import DLRACD, DLRACDToTensor, DLRACDAddUniformNoise

empty_cache() # free up memory for cuda

dlracd_pre = Compose([
    DLRACDToTensor(),
    DLRACDAddUniformNoise(),
    # DLRACDDoubleCrop
    # DLRACDRotateFlipScaling
    ])

dlr_dataset = DLRACD(root_dir=c.proj_dir,transform = dlracd_pre)

t_indices, v_indices  = dlr_dataset.train_indices,dlr_dataset.test_indices

train_sampler = SubsetRandomSampler(t_indices)
val_sampler = SubsetRandomSampler(v_indices)   

train_loader = DataLoader(dlr_dataset, batch_size=c.batch_size[0],shuffle=False, 
                        num_workers=0,collate_fn=dlr_dataset.custom_collate_aerial,
                        pin_memory=False,sampler=train_sampler)

val_loader = DataLoader(dlr_dataset, batch_size=c.batch_size[0],shuffle=False, 
                        num_workers=0,collate_fn=dlr_dataset.custom_collate_aerial,
                        pin_memory=False,sampler=val_sampler)

dlr_dataset.show_annotations(2345)

mdl = train(train_loader,val_loader,lr_i=c.lr_init)