# External
from torch.cuda import empty_cache
from torchvision.transforms import Compose
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader # Dataset   

# Internal
from train import train     
import os                                                                                                                                                            
import config as c
from utils import load_model
import arguments as a
from eval import dmap_metrics
from data_loader import DLRACD, DLRACDToTensor, DLRACDAddUniformNoise,DLRACDCropRotateFlipScaling

def dlr_acd(load_only=False):    
    empty_cache() # free up memory for cuda
    
    dlracd_pre = Compose([
        DLRACDToTensor(),
        DLRACDAddUniformNoise(),
        # DLRACDDoubleCrop
        DLRACDCropRotateFlipScaling(),
        ])
    
    dlr_dataset = DLRACD(root_dir=c.proj_dir,transform = dlracd_pre,overlap=0.5)
    
    t_indices, v_indices  = dlr_dataset.train_indices,dlr_dataset.test_indices
    
    if a.args.test_run:
        t_indices = range(1000)
        v_indices = range(1000)
    
    train_sampler = SubsetRandomSampler(t_indices)
    val_sampler = SubsetRandomSampler(v_indices)
    
    train_loader = DataLoader(dlr_dataset, batch_size=a.args.batch_size,shuffle=False, 
                            num_workers=1,collate_fn=dlr_dataset.custom_collate_aerial,
                            pin_memory=False,sampler=train_sampler)
    
    val_loader = DataLoader(dlr_dataset, batch_size=a.args.batch_size,shuffle=False, 
                            num_workers=1,collate_fn=dlr_dataset.custom_collate_aerial,
                            pin_memory=False,sampler=val_sampler)
    
    if a.args.mode == 'eval':
        
            mdl = load_model(a.args.mdl_path)
            dmap_metrics(mdl,val_loader,mode='val',n=1)
    
    if any('SPYDER' in name for name in os.environ):
        dlr_dataset.show_annotations(1) #2345
    
    if not load_only and a.args.mode == train:
        mdl = train(train_loader,val_loader)
    else:
        mdl = None
    
    return mdl, train_loader, val_loader