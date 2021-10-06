from torch import randn
from tqdm import tqdm
import numpy as np
import config as c
import torch

# TODO extra args: plot = True, save=True,hist=True
# TODO: don't shift computation over to cpu after sampling from model
def eval_mnist(model, validloader, trainloader,samples = 1): 
    
    model.eval()
    
    for loader in [validloader,trainloader]:
        
        tally = 0
        accuracies = []
    
        for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):

            if c.mnist:
                images,labels = data
            else:
                images,dmaps,labels = data
            
        
            if labels.size: # triggers only if there is at least one annotation
                
                # Z shape: torch.Size([2, 4, 300, 400]) (batch size = 2)
                # channels * 4 is a result of haar downsampling
                dummy_z = (randn(loader.batch_size, c.channels*4 , c.density_map_h // 2,c.density_map_w  // 2, requires_grad=True)).to(c.device)
                
                images = images.float().to(c.device)
                dummy_z = dummy_z.float().to(c.device)
                
                model = model.to(c.device)
                
                mean_array = []
                
                for i in range(samples):
                    x, _ = model(images,dummy_z,rev=True) # TODO: investigate eval use for log det j?
                    
                    if c.one_hot:
                        x = x.argmax(-3).to(torch.float)
                     
                    dims = (1,2,3)
                     
                    if c.one_hot:
                        dims = (1,2)
                        
                    mean_preds = np.round(x.mean(dim = dims).cpu().detach().numpy())
                    mean_array.append(mean_preds)
                
                # https://stackoverflow.com/questions/48199077/elementwise-aggregation-average-of-values-in-a-list-of-numpy-arrays-with-same
                mean_preds = np.mean(mean_array,axis=0)
                    
                labels = labels.cpu().detach().numpy()
                
                tally += (labels == mean_preds).sum()
                
#                if plot:
#                    pass
                
            accuracies.append(round(tally/(len(loader.dataset))*100,2))

    return  accuracies[0], accuracies[1] # train, valid

def eval_model(model, validloader, trainloader, plot = True, save=True,hist=True):
    
    return 0,0
