from torch import randn
from tqdm import tqdm
import config as c
import numpy as np
import torch

# TODO extra args: plot = True, save=True,hist=True
# TODO: don't shift computation over to cpu after sampling from model
def eval_mnist(model, validloader, trainloader,samples = 1,confusion = False, preds = False): 
    ''' currently the confusion matrix is calculated across both train and validation splits'''
    model.eval()
    accuracies = []
    p = []
    l = []
    
    for loader in [validloader,trainloader]:
        
        tally = 0

        for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            
            images,labels = data
                
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
                
                x = torch.reshape(x,(loader.batch_size,c.density_map_h ** 2)) # torch.Size([200, 12, 12])
                raw_preds = x  # torch.Size([200, 144])
                x = torch.mode(x,dim = 1).values.cpu().detach().numpy() #print(x.shape) (200,)
                
                #mean_preds = np.round(x.mean(dim = dims).cpu().detach().numpy()) # calculate using mean of preds, not mode
                mean_array.append(x) # mean_preds
            
            # https://stackoverflow.com/questions/48199077/elementwise-aggregation-average-of-values-in-a-list-of-numpy-arrays-with-same
            #mean_preds = np.mean(mean_array,axis=0)
                
            labels = labels.cpu().detach().numpy()
            
            tally += (labels == x).sum() # mean_preds
            #print(tally)
            
            if confusion:
                p.append(x)
                l.append(labels)
        
        accuracies.append(round(tally/(len(loader.dataset))*100,2))
  
    out = accuracies[0], accuracies[1]
        
    if confusion:
        l = torch.IntTensor([item for batch in l for item in batch]) # flatten lists
        p = torch.IntTensor([item for batch in p for item in batch]) 
        stacked = torch.stack((l,p),dim = 1)
        cmt = torch.zeros(10,10, dtype=torch.int64) 
        
        for p in stacked:
            tl, pl = p.tolist()
            cmt[tl, pl] = cmt[tl, pl] + 1
        
        out = out + (cmt,) 
    
    if preds:
        out = out + (raw_preds,)

    return  out # train, valid

def eval_model(model, validloader, trainloader, plot = True, save=True,hist=True):
    
    return 0,0
