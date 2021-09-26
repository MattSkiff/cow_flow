from torch import randn
from tqdm import tqdm
import numpy as np
import config as c

def eval_mnist(model, validloader, trainloader, plot = True, save=True,hist=True):

    k = 0
    
    for loader in [validloader,trainloader]:
        
        n_correct = 0
        n_sz = len(loader)*c.batch_size
        n = 0
    
        for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):

            if c.mnist:
                images,labels = data
            else:
                images,dmaps,labels = data
            
        
            if labels.size: # triggers only if there is at least one annotation
                
                # Z shape: torch.Size([2, 4, 300, 400]) (batch size = 2)
                dummy_z = (randn(c.batch_size, 4 , c.density_map_h // 2,c.density_map_w  // 2, requires_grad=True)).to(c.device)
                
                images = images.float().to(c.device)
                dummy_z = dummy_z.float().to(c.device)
                
                model = model.to(c.device)
                
                x, log_det_jac = model(images,dummy_z,rev=True)
                mean_preds = np.round(x.mean(dim = (1,2,3)).cpu().detach().numpy())
                labels = labels.cpu().detach().numpy()
                
                n_correct += (labels == mean_preds).sum()
                n += len(labels)
                
                if plot:
                    pass
                
                if n >= n_sz:
                    if k == 0:
                        k += 1
                        validation_accuracy = round((n_correct/1e4)*100,2)
                    elif k == 1:
                        train_accuracy = round((n_correct/6e4)*100,2)
                    break

    return  validation_accuracy, train_accuracy
