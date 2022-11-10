# external
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import os
import torch
import torch.nn as nn
from torch.cuda import empty_cache      
from torch.utils.data.sampler import SubsetRandomSampler # RandomSampling
from torch.utils.data import DataLoader, Dataset
import numpy as np
                                                                                                                                                          
import sys

# internal
import model
import config as c
import arguments as a
from data_loader import prep_transformed_dataset, make_loaders, train_val_split
from train import train, train_baselines
from eval import eval_baselines,dmap_metrics

if a.args.n_pyramid_blocks > 32:
    sys.setrecursionlimit(10000) # avoids error when saving model
    # see here: https://discuss.pytorch.org/t/using-dataloader-recursionerror-maximum-recursion-depth-exceeded-while-calling-a-python-object/36947/5

if c.gpu:
    empty_cache() # free up memory for cuda

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):

    config = {
        # shared hyper parameters between flow and baselines
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([8,16,32,64]),
        'scheduler':tune.choice(['exponential','step','none']),
        'optim':tune.choice(['sgd','adam','adamw']),
        'weight_decay':tune.uniform(1e-5, 1e-2),
        'noise':tune.choice([0])
    }
        
    if a.args.model_name == 'NF':
        config['batch_size'] = tune.choice([8,16,32])
        config['n_coupling_blocks'] = tune.sample_from(lambda _: np.random.randint(1, 6))
        config['joint_optim'] = tune.choice([True,False]),
        config['fixed1x1conv'] = tune.choice([True,False])
        config['noise'] = tune.uniform(1e-4, 1e-2)
        config['freeze_bn'] = tune.choice([True,False])
        config['subnet_bn'] = tune.choice([True,False])
        config['filters'] = tune.choice([16,32,64,128,256])
        config['clamp'] = tune.choice([1,1.2,1.9,4,30])
        config['feat_extractor'] = tune.choice(['resnet18','vgg16_bn','resnet50','resnet9'])
    
    if a.args.model_name == 'LCFCN':
        config['batch_size'] = tune.choice([8,16,32])
        config['batch_size'] = tune.choice([1])  

    gpus_per_trial = 2

    scheduler = ASHAScheduler(
       metric="loss",
       mode="min",
       max_t=max_num_epochs,
       grace_period=1,
       reduction_factor=2)
     
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    def train_search(config, checkpoint_dir='./checkpoints/'):
        
        dataset = prep_transformed_dataset(is_eval=a.args.mode=='eval',config=config)
        
        t_indices, t_weights, v_indices, v_weights  = train_val_split(dataset = dataset,
                                                          train_percent = c.test_train_split,
                                                          annotations_only = (a.args.sampler == 'anno'),
                                                          seed = c.seed,
                                                          oversample= (a.args.sampler == 'weighted'))
        
        f_t_indices, f_t_weights, f_v_indices, f_v_weights  = train_val_split(dataset = dataset,
                                                          train_percent = c.test_train_split,
                                                          annotations_only = False,
                                                          seed = c.seed,
                                                          oversample=False)
        
        train_sampler = SubsetRandomSampler(t_indices,generator=torch.Generator().manual_seed(c.seed))
        val_sampler = SubsetRandomSampler(v_indices,generator=torch.Generator().manual_seed(c.seed))
              
        # leave shuffle off for use of any samplers
        train_loader = DataLoader(dataset, batch_size=int(config['batch_size']),shuffle=False, 
                            num_workers=a.args.workers,collate_fn=dataset.custom_collate_aerial,
                            pin_memory=True,sampler=train_sampler,persistent_workers=False,prefetch_factor=2)
        
        val_loader = DataLoader(dataset, batch_size=int(config['batch_size']),shuffle=False, 
                            num_workers=a.args.workers,collate_fn=dataset.custom_collate_aerial,
                            pin_memory=True,sampler=val_sampler,persistent_workers=False,prefetch_factor=2) 
        
        if a.args.model_name == 'NF':
            train(train_loader,val_loader)
        else:
            train_baselines(a.args.model_name,train_loader,val_loader)

    result = tune.run(
        train_search,
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=False)
        
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = model.CowFlow(modelname='best_mdl',feat_extractor = model.select_feat_extractor(a.args.feat_extractor),config=best_trial.config)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
    
    transformed_dataset = prep_transformed_dataset(is_eval=a.args.mode=='eval')
    
    OOD_dataset = prep_transformed_dataset(is_eval=False,holdout=True)
    OOD_loader = DataLoader(OOD_dataset, batch_size=a.args.batch_size,shuffle=False, 
                        num_workers=4,collate_fn=transformed_dataset.custom_collate_aerial,
                        pin_memory=False,holdout=True)
    
    if a.args.model_name == 'NF':
        holdout_metric_dict = dmap_metrics(best_trained_model, OOD_loader,n=c.eval_n,mode='val')
    else:
        holdout_metric_dict = eval_baselines(best_trained_model,OOD_loader,mode='val')

    print("Best trial OOD test set metrics")
    print(holdout_metric_dict)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
