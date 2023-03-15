# external
# from functools import partial
from ray import tune,air,init
from ray.tune import CLIReporter, ExperimentAnalysis
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import ScalingConfig #, RunConfig

import torch
from torch.cuda import empty_cache      
from torch.utils.data.sampler import SubsetRandomSampler # RandomSampling
from torch.utils.data import DataLoader
import numpy as np
                                                                                                                                                          
import sys

# internal
import model
import config as c
import arguments as a
import gvars as g
from utils import save_model, save_weights
from data_loader import prep_transformed_dataset, train_val_split
from train import train, train_baselines
from eval import eval_baselines,dmap_metrics

if a.args.n_pyramid_blocks > 32:
    sys.setrecursionlimit(10000) # avoids error when saving model
    # see here: https://discuss.pytorch.org/t/using-dataloader-recursionerror-maximum-recursion-depth-exceeded-while-calling-a-python-object/36947/5

if c.gpu:
    empty_cache() # free up memory for cuda

def main(num_samples, max_num_epochs):
    
    # args = a.create_args()
    # a.arguments_check(args)
    
    #runtime_env = {"working_dir": g.ABSDIR, "excludes": ["/data/","/models/","/ray/","/weights/","/runs/","/.git/"]}

    init(local_mode=True) # runtime_env=runtime_env,
    
    #init(local_mode=False) # needed to prevent conflict with worker.py and args parsing occuring in raytune 
    # https://github.com/ray-project/ray/issues/4786
    # https://github.com/ray-project/ray/issues/21037
    
    config = {
        # shared hyper parameters between flow and baselines
        "lr": tune.loguniform(g.MAX_LR, g.MIN_LR),
        "batch_size": tune.choice([8,16,32,64]), #[8,16,32,64]),
        #'scheduler':tune.choice(['cyclic']),
        'scheduler':tune.choice(['exponential','step','none']),
        'optimiser':tune.choice(['sgd','adam','adamw']),
        'weight_decay':tune.uniform(1e-5, 1e-2),
        'noise':tune.choice([0]),
        'model_name':a.args.model_name,
        'scaling_config': ScalingConfig(num_workers=4,use_gpu=True) # worker = ray actor = single exp.
    }
    
    if a.args.model_name == 'NF':
        
        if a.args.resize:
          config['batch_size'] = tune.choice([16,32,64,128,256])
        else:
          config['batch_size'] = tune.choice([4,8,16,32]) #([8,16,32])
          
        config['n_pyramid_blocks'] = tune.sample_from(lambda _: np.random.randint(1, 5))
        config['joint_optim'] = tune.choice([1]) # enabled JO always on HP search for now
        config['fixed1x1conv'] = tune.choice([0])  # diabled 1x1 always on HP search for now
        config['scheduler'] = tune.choice(['exponential','step','none'])
        config['noise'] = tune.grid_search([1e-5,1e-4,1e-3,1e-2])
        config['freeze_bn'] = tune.grid_search([1,0])
        config['subnet_bn'] = tune.grid_search([1,0])
        config['filters'] = tune.grid_search([8,16,32,64,128,256])
        config['clamp'] = tune.grid_search([1,1.2,1.9,4,30])
        config['feat_extractor'] = tune.grid_search(['resnet9','resnet18','vgg16_bn','resnet50']) #  
        
    if a.args.small_batches:
        if a.args.model_name == 'NF': 
            # 1,036,800 combinations
            config['batch_size'] = tune.grid_search([2,4,8]) #,4,8])
        else:
            config['batch_size'] = tune.grid_search([4,8,16,32]) #,4,8])
        
    if a.args.small_batches:
        config['batch_size'] = tune.choice([2,4,8]) #,4,8])
        config['n_pyramid_blocks'] = tune.choice([1]) #,4,8])
        
    if a.args.model_name == 'LCFCN':
        config['batch_size'] = tune.choice([1])  
    
    logdir = g.ABSDIR+'ray/'
    
    def train_search(config=config, checkpoint_dir='./checkpoints/'):
        
        # updating global vars with config vars, as these are used when instantiating subnets when passing subnet constructor func to flow layers in FrEIA
        if a.args.model_name == 'NF':
            g.FILTERS = config['filters']
            g.SUBNET_BN = config['subnet_bn']
        
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
            train(train_loader,val_loader,config=config)
        else:
            train_baselines(config['model_name'],train_loader,val_loader,config=config)
            
        return
    
    def load_best_model_from_result(result,best_config):    
        
        best_trial = result.get_best_result(metric="loss",mode="min",scope='all') # ,"last"
        
        if a.args.model_name == 'NF':
            fe = model.select_feat_extractor(a.args.feat_extractor,config=best_config)
        else:
            fe = None
    
        best_trained_model = model.init_model(feat_extractor=fe,config=best_trial.config)
        
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            # if gpus_per_trial > 1:
            #     best_trained_model = nn.DataParallel(best_trained_model)
                            
        # empty model initialised
        best_trained_model.to(device)
        
        # https://docs.ray.io/en/latest/tune/getting-started.html#tune-tutorial
        # https://docs.ray.io/en/latest/train/dl_guide.html 
        
        best_result = result.get_best_result("loss", mode="min",scope='all') # doesn't include checkpoint name
        best_checkpoint = best_result.best_checkpoints[0][0]
        best_checkpoint_dict = best_checkpoint.to_dict()
        best_trained_model.load_state_dict(best_checkpoint_dict.get("model_weights"))
        
        return best_trained_model

    if a.args.exp_dir != '' and not a.args.resume:
        
        print(f"Loading results from {a.args.results_dir}...")

        # ray 2.3
        # restored_tuner = tune.Tuner.restore(a.args.results_dir)
        # result_grid = restored_tuner.get_results()
        result = ExperimentAnalysis(a.args.results_dir)

    else:
        assert a.args.num_samples
        assert a.args.max_num_epochs
        assert a.args.max_num_epochs and a.args.num_samples > 0
        
        scheduler = ASHAScheduler(
           metric="loss",
           mode="min",
           max_t=max_num_epochs,
           grace_period=1,
           reduction_factor=2)
         
        reporter = CLIReporter(
            # parameter_columns=["l1", "l2", "lr", "batch_size"],
            metric_columns=["loss", "accuracy", "training_iteration"])
        
        if a.args.resume:
            tuner = tune.Tuner.restore(a.args.exp_dir)
        else:
            tuner = tune.Tuner( 
                  train_search,
                  param_space=config,
                  tune_config=tune.TuneConfig(
                      max_concurrent_trials=10,
                      #metric='loss', # tune complains ASHA already has metric set  
                      reuse_actors=True,
                      num_samples=num_samples,
                      scheduler=scheduler),
                  run_config=air.RunConfig(local_dir=logdir,name=a.args.schema,
                                           progress_reporter=reporter,
                                           checkpoint_config=air.CheckpointConfig(checkpoint_at_end=False,
                                                                                  num_to_keep=1,
                                                                                  checkpoint_score_order='min',
                                                                                  checkpoint_score_attribute='loss'#,
                                                                                  #checkpoint_frequency=1 # only for class API
                                                                                  ))
                  )
        
        result = tuner.fit()
    
    best_trial = result.get_best_result(metric="loss",mode="min",scope='all') 
    best_config = best_trial.config
    best_model = load_best_model_from_result(result,best_config)

    transformed_dataset = prep_transformed_dataset(is_eval=a.args.mode=='eval',config=best_config)
    
    OOD_dataset = prep_transformed_dataset(is_eval=True,holdout=True,config=best_config)
    OOD_loader = DataLoader(OOD_dataset, batch_size=1,shuffle=False, 
                        num_workers=4,collate_fn=transformed_dataset.custom_collate_aerial,
                        pin_memory=False)
    
    if a.args.model_name == 'NF':
        holdout_metric_dict = dmap_metrics(best_model, OOD_loader,n=c.eval_n,mode='val')
    else:
        holdout_metric_dict = eval_baselines(best_model,OOD_loader,mode='val')
    
    if a.args.exp_dir != '' and not a.args.resume:
        search_name = a.args.model_name+'_hp_search_'+str(num_samples)+"_"+str(max_num_epochs)
    else:
        search_name = a.args.model_name+'_hp_search_'+a.args.exp_dir
    
    print("Best trial OOD test set metrics")
    print(holdout_metric_dict)

    print("Best trial config")
    print(best_trial.config)
    
    with open("/home/mks29/clones/cow_flow/models/"+search_name+".txt", 'w') as f:
        print("Best trial OOD test set metrics",file=f)
        print(holdout_metric_dict, file=f)
        print("Best trial config",file=f)
        print(best_trial.config, file=f)
    
    if a.args.exp_dir != '' and not a.args.resume:
        save_model(best_model,search_name)
        save_weights(best_model,search_name)

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=a.args.num_samples, max_num_epochs=a.args.max_num_epochs) #, gpus_per_trial=a.args.gpus_per_trial)
