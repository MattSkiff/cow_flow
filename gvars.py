import arguments as a
import numpy as np

FEAT_MOD_DIR = './models/feat_extractors/'
VIZ_DIR = './viz'
WEIGHT_DIR = './weights'
MODEL_DIR = './models'
LOG_DIR = './logs'
C_DIR = './cstates'
DMAP_DIR = './data/precompute/size_{}_sigma_{}/'.format(15,a.args.sigma)

bm_dir = 'final_models_paper/256/'

BEST_MODEL_PATH_DICT = {
    'UNet':bm_dir+'best_16_05B_UNet_hydra_BS64_LR_I0.002_E5000_DIM256_OPTIMsgd_weighted_none_16_05_2022_13_52_17',
    'UNet_seg':bm_dir+'best_9S1_UNet_seg_ml-17_BS16_LR_I0.0002_E500_DIM256_OPTIMsgd_weight_MX_SZ_4_none_11_05_2022_13_13_32',
    'CSRNet':bm_dir+'best_4H_CSRNet_hydra_BS64_LR_I0.0001_E1000_DIM256_OPTIMsgd_weight_none_04_05_2022_16_48_45',
    'FCRN':bm_dir+'best_9CC_FCRN_hydra_BS64_LR_I0.1_E900_DIM256_OPTIMsgd_weight_none_WD_None_09_05_2022_20_05_07',
    'LCFCN':bm_dir+'best_4QQ2_LCFCN_ml-16_BS1_LR_I0.001_E1000_DIM256_OPTIMadam_anno_MX_SZ_3_none_WD_0.001_05_05_2022_16_44_46',
    'MCNN':bm_dir+'best_3X_MCNN_hydra_BS64_LR_I0.001_E1000_DIM256_OPTIMsgd_weight_none_03_05_2022_18_04_04',
    'Res50':bm_dir+'best_4G2_Res50_hydra_BS64_LR_I0.0001_E1000_DIM256_OPTIMsgd_weight_none_04_05_2022_17_21_42',
    'NF':bm_dir+'best_9ZZ2_NF_quatern_BS64_LR_I0.0002_E10000_DIM256_OPTIMadam_FE_resnet18_NC5_weight_step_JO_PY_1_1x1_10_05_2022_17_31_16'
    }

BASELINE_MODEL_NAMES = ['UNet','CSRNet','FCRN','LCFCN','UNet_seg','MCNN','Res50']
SUBNETS = ['conv','conv_shallow','fc','MCNN','UNet','']
THRES_SEQ = np.arange(0, 20, 0.5, dtype=float)

if a.args.model_name in BASELINE_MODEL_NAMES:
    assert a.args.noise == 0
    
assert a.args.model_name in BASELINE_MODEL_NAMES + ['NF','ALL']

if a.args.model_name == 'NF':
    assert a.args.subnet_type in SUBNETS