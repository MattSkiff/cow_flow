import arguments as a
import numpy as np

FEAT_MOD_DIR = './models/feat_extractors/'
VIZ_DIR = './viz'
VIZ_DATA_DIR = './viz/data/'
WEIGHT_DIR = './weights'
MODEL_DIR = './models'
LOG_DIR = './logs'
C_DIR = './cstates'
DMAP_DIR = './data/precompute/size_{}_sigma_{}/'.format(15,a.args.sigma)

bm_dir = 'final_models_paper/256/'
bm_dir_86 = 'final_models_paper/800x600/'

BEST_MODEL_PATH_DICT = {
    'UNet':bm_dir+'best_5DD_UNet_hydra_BS64_LR_I0.002_E5000_DIM256_OPTIMadam_weight_none_09_05_2022_20_05_07',
    'UNet_seg':bm_dir+'best_9S1_UNet_seg_ml-17_BS16_LR_I0.0002_E500_DIM256_OPTIMsgd_weight_MX_SZ_4_none_11_05_2022_13_13_32',
    'CSRNet':bm_dir+'best_4H_CSRNet_hydra_BS64_LR_I0.0001_E1000_DIM256_OPTIMsgd_weight_none_04_05_2022_16_48_45',
    'FCRN':bm_dir+'best_9CC_FCRN_hydra_BS64_LR_I0.1_E900_DIM256_OPTIMsgd_weight_none_WD_None_09_05_2022_20_05_07',
    'LCFCN':bm_dir_86+'best_86_LCFCN_A_LCFCN_quartet_BS1_LR_I1e-05_E1000_DIM608_OPTIMadam_weighted_MX_SZ_4_none_WD_1e-05_15_06_2022_14_17_53',
    #'MCNN':bm_dir+'best_86_MCNN_B_MCNN_quatern_BS32_LR_I0.001_E1000_DIM608_OPTIMadam_weighted_none_WD_1e-05_FT_15_06_2022_17_21_20',
    'MCNN':bm_dir+'best_relu3X_MCNN_hydra_BS64_LR_I0.001_E1000_DIM256_OPTIMsgd_SC1000_weighted_none_WD_1e-05_12_08_2022_12_01_49',
    'Res50':bm_dir+'best_4G2_Res50_hydra_BS64_LR_I0.0001_E1000_DIM256_OPTIMsgd_weight_none_04_05_2022_17_21_42',
    'NF':bm_dir_86+'best_86_widereal_w_vgg1NF_C5_NF_hydra_BS8_LR_I0.0001_E250_DIM608_OPTIMadam_FE_vgg16_bn_NC5_conv_JC_weighted_none_JO_PY_1_WD_1e-05_15_07_2022_17_52_19',
    'NF_FRZ':bm_dir_86+'best_2_FRZ86_widereal_w_vgg1NF_C5_NF_hydra_BS12_LR_I0.0001_E250_DIM608_OPTIMadam_FE_vgg16_bn_NC5_conv_JC_weighted_none_PT_PY_1_WD_1e-05_29_08_2022_13_28_01',
    'VGG':bm_dir_86+'best_vgg_real1_VGG_hydra_BS16_LR_I0.001_E1000_DIM608_OPTIMsgd_SC1000_weighted_none_WD_0.01_29_09_2022_19_06_45'
    }

BASELINE_MODEL_NAMES = ['UNet','CSRNet','FCRN','LCFCN','UNet_seg','MCNN','Res50','VGG']
SUBNETS = ['conv','conv_shallow','fc','MCNN','UNet','conv_deep']
THRES_SEQ = np.arange(0, 20, 0.5, dtype=float)

global FILTERS
FILTERS = 0

global SUBNET_BN
SUBNET_BN = False

if a.args.model_name in BASELINE_MODEL_NAMES:
    assert a.args.noise == 0

if not (a.args.mode == 'plot' and a.args.plot_errors):
    assert a.args.model_name in BASELINE_MODEL_NAMES + ['NF','ALL']

if a.args.model_name == 'NF' and a.args.mode == 'train':
    assert a.args.subnet_type in SUBNETS