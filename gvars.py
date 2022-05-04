import arguments as a

FEAT_MOD_DIR = './models/feat_extractors/'
VIZ_DIR = './viz'
WEIGHT_DIR = './weights'
MODEL_DIR = './models'
LOG_DIR = './logs'
C_DIR = './cstates'
DMAP_DIR = './data/precompute/size_{}_sigma_{}/'.format(15,a.args.sigma)

BASELINE_MODEL_NAMES = ['UNet','CSRNet','FCRN','LCFCN','UNet_seg','MCNN','Res50']

if a.args.model_name in BASELINE_MODEL_NAMES:
    assert a.args.noise == 0
    
assert a.args.model_name in BASELINE_MODEL_NAMES + ['NF']