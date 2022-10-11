# plot predicted versus actual for cows dataset for models from both val/train splits of holdout dataset

import sys
import os

sys.path.append("/home/mks29/clones/cow_flow/")

# external
from matplotlib import pyplot as plt  
from pathlib import Path
import numpy as np  
import pandas as pd                                                                                                                                                               

# internal
import gvars as g

# want to plot width of prediction interval from NF (as measure of uncertainty)
# against actual error from baseline model
# high correspondence shows the NF is correctly assigning uncertainty to predictions

# want to combine errors across all predictions from all models

# baseline models
error_file_list = ["4H_CSRNet_hydra_BS64_LR_I0.0001_E1000_DIM256_OPTIMsgd_weight_none_04_05_2022_16_48_45.errors", # CSRNet
                   "4G2_Res50_hydra_BS64_LR_I0.0001_E1000_DIM256_OPTIMsgd_weight_none_04_05_2022_17_21_42.errors", # Res50
                   "86_LCFCN_A_LCFCN_quartet_BS1_LR_I1e-05_E1000_DIM608_OPTIMadam_weighted_MX_SZ_4_none_WD_1e-05_15_06_2022_14_17_53.errors", # LCFCN
                   "5DD_UNet_hydra_BS64_LR_I0.002_E5000_DIM256_OPTIMadam_weight_none_09_05_2022_20_05_07.errors", # UNet
                   "9S1_UNet_seg_ml-17_BS16_LR_I0.0002_E500_DIM256_OPTIMsgd_weight_MX_SZ_4_none_11_05_2022_13_13_32.errors", # UNet seg
                   "vgg_real1_VGG_hydra_BS16_LR_I0.001_E1000_DIM608_OPTIMsgd_SC1000_weighted_none_WD_0.01_29_09_2022_19_06_45.errors", # VGG
                   "relu3X_MCNN_hydra_BS64_LR_I0.001_E1000_DIM256_OPTIMsgd_SC1000_weighted_none_WD_1e-05_12_08_2022_12_01_49.errors"] # MCNN


# from NF
prediction_interval_file_name = "86_widereal_w_vgg1NF_C5_NF_hydra_BS8_LR_I0.0001_E250_DIM608_OPTIMadam_FE_vgg16_bn_NC5_conv_JC_weighted_none_JO_PY_1_WD_1e-05_15_07_2022_17_52_19.pred_int"

error_file_path = os.path.join(g.VIZ_DATA_DIR,error_file_name)
prediction_interval_file_path = os.path.join(g.VIZ_DATA_DIR,prediction_interval_file_name)

errors = []; prediction_intervals = []

for error_file_path in error_file_list:
    
    errors_input = []
    with open(error_file_path) as f:
        errors_input = f.readlines()
    
        for line in errors_input:
            errors.append(line)    

with open(prediction_interval_file_path) as f:
    prediction_interval_input = f.readlines()

    for line in prediction_interval_input:
        line = np.array(line[1:-2].split(),dtype=np.float32)
        prediction_intervals.append(line)  
        
intervals_df = pd.DataFrame(np.array(prediction_intervals), columns=['lb', 'ub'])
pred_width = np.array(np.abs(intervals_df['lb']-intervals_df['ub']))
errors = np.array(errors,dtype=np.float32)

fig, ax = plt.subplots(1)   
#ax.vlines(errors,intervals_df['lb'],intervals_df['ub'])
ax.scatter(errors,pred_width)
ax.set_xlabel('Baseline LCFCN error size') # .format(Path(error_file_path).stem.split('.')[0])
ax.set_ylabel('NF prediction interval width')

plt.show()


 fig2, axs2 = plt.subplots(4, 2)
 fig2.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,hspace=0.15,wspace=0.05)
 fig2.set_size_inches(10*1,10*1)
 fig2.set_dpi(100)