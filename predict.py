#!/usr/bin/env python3

import utils

#utils.predict_image(mdl_path,mdl_type='UNet_seg')

path = '/home/mks29/clones/cow_flow/models/'

# Done - good
#mdl_path = path + 'UNet_seg/best_9S1_UNet_seg_ml-17_BS16_LR_I0.0002_E500_DIM256_OPTIMsgd_weight_MX_SZ_4_none_11_05_2022_13_13_32'
#utils.predict_image(mdl_path,mdl_type='UNet_seg',geo=False,image_path='/home/mks29/Desktop/BB34_5000_1006.jpg')

# Done - good
# mdl_path = path + 'UNet/best_16_05B_UNet_hydra_BS64_LR_I0.002_E5000_DIM256_OPTIMsgd_weighted_none_16_05_2022_13_52_17'
# utils.predict_image(mdl_path,mdl_type='UNet',geo=False,image_path='/home/mks29/Desktop/BB34_5000_1006.jpg')

# Done - good
# mdl_path = path + 'Res50/good/best_4G2_Res50_hydra_BS64_LR_I0.0001_E1000_DIM256_OPTIMsgd_weight_none_04_05_2022_17_21_42'
# utils.predict_image(mdl_path,mdl_type='Res50',geo=False,image_path='/home/mks29/Desktop/BB34_5000_1006.jpg')

# Done - good
# mdl_path = path + 'Res50/good/best_4G2_Res50_hydra_BS64_LR_I0.0001_E1000_DIM256_OPTIMsgd_weight_none_04_05_2022_17_21_42'
# utils.predict_image(mdl_path,mdl_type='Res50',geo=False,image_path='/home/mks29/Desktop/BB34_5000_1006.jpg')

# Done - good
# mdl_path = path + 'FCRN/best_9CC_FCRN_hydra_BS64_LR_I0.1_E900_DIM256_OPTIMsgd_weight_none_WD_None_09_05_2022_20_05_07'
# utils.predict_image(mdl_path,mdl_type='FCRN',geo=False,image_path='/home/mks29/Desktop/BB34_5000_1006.jpg')

# Done - good
# mdl_path = path + 'MCNN/best/best_3X_MCNN_hydra_BS64_LR_I0.001_E1000_DIM256_OPTIMsgd_weight_none_03_05_2022_18_04_04'
# utils.predict_image(mdl_path,mdl_type='MCNN',geo=False,image_path='/home/mks29/Desktop/BB34_5000_1006.jpg')

# Done - good
# mdl_path = path + 'CSRNet/post_scaling_viable/best_4H_CSRNet_hydra_BS64_LR_I0.0001_E1000_DIM256_OPTIMsgd_weight_none_04_05_2022_16_48_45'
# utils.predict_image(mdl_path,mdl_type='CSRNet',geo=False,image_path='/home/mks29/Desktop/BB34_5000_1006.jpg')

# done - LCFCN
# mdl_path = path + 'LCFCN/viable/best_4QQ_LCFCN_ml-21_BS1_LR_I0.001_E1000_DIM256_OPTIMadam_anno_MX_SZ_3_none_WD_0.001_05_05_2022_12_33_33'
# utils.predict_image(mdl_path,mdl_type='LCFCN',geo=False,image_path='/home/mks29/Desktop/BB34_5000_1006.jpg')

# done - NF
mdl_path = path + '/COWS/viable/final_9ZZ2_NF_quatern_BS64_LR_I0.0002_E10000_DIM256_OPTIMadam_FE_resnet18_NC5_weight_step_JO_PY_1_1x1_10_05_2022_17_31_16'
# mdl_path = path + 'best_res50_w_C5_NF_hydra_BS8_LR_I0.0001_E1000_DIM608_OPTIMadam_FE_resnet50_NC5_conv_weighted_none_JO_PY_1_WD_1e-05_13_07_2022_18_18_36'
utils.predict_image(mdl_path,mdl_type='NF',nf=True,geo=False,nf_n=1,image_path='/home/mks29/Desktop/BB34_5000_1006.jpg')




