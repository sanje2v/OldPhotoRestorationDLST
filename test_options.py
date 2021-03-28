import os.path
import tensorflow_addons as tfa

import consts

ntest = float('inf')
results_dir = './results'
aspect_ratio = 1.0
phase = 'test'
which_epoch = 'latest'
cluster_path = 'features_clustered_010.npy'
use_encoded_image = False
start_epoch = -1
no_degradation = False
no_load_VAE = False
use_v2_degradation = False
use_vae_which_epoch = 'latest'
multi_scale_test = 0.5
multi_scale_threshold = 0.5
mask_need_scale = False
scale_num = 1
save_feature_url = ''
test_input = ''
test_mask = ''
test_gt = ''
scale_input = False
save_feature_name = 'features.json'
test_rgb_old_two_wo_scratch = False
test_mode = 'Crop'
Quality_restore = True
Scratch_and_Quality_restore = False
serial_batches = True  # no shuffle
no_flip = True  # no flip
label_nc = 0
n_downsample_global = 3
mc = 64
k_size = 4
start_r = 1
mapping_n_block = 6
map_mc = 512
no_instance = True
checkpoints_dir = 'checkpoints/restoration'

input_nc = consts.NUM_RGB_CHANNELS
output_nc = input_nc
ngf = 64
norm = tfa.layers.InstanceNormalization
non_local = ''
spatio_size = 64
feat_dim = -1
use_segmentation_model = False

if Quality_restore:
    name = 'mapping_quality'
    NL_use_mask = False
    load_pretrainA = os.path.join(checkpoints_dir, "VAE_A_quality")
    load_pretrainB = os.path.join(checkpoints_dir, "VAE_B_quality")

if Scratch_and_Quality_restore:
    name = 'mappint-scratch'
    NL_res = True
    use_SN = True
    correlation_renormalize = True
    NL_use_mask = True
    NL_fusion_method = 'combine'
    non_local = 'Setting_42'
    load_pretrainA = os.path.join(checkpoints_dir, "VAE_A_quality")
    load_pretrainB = os.path.join(checkpoints_dir, "VAE_B_scratch")