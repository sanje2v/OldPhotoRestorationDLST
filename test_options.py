import os.path
import tensorflow_addons as tfa

import consts


ntest = float('inf')
results_dir = './results'
aspect_ratio = 1.0
phase = 'test'
which_epoch = 'latest'
cluster_path = 'features_clustered_010.npy'
use_encoded_image = False               # If specified, encode the real image to get the feature map
start_epoch = -1                        # Write the start_epoch of iter.txt into this parameter
no_degradation = False                  # When train the mapping, enable this parameter --> no degradation will be added into clean image
no_load_VAE = False                     # When train the mapping, enable this parameter --> random initialize the encoder an decoder
use_v2_degradation = False              # Enable this parameter --> 4 kinds of degradations will be used to synthesize corruption
use_vae_which_epoch = 'latest'
multi_scale_test = 0.5
multi_scale_threshold = 0.5
mask_need_scale = False                 # Enable this param meas that the pixel range of mask is 0-255
scale_num = 1
save_feature_url = ''                   # While extracting the features, where to put
test_input = ''                         # A directory or a root of bigfile
test_mask = ''                          # A directory or a root of bigfile
test_gt = ''                            # A directory or a root of bigfile
scale_input = False                     # While testing, choose to scale the input firstly
save_feature_name = 'features.json'     # The name of saved features
test_rgb_old_two_wo_scratch = False
test_mode = 'Full'                      # Scale|Full|Crop
Quality_restore = True                  # For RGB images
Scratch_and_Quality_restore = False     # For scratched images
serial_batches = True                   # no shuffle
no_flip = True                          # no flip
label_nc = 0                            # Num of input label channels
n_downsample_global = 3                 # Number of downsampling layers in netG
mc = 64                                 # Num max channel
k_size = 4                              # Num kernel size conv layer
start_r = 1                             # Start layer to use resblock
mapping_n_block = 6                     # Number of resblock in mapping
map_mc = 512                            # Max channel of mapping
no_instance = True                      # If specified, do *not* add instance map as input
checkpoints_dir = 'checkpoints/restoration'

input_nc = consts.NUM_RGB_CHANNELS      # Num of input image channels
output_nc = input_nc                    # Num of output image channels
ngf = 64                                # Num of gen filters in first conv layer
norm = tfa.layers.InstanceNormalization
spatio_size = 64
feat_dim = -1
use_segmentation_model = False
mapping_net_dilation = 1                # This parameter is the dilation size of the translation net

if Quality_restore:
    name = 'mapping_quality'
    NL_res = False
    use_SN = False
    correlation_renormalize = False
    NL_use_mask = False
    NL_fusion_method = 'add'
    non_local = ''                      # Which non_local setting
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