import os.path
import tensorflow_addons as tfa
import functools

import consts


aspect_ratio = 1.0
test_mode = 'Full'                      # Scale|Full|Crop
label_nc = 0                            # Num of input label channels
n_downsample_global = 3                 # Number of downsampling layers in netG
mc = 64                                 # Num max channel
k_size = 4                              # Num kernel size conv layer
start_r = 1                             # Start layer to use resblock
mapping_n_block = 6                     # Number of resblock in mapping
map_mc = 512                            # Max channel of mapping

input_nc = consts.NUM_RGB_CHANNELS      # Num of input image channels
output_nc = input_nc                    # Num of output image channels
ngf = 64                                # Num of gen filters in first conv layer
norm = functools.partial(tfa.layers.InstanceNormalization, epsilon=1e-05)
spatio_size = 64
feat_dim = -1
use_segmentation_model = False
mapping_net_dilation = 1                # This parameter is the dilation size of the translation net
no_parsing_map = True
semantic_nc = 18
num_upsampling_layers = 'normal'
crop_size = 256
injection_layer = 'all'
preprocess_mode = 'resize'
softmax_temperature = 1.0
use_self = False
cosin_similarity = False
correlation_renormalize = True