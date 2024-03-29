import os
from models import FaceEnhancer
import test_options as opts
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, ReLU, InputLayer
from tensorflow_addons.layers import SpectralNormalization, InstanceNormalization
from pprint import pprint

import consts

from models import *
from models.modules import *


with tf.device("/CPU"):
    scratch_detector = ScratchDetector()
    a = scratch_detector(tf.zeros((256, 384, 1), dtype=tf.dtypes.float32), training=False)
    print(a.shape)
    pprint([x.name for x in scratch_detector.variables])


    #opts.with_scratch = False
    #model = ImageEnhancer(opts)
    #model([np.empty((1, 608, 928, consts.NUM_RGB_CHANNELS), dtype=np.float32),
    #       np.empty((1, 608, 928, 1), dtype=np.float32)])

    #def iterate_layer(prefix, submodules):
    #    i = 0
    #    for inner_layer in submodules:
    #        if isinstance(inner_layer, (Sequential, InputLayer)):
    #            continue
    #        elif isinstance(inner_layer, (Conv2D, Conv2DTranspose)):
    #            out = prefix + '.' + layer.inner_layers.name + "." + str(i) if inner_layer.name.startswith('conv2d') else inner_layer.name
    #            i = i + 1
    #            print(out)
    #        elif isinstance(inner_layer, (BatchNormalization, InstanceNormalization, ReLU, LeakyReLU)):
    #            i = i + 1
    #            continue
    #        else:
    #            i = i + 1

    #            if len(inner_layer.submodules) > 0:
    #                iterate_layer(prefix + '.' + inner_layer.name, inner_layer.submodules)

    #for layer in model.layers:
    #    print(layer.name)

    #    iterate_layer(layer.name, layer.submodules)

#    opts.with_scratch = True
#    model = ImageEnhancer(opts)
#    model([np.empty((1, 608, 928, consts.NUM_RGB_CHANNELS), dtype=np.float32),
#           np.empty((1, 608, 928, 1), dtype=np.float32)])

    

    #test = NonLocalBlock2D_with_mask_Res(512, 512, 'combine', True, 1.0, False, False)
    #y = test([tf.zeros((1, 152, 232, 512)), tf.ones((1, 608, 928, 1))], training=False)

    #opts.with_scratch = True
    #face_enhancer = FaceEnhancer(opts)
    #face_enhancer([np.empty((1, 604, 920, consts.NUM_RGB_CHANNELS), dtype=np.float32),
    #               np.empty((1, 604, 920, 1), dtype=np.float32)])

#    # Load model weights
#    face_enhancer([np.empty((1, 256, 256, consts.NUM_RGB_CHANNELS), dtype=np.float32),
#                   np.empty((1, 256, 256, consts.NUM_RGB_CHANNELS), dtype=np.float32)])
#    face_enhancer.load_weights(os.path.abspath('./weights/Face_Enhancement/tf_keras/out.weights')).assert_consumed()

#    face_enhancer([np.zeros((1, 256, 256, 18), dtype=np.float32),
#                   np.ones((1, 256, 256, consts.NUM_RGB_CHANNELS), dtype=np.float32)])

# class A(tf.keras.layers.Layer):
    # def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        # self.k = [tf.keras.layers.Conv2D(2, 2, name='k'), tf.keras.layers.Conv2D(3, 3, name='l')]
        
    # def call(self, inputs, training):
        # return self.k[1](self.k[0](inputs, training=training), training=training)
        
# a=A()
# o=a(tf.random.normal((1, 3, 3, 3)), training=False)
# print(a.variables)
# print(a.submodules)

#a=['netG/fc/kernel:0',
# 'netG/fc/bias:0',
# 'netG/head_0/norm_0/mlp_shared/kernel:0',
# 'netG/head_0/norm_0/mlp_shared/bias:0',
# 'netG/head_0/norm_0/mlp_gamma/kernel:0',
# 'netG/head_0/norm_0/mlp_gamma/bias:0',
# 'netG/head_0/norm_0/mlp_beta/kernel:0',
# 'netG/head_0/norm_0/mlp_beta/bias:0',
# 'netG/head_0/spectral_normalization/kernel:0',
# 'netG/head_0/spectral_normalization/bias:0',
# 'netG/head_0/norm_1/mlp_shared/kernel:0',
# 'netG/head_0/norm_1/mlp_shared/bias:0',
# 'netG/head_0/norm_1/mlp_gamma/kernel:0',
# 'netG/head_0/norm_1/mlp_gamma/bias:0',
# 'netG/head_0/norm_1/mlp_beta/kernel:0',
# 'netG/head_0/norm_1/mlp_beta/bias:0',
# 'netG/head_0/spectral_normalization_1/kernel:0',
# 'netG/head_0/spectral_normalization_1/bias:0',
# 'netG/G_middle_0/norm_0/mlp_shared/kernel:0',
# 'netG/G_middle_0/norm_0/mlp_shared/bias:0',
# 'netG/G_middle_0/norm_0/mlp_gamma/kernel:0',
# 'netG/G_middle_0/norm_0/mlp_gamma/bias:0',
# 'netG/G_middle_0/norm_0/mlp_beta/kernel:0',
# 'netG/G_middle_0/norm_0/mlp_beta/bias:0',
# 'netG/G_middle_0/spectral_normalization_2/kernel:0',
# 'netG/G_middle_0/spectral_normalization_2/bias:0',
# 'netG/G_middle_0/norm_1/mlp_shared/kernel:0',
# 'netG/G_middle_0/norm_1/mlp_shared/bias:0',
# 'netG/G_middle_0/norm_1/mlp_gamma/kernel:0',
# 'netG/G_middle_0/norm_1/mlp_gamma/bias:0',
# 'netG/G_middle_0/norm_1/mlp_beta/kernel:0',
# 'netG/G_middle_0/norm_1/mlp_beta/bias:0',
# 'netG/G_middle_0/spectral_normalization_3/kernel:0',
# 'netG/G_middle_0/spectral_normalization_3/bias:0',
# 'netG/G_middle_1/norm_0/mlp_shared/kernel:0',
# 'netG/G_middle_1/norm_0/mlp_shared/bias:0',
# 'netG/G_middle_1/norm_0/mlp_gamma/kernel:0',
# 'netG/G_middle_1/norm_0/mlp_gamma/bias:0',
# 'netG/G_middle_1/norm_0/mlp_beta/kernel:0',
# 'netG/G_middle_1/norm_0/mlp_beta/bias:0',
# 'netG/G_middle_1/spectral_normalization_4/kernel:0',
# 'netG/G_middle_1/spectral_normalization_4/bias:0',
# 'netG/G_middle_1/norm_1/mlp_shared/kernel:0',
# 'netG/G_middle_1/norm_1/mlp_shared/bias:0',
# 'netG/G_middle_1/norm_1/mlp_gamma/kernel:0',
# 'netG/G_middle_1/norm_1/mlp_gamma/bias:0',
# 'netG/G_middle_1/norm_1/mlp_beta/kernel:0',
# 'netG/G_middle_1/norm_1/mlp_beta/bias:0',
# 'netG/G_middle_1/spectral_normalization_5/kernel:0',
# 'netG/G_middle_1/spectral_normalization_5/bias:0',
# 'netG/up_0/norm_s/mlp_shared/kernel:0',
# 'netG/up_0/norm_s/mlp_shared/bias:0',
# 'netG/up_0/norm_s/mlp_gamma/kernel:0',
# 'netG/up_0/norm_s/mlp_gamma/bias:0',
# 'netG/up_0/norm_s/mlp_beta/kernel:0',
# 'netG/up_0/norm_s/mlp_beta/bias:0',
# 'netG/up_0/spectral_normalization_8/kernel:0',
# 'netG/up_0/norm_0/mlp_shared/kernel:0',
# 'netG/up_0/norm_0/mlp_shared/bias:0',
# 'netG/up_0/norm_0/mlp_gamma/kernel:0',
# 'netG/up_0/norm_0/mlp_gamma/bias:0',
# 'netG/up_0/norm_0/mlp_beta/kernel:0',
# 'netG/up_0/norm_0/mlp_beta/bias:0',
# 'netG/up_0/spectral_normalization_6/kernel:0',
# 'netG/up_0/spectral_normalization_6/bias:0',
# 'netG/up_0/norm_1/mlp_shared/kernel:0',
# 'netG/up_0/norm_1/mlp_shared/bias:0',
# 'netG/up_0/norm_1/mlp_gamma/kernel:0',
# 'netG/up_0/norm_1/mlp_gamma/bias:0',
# 'netG/up_0/norm_1/mlp_beta/kernel:0',
# 'netG/up_0/norm_1/mlp_beta/bias:0',
# 'netG/up_0/spectral_normalization_7/kernel:0',
# 'netG/up_0/spectral_normalization_7/bias:0',
# 'netG/up_1/norm_s/mlp_shared/kernel:0',
# 'netG/up_1/norm_s/mlp_shared/bias:0',
# 'netG/up_1/norm_s/mlp_gamma/kernel:0',
# 'netG/up_1/norm_s/mlp_gamma/bias:0',
# 'netG/up_1/norm_s/mlp_beta/kernel:0',
# 'netG/up_1/norm_s/mlp_beta/bias:0',
# 'netG/up_1/spectral_normalization_11/kernel:0',
# 'netG/up_1/norm_0/mlp_shared/kernel:0',
# 'netG/up_1/norm_0/mlp_shared/bias:0',
# 'netG/up_1/norm_0/mlp_gamma/kernel:0',
# 'netG/up_1/norm_0/mlp_gamma/bias:0',
# 'netG/up_1/norm_0/mlp_beta/kernel:0',
# 'netG/up_1/norm_0/mlp_beta/bias:0',
# 'netG/up_1/spectral_normalization_9/kernel:0',
# 'netG/up_1/spectral_normalization_9/bias:0',
# 'netG/up_1/norm_1/mlp_shared/kernel:0',
# 'netG/up_1/norm_1/mlp_shared/bias:0',
# 'netG/up_1/norm_1/mlp_gamma/kernel:0',
# 'netG/up_1/norm_1/mlp_gamma/bias:0',
# 'netG/up_1/norm_1/mlp_beta/kernel:0',
# 'netG/up_1/norm_1/mlp_beta/bias:0',
# 'netG/up_1/spectral_normalization_10/kernel:0',
# 'netG/up_1/spectral_normalization_10/bias:0',
# 'netG/up_2/norm_s/mlp_shared/kernel:0',
# 'netG/up_2/norm_s/mlp_shared/bias:0',
# 'netG/up_2/norm_s/mlp_gamma/kernel:0',
# 'netG/up_2/norm_s/mlp_gamma/bias:0',
# 'netG/up_2/norm_s/mlp_beta/kernel:0',
# 'netG/up_2/norm_s/mlp_beta/bias:0',
# 'netG/up_2/spectral_normalization_14/kernel:0',
# 'netG/up_2/norm_0/mlp_shared/kernel:0',
# 'netG/up_2/norm_0/mlp_shared/bias:0',
# 'netG/up_2/norm_0/mlp_gamma/kernel:0',
# 'netG/up_2/norm_0/mlp_gamma/bias:0',
# 'netG/up_2/norm_0/mlp_beta/kernel:0',
# 'netG/up_2/norm_0/mlp_beta/bias:0',
# 'netG/up_2/spectral_normalization_12/kernel:0',
# 'netG/up_2/spectral_normalization_12/bias:0',
# 'netG/up_2/norm_1/mlp_shared/kernel:0',
# 'netG/up_2/norm_1/mlp_shared/bias:0',
# 'netG/up_2/norm_1/mlp_gamma/kernel:0',
# 'netG/up_2/norm_1/mlp_gamma/bias:0',
# 'netG/up_2/norm_1/mlp_beta/kernel:0',
# 'netG/up_2/norm_1/mlp_beta/bias:0',
# 'netG/up_2/spectral_normalization_13/kernel:0',
# 'netG/up_2/spectral_normalization_13/bias:0',
# 'netG/up_3/norm_s/mlp_shared/kernel:0',
# 'netG/up_3/norm_s/mlp_shared/bias:0',
# 'netG/up_3/norm_s/mlp_gamma/kernel:0',
# 'netG/up_3/norm_s/mlp_gamma/bias:0',
# 'netG/up_3/norm_s/mlp_beta/kernel:0',
# 'netG/up_3/norm_s/mlp_beta/bias:0',
# 'netG/up_3/spectral_normalization_17/kernel:0',
# 'netG/up_3/norm_0/mlp_shared/kernel:0',
# 'netG/up_3/norm_0/mlp_shared/bias:0',
# 'netG/up_3/norm_0/mlp_gamma/kernel:0',
# 'netG/up_3/norm_0/mlp_gamma/bias:0',
# 'netG/up_3/norm_0/mlp_beta/kernel:0',
# 'netG/up_3/norm_0/mlp_beta/bias:0',
# 'netG/up_3/spectral_normalization_15/kernel:0',
# 'netG/up_3/spectral_normalization_15/bias:0',
# 'netG/up_3/norm_1/mlp_shared/kernel:0',
# 'netG/up_3/norm_1/mlp_shared/bias:0',
# 'netG/up_3/norm_1/mlp_gamma/kernel:0',
# 'netG/up_3/norm_1/mlp_gamma/bias:0',
# 'netG/up_3/norm_1/mlp_beta/kernel:0',
# 'netG/up_3/norm_1/mlp_beta/bias:0',
# 'netG/up_3/spectral_normalization_16/kernel:0',
# 'netG/up_3/spectral_normalization_16/bias:0',
# 'netG/conv_img/kernel:0',
# 'netG/conv_img/bias:0',
# 'netG/head_0/norm_0/param_free_norm/moving_mean:0',
# 'netG/head_0/norm_0/param_free_norm/moving_variance:0',
# 'netG/head_0/spectral_normalization/sn_u:0',
# 'netG/head_0/norm_1/param_free_norm/moving_mean:0',
# 'netG/head_0/norm_1/param_free_norm/moving_variance:0',
# 'netG/head_0/spectral_normalization_1/sn_u:0',
# 'netG/G_middle_0/norm_0/param_free_norm/moving_mean:0',
# 'netG/G_middle_0/norm_0/param_free_norm/moving_variance:0',
# 'netG/G_middle_0/spectral_normalization_2/sn_u:0',
# 'netG/G_middle_0/norm_1/param_free_norm/moving_mean:0',
# 'netG/G_middle_0/norm_1/param_free_norm/moving_variance:0',
# 'netG/G_middle_0/spectral_normalization_3/sn_u:0',
# 'netG/G_middle_1/norm_0/param_free_norm/moving_mean:0',
# 'netG/G_middle_1/norm_0/param_free_norm/moving_variance:0',
# 'netG/G_middle_1/spectral_normalization_4/sn_u:0',
# 'netG/G_middle_1/norm_1/param_free_norm/moving_mean:0',
# 'netG/G_middle_1/norm_1/param_free_norm/moving_variance:0',
# 'netG/G_middle_1/spectral_normalization_5/sn_u:0',
# 'netG/up_0/norm_s/param_free_norm/moving_mean:0',
# 'netG/up_0/norm_s/param_free_norm/moving_variance:0',
# 'netG/up_0/spectral_normalization_8/sn_u:0',
# 'netG/up_0/norm_0/param_free_norm/moving_mean:0',
# 'netG/up_0/norm_0/param_free_norm/moving_variance:0',
# 'netG/up_0/spectral_normalization_6/sn_u:0',
# 'netG/up_0/norm_1/param_free_norm/moving_mean:0',
# 'netG/up_0/norm_1/param_free_norm/moving_variance:0',
# 'netG/up_0/spectral_normalization_7/sn_u:0',
# 'netG/up_1/norm_s/param_free_norm/moving_mean:0',
# 'netG/up_1/norm_s/param_free_norm/moving_variance:0',
# 'netG/up_1/spectral_normalization_11/sn_u:0',
# 'netG/up_1/norm_0/param_free_norm/moving_mean:0',
# 'netG/up_1/norm_0/param_free_norm/moving_variance:0',
# 'netG/up_1/spectral_normalization_9/sn_u:0',
# 'netG/up_1/norm_1/param_free_norm/moving_mean:0',
# 'netG/up_1/norm_1/param_free_norm/moving_variance:0',
# 'netG/up_1/spectral_normalization_10/sn_u:0',
# 'netG/up_2/norm_s/param_free_norm/moving_mean:0',
# 'netG/up_2/norm_s/param_free_norm/moving_variance:0',
# 'netG/up_2/spectral_normalization_14/sn_u:0',
# 'netG/up_2/norm_0/param_free_norm/moving_mean:0',
# 'netG/up_2/norm_0/param_free_norm/moving_variance:0',
# 'netG/up_2/spectral_normalization_12/sn_u:0',
# 'netG/up_2/norm_1/param_free_norm/moving_mean:0',
# 'netG/up_2/norm_1/param_free_norm/moving_variance:0',
# 'netG/up_2/spectral_normalization_13/sn_u:0',
# 'netG/up_3/norm_s/param_free_norm/moving_mean:0',
# 'netG/up_3/norm_s/param_free_norm/moving_variance:0',
# 'netG/up_3/spectral_normalization_17/sn_u:0',
# 'netG/up_3/norm_0/param_free_norm/moving_mean:0',
# 'netG/up_3/norm_0/param_free_norm/moving_variance:0',
# 'netG/up_3/spectral_normalization_15/sn_u:0',
# 'netG/up_3/norm_1/param_free_norm/moving_mean:0',
# 'netG/up_3/norm_1/param_free_norm/moving_variance:0',
# 'netG/up_3/spectral_normalization_16/sn_u:0']
 
#def makePalatable(x):
#    x = x[5:].replace('/', '.')
#    return x
   
#from pprint import pprint   
#pprint(list(map(makePalatable, a)))