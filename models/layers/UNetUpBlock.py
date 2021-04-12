import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D

from utils import *
from .ReflectionPadding2D import ReflectionPadding2D
from .UNetConvBlock import UNetConvBlock


class UNetUpBlock(tf.keras.layers.Layer):
    @staticmethod
    def center_crop(layer, target_size):
        _, layer_height, layer_width, _ = layer.shape
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return tf.image.crop_to_bounding_box(layer,
                                             diff_y,
                                             diff_x,
                                             (diff_y + target_size[0]),
                                             (diff_x + target_size[1]))


    def __init__(self, conv_num, out_size, up_mode, padding, norm_layer, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        up_mode = up_mode.casefold()
        if up_mode == 'upconv':
            up = [Conv2DTranspose(out_size, kernel_size=2, strides=2, padding='valid', name='up')]
        else:
            up = [UpSampling2D(size=2, interpolation='bilinear'),
                  ReflectionPadding2D(padding=1),
                  Conv2D(out_size, kernel_size=3, padding='valid', name='up')]
        conv_block = UNetConvBlock(conv_num, out_size, padding, norm_layer, name='conv_block')

        self.inner_layers = [up, conv_block]

    def call(self, inputs, training):
        x, bridge = inputs

        up = iterative_call(self.inner_layers[0], x, training=training)
        crop = UNetUpBlock.center_crop(bridge, up.shape[1:3])
        out = tf.concat([up, crop], axis=3)
        return self.inner_layers[1](out, training)