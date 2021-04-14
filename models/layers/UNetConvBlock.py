import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU

from utils import *
from .ReflectionPadding2D import ReflectionPadding2D


class UNetConvBlock(tf.keras.layers.Layer):
    def __init__(self, conv_num, out_size, padding, norm_layer, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_num = conv_num
        self.out_size = out_size
        self.padding = padding
        self.norm_layer = norm_layer

        block = []
        for _ in range(conv_num):
            block.extend([ReflectionPadding2D(padding=padding),
                          Conv2D(out_size, kernel_size=3, padding='valid', name='block')])
            if norm_layer:
                block.append(norm_layer())
            block.append(LeakyReLU(alpha=0.2))

        self.inner_layers = block


    def call(self, inputs, training):
        return iterative_call(self.inner_layers, inputs, training=training)