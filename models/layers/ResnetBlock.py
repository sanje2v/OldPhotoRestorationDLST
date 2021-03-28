import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU, Lambda, Dropout

from . import ReflectionPadding2D


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, dim, padding_type, norm_layer, activation_layer, use_dropout=False, dilation=1):
        super().__init__()

        self.dim = dim
        self.padding_type = padding_type
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.use_dropout = use_dropout
        self.dilation = dilation

    def build(self, input_shape):
        self.conv_block = tf.keras.Sequential()

        padding = (0, 0)
        if self.padding_type in ['reflect', 'replicate']:
            self.conv_block.add(ReflectionPadding2D(self.dilation))
        elif self.padding_type == 'zero':
            padding = (self.dilation, self.dilation)

        self.conv_block.add(Conv2D(self.dim, kernel_size=3, padding=padding, dilation=self.dilation))
        self.conv_block.add(self.norm_layer())
        self.conv_block.add(self.activation_layer())

        if self.use_dropout:
            self.conv_block.add(Dropout(rate=0.5))

        padding = (0, 0)
        if self.padding_type in ['reflect', 'replicate']:
            self.conv_block.add(ReflectionPadding2D(padding=1))
        elif self.padding_type == 'zero':
            padding = 1

        self.conv_block.add(Conv2D(filters=self.dim, kernel_size=3, padding=padding))
        self.conv_block.add(self.norm_layer())

    def get_config(self):
        pass

    def call(self, inputs):
        return inputs + self.conv_block(inputs)