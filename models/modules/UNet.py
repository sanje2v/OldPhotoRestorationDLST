import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Lambda
from tensorflow.keras.activations import tanh

from ..layers import BlurPool2D, ReflectionPadding2D, UNetConvBlock, UNetUpBlock
from utils import *


class UNet(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, depth, conv_num, wf, padding,
                 norm_layer, up_mode='upsample', with_tanh=False, antialiasing=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert depth > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth - 1
        self.conv_num = conv_num
        self.wf = wf
        self.padding = padding
        self.norm_layer = norm_layer
        self.up_mode = up_mode
        self.with_tanh = with_tanh
        self.antialiasing = antialiasing

        first_layers = [ReflectionPadding2D(padding=3),
                        Conv2D(2**wf, kernel_size=7, padding='valid', name='first'),
                        LeakyReLU(alpha=0.2)]

        prev_channels = 2**wf
        down_sample = []
        down_path = []
        for i in range(depth):
            if antialiasing and depth > 0:
                down_sample.append([ReflectionPadding2D(padding=1),
                                    Conv2D(prev_channels, kernel_size=3, strides=1, padding='valid', name='down_sample'),
                                    BatchNormalization(),
                                    LeakyReLU(alpha=0.2),
                                    BlurPool2D(stride=2)])
            else:
                down_sample.append([ReflectionPadding2D(padding=1),
                                    Conv2D(prev_channels, kernel_size=4, strides=2, padding='valid', name='down_sample'),
                                    BatchNormalization(),
                                    LeakyReLU(alpha=0.2)])

            down_path.append(UNetConvBlock(conv_num, 2**(wf + i + 1), padding, norm_layer))
            prev_channels = 2**(wf+ i + 1)

        up_path = []
        for i in reversed(range(depth)):
            up_path.append(UNetUpBlock(conv_num, 2**(wf + i), up_mode, padding, norm_layer))
            prev_channels = 2**(wf + i)

        last_layers = [ReflectionPadding2D(padding=1),
                       Conv2D(out_channels, kernel_size=3, padding='valid', name='last')]
        if with_tanh:
            last_layers.append(Lambda(lambda x: tanh(x), trainable=False))

        self.inner_layers = [first_layers, down_sample, down_path, up_path, last_layers]


    def call(self, inputs, training):
        x = iterative_call(self.inner_layers[0], inputs, training=training)

        blocks = []
        for i, down_block in enumerate(self.inner_layers[2]):
            blocks.append(x)
            x = iterative_call(self.inner_layers[1][i], x, training=training)
            x = down_block(x, training=training)

        for i, up in enumerate(self.inner_layers[3]):
            x = up([x, blocks[-i - 1]], training=training)

        return iterative_call(self.inner_layers[4], x, training=training)