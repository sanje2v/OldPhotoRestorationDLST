import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, Lambda
from tensorflow.keras.activations import softmax
from tensorflow_addons.layers import InstanceNormalization

from ..layers import ResnetBlock
from utils import *


class NonLocalBlock2D_with_mask_Res(tf.keras.layers.Layer):
    def __init__(self, in_channels, inter_channels, mode='add', re_norm=False, temperature=1.0, use_self=False, cosin=False,
                 norm_layer=InstanceNormalization, activation_layer=ReLU, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert temperature != 0.

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.mode = mode
        self.re_norm = re_norm
        self.temperature = temperature
        self.use_self = use_self
        self.cosin = cosin
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer

        calc_g_x = [Conv2D(self.inter_channels, kernel_size=1, strides=1, padding='valid'),
                    Lambda(lambda x: tf.transpose(tf.reshape(x, (x.shape[0], self.inter_channels, -1)), perm=(0, 2, 1)), trainable=False)]
        calc_theta_x = [Conv2D(self.inter_channels, kernel_size=1, strides=1, padding='valid'),
                        Lambda(lambda x: tf.transpose(tf.reshape(x, (x.shape[0], self.inter_channels, -1)), perm=(0, 2, 1)), trainable=False),
                        Lambda(lambda x: tf.linalg.norm(x, ord=2, axis=2) if self.cosin else x, trainable=False)]
        calc_phi_x = [Conv2D(self.inter_channels, kernel_size=1, strides=1, padding='valid'),
                      Lambda(lambda x: tf.reshape(x, (x.shape[0], self.inter_channels, -1)), trainable=False),
                      Lambda(lambda x: tf.linalg.norm(x, ord=2, axis=1) if self.cosin else x, trainable=False)]
        calc_W = [Conv2D(self.in_channels, kernel_size=1, strides=1, padding='valid')]
        res_blocks = [ResnetBlock(self.inter_channels,
                                  padding_type='reflect',
                                  norm_layer=self.norm_layer,
                                  activation_layer=self.activation_layer) for _ in range(3)]

        self.inner_layers = [calc_g_x, calc_theta_x, calc_phi_x, calc_W, res_blocks]

    def __call__(self, inputs, training):
        x, mask = inputs

        g_x = iterative_call(self.inner_layers[0], x, training=training)
        theta_x = iterative_call(self.inner_layers[1], x, training=training)
        phi_x = iterative_call(self.inner_layers[2], x, training=training)

        f_div_C =  softmax(tf.linalg.matmul(theta_x, phi_x) / self.temperature, axis=2)

        inverted_mask = 1. - mask
        mask = tf.image.resize(mask, size=x.shape[1:3], method=tf.image.ResizeMethod.BILINEAR)
        mask = tf.where(tf.greater(mask, 0.), tf.zeros_like(mask), mask)
        mask = 1. - mask

        inverted_mask = tf.image.resize(inverted_mask, size=(x.shape[1:3]), method=tf.image.ResizeMethod.BILINEAR)
        mask *= inverted_mask

        mask_expand = tf.reshape(mask, (x.shape[0], 1, -1))
        mask_expand = tf.repeat(mask_expand, x.shape[1] * x.shape[2], axis=1)
        if self.use_self:
            mask_expand[:, range(x.shape[1] * x.shape[2]), range(x.shape[1] * x.shape[2])] = 1.0

        f_div_C = mask_expand * f_div_C
        if self.re_norm:
            f_div_C = tf.keras.utils.normalize(f_div_C, axis=2, order=1)

        y = tf.reshape(tf.linalg.matmul(f_div_C, g_x), (x.shape[0], *x.shape[1:3], self.inter_channels))

        W_y = iterative_call(self.inner_layers[3], y, training=training)
        W_y = iterative_call(self.inner_layers[4], W_y, training=training)

        assert self.mode.casefold() == 'combine'
        full_mask = tf.repeat(mask, self.inter_channels, axis=3)
        z = full_mask * x + (1 - full_mask) * W_y
        return z