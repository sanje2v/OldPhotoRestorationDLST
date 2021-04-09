import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.layers import Conv2D, Lambda, LeakyReLU, BatchNormalization, ZeroPadding2D
from tensorflow.keras.layers.experimental import SyncBatchNormalization

from . import SPADE
from utils import *


class SPADEResnetBlock(tf.keras.layers.Layer):
    def __init__(self, opts, fin, fout, use_spade=True, use_spectral_norm=True, norm_layer=BatchNormalization, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.opts = opts
        self.fin = fin
        self.fout = fout
        self.use_spectral_norm = use_spectral_norm
        self.norm_layer = norm_layer

        fmiddle = min(fin, fout)
        norm_s = SPADE(opts, fin, opts.semantic_nc, name='norm_s') if fin != fout else Lambda(lambda x: x[0], trainable=False)
        conv_s = Conv2D(fout, kernel_size=1, use_bias=False, name='conv_s') if fin != fout else Lambda(lambda x: x, trainable=False)
        shortcut = [norm_s, conv_s]

        if use_spade:
            conv_0 = SPADE(opts, fin, opts.semantic_nc, norm_layer=SyncBatchNormalization, name='norm_0')
        else:
            conv_0 = Lambda(lambda x: x[0], trainable=False)

        conv_0 = [conv_0,
                  LeakyReLU(alpha=2e-1),
                  Conv2D(fmiddle, kernel_size=3, padding='same', name='conv_0')]

        if use_spade:
            conv_1 = SPADE(opts, fmiddle, opts.semantic_nc, norm_layer=SyncBatchNormalization, name='norm_1')
        else:
            conv_1 = Lambda(lambda x: x, trainable=False)

        conv_1 = [conv_1,
                  LeakyReLU(alpha=2e-1),
                  Conv2D(fout, kernel_size=3, padding='same', name='conv_1')]

        if use_spectral_norm:
            conv_0[-1] = SpectralNormalization(conv_0[-1], name='conv_0')
            conv_1[-1] = SpectralNormalization(conv_1[-1], name='conv_1')
            if fin != fout:
                shortcut[-1] = SpectralNormalization(conv_s, name='conv_s')

        self.inner_layers = [shortcut, conv_0, conv_1]


    def call(self, inputs, training):
        x, seg, degraded_image = inputs

        x_s = iterative_call(self.inner_layers[0], [x, seg, degraded_image], training=training)

        dx = iterative_call(self.inner_layers[1], [x, seg, degraded_image], training=training)
        dx = iterative_call(self.inner_layers[2], [dx, seg, degraded_image], training=training)

        return x_s + dx