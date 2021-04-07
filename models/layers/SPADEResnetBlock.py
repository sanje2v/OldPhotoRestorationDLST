import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, Conv2D, Lambda, UpSampling2D, LeakyReLU, BatchNormalization, ZeroPadding2D
from tensorflow.keras.layers.experimental import SyncBatchNormalization

from . import SPADE


class SPADEResnetBlock(tf.keras.layers.Layer):
    def __init__(self, opts, fin, fout, use_spade=True, use_spectral_norm=True, norm_layer=BatchNormalization, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.opts = opts
        self.fin = fin
        self.fout = fout
        self.use_spectral_norm = use_spectral_norm
        self.norm_layer = norm_layer

        fmiddle = min(fin, fout)
        conv_0 = tf.keras.Sequential([ZeroPadding2D(padding=1), Conv2D(fmiddle, kernel_size=3, padding='same')])
        conv_1 = tf.keras.Sequential([ZeroPadding2D(padding=1), Conv2D(fout, kernel_size=3, padding='same')])
        conv_s = Conv2D(fout, kernel_size=1, use_bias=False) if fin != fout else Lambda(lambda x: x)

        if use_spectral_norm:
            conv_0 = tfa.layers.SpectralNormalization(conv_0.layers[-1])
            conv_1 = tfa.layers.SpectralNormalization(conv_1.layers[-1])
            conv_s = tfa.layers.SpectralNormalization(conv_s) if fin != fout else Lambda(lambda x: x)

        norm_0 = SPADE(opts, fin, opts.semantic_nc)
        norm_1 = SPADE(opts, fmiddle, opts.semantic_nc)
        norm_s = SPADE(opts, fin, opts.semantic_nc) if fin != fout else Lambda(lambda x: x[0])

        self.inner_layer = []
        self.inner_layer.append(tf.keras.Sequential([norm_s, conv_s]) if fin != fout else Lambda(lambda x: x[0]))
        self.inner_layer.extend([tf.keras.Sequential([norm_0, LeakyReLU(alpha=2e-1), conv_0]),
                                 tf.keras.Sequential([norm_1, LeakyReLU(alpha=2e-1), conv_1])])

    def call(self, inputs, training):
        x, seg, degraded_image = inputs

        x_s = self.inner_layer[0]([x, seg, degraded_image], training=training)

        dx = self.inner_layer[1]([x, seg, degraded_image], training=training)
        dx = self.inner_layer[2]([dx, seg, degraded_image], training=training)

        return x_s + dx