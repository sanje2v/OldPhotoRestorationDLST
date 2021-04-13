import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU

from ..layers import ResnetBlock
from .NonLocalBlock2D_with_mask_Res import NonLocalBlock2D_with_mask_Res
from utils import *


class MappingModule(tf.keras.layers.Layer):
    def __init__(self, opts, use_mask, nc, mc=64, n_blocks=3, padding_type='reflect',
                 norm_layer=BatchNormalization, activation_layer=ReLU, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.opts = opts
        self.use_mask = use_mask
        self.nc = nc
        self.mc = mc
        self.n_blocks = n_blocks
        self.padding_type = padding_type
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer

        self.inner_layers = []
        tmp_nc = 64
        n_up = 4

        for i in range(n_up):
            ic = min(tmp_nc * 2**i, mc)
            oc = min(tmp_nc * 2**(i + 1), mc)
            self.inner_layers.extend([Conv2D(oc, kernel_size=3, padding='same'),
                                      self.norm_layer(),
                                      self.activation_layer()])

        self.index_before_NL_res = len(self.inner_layers) - 1
        if use_mask:
            self.inner_layers.append(NonLocalBlock2D_with_mask_Res(mc,
                                                                   mc,
                                                                   'combine',
                                                                   opts.correlation_renormalize,
                                                                   opts.softmax_temperature,
                                                                   opts.use_self,
                                                                   opts.cosin_similarity))

        for i in range(self.n_blocks):
            self.inner_layers.append(ResnetBlock(mc,
                                                 padding_type=self.padding_type,
                                                 norm_layer=self.norm_layer,
                                                 activation_layer=self.activation_layer,
                                                 dilation=self.opts.mapping_net_dilation))

        for i in range(n_up - 1):
            ic = min(64 * 2**(4 - i), self.mc)
            oc = min(64 * 2**(3 - i), self.mc)
            self.inner_layers.extend([Conv2D(oc, kernel_size=3, padding='same'),
                                      self.norm_layer(),
                                      self.activation_layer()])
        self.inner_layers.append(Conv2D(tmp_nc, kernel_size=3, padding='same'))

        if self.opts.feat_dim > 0 and opt.feat_dim < 64:
            self.inner_layers.extend([self.norm_layer(),
                                      self.activation_layer(),
                                      Conv2D(opts.feat_dim, kernel_size=1, padding='same')])

        setattr(self.inner_layers, 'name', 'model')

    def call(self, inputs, training):
        x, mask = inputs

        if self.use_mask:
            x = iterative_call(self.inner_layers[:(self.index_before_NL_res+1)], x, training=training)
            x = self.inner_layers[self.index_before_NL_res+1]([x, mask], training=training)
            x = iterative_call(self.inner_layers[(self.index_before_NL_res+2):], x, training=training)
        else:
            x = iterative_call(self.inner_layers, x, training=training)
        return x