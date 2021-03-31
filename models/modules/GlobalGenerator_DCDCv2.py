import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU, Lambda
from tensorflow.keras.activations import tanh

from ..layers import *


class GlobalGenerator_DCDCv2(tf.keras.layers.Layer):

    def _build_encoder(self, name):
        encoder = tf.keras.Sequential(\
        [
            ReflectionPadding2D(padding=3),
            Conv2D(filters=min(self.ngf, self.opts.mc), kernel_size=7, padding='valid'),
            self.norm_layer(),
            self.activation_layer()
        ], name=name)

        for i in range(self.opts.start_r):
            mult = 2**i

            encoder.add(Conv2D(filters=min(self.ngf * mult * 2, self.opts.mc),
                               kernel_size=self.k_size,
                               strides=2,
                               padding='same'))
            encoder.add(self.norm_layer())
            encoder.add(self.activation_layer())

        for i in range(self.opts.start_r, self.n_downsampling - 1):
            mult = 2**i

            encoder.add(Conv2D(filters=min(self.ngf * mult * 2, self.opts.mc),
                               kernel_size=self.k_size,
                               strides=2,
                               padding='same'))
            encoder.add(self.norm_layer())
            encoder.add(self.activation_layer())
            encoder.add(ResnetBlock(min(self.ngf * mult * 2, self.opts.mc),
                                    padding_type=self.padding_type,
                                    norm_layer=self.norm_layer,
                                    activation_layer=self.activation_layer))
            encoder.add(ResnetBlock(min(self.ngf * mult * 2, self.opts.mc),
                                    padding_type=self.padding_type,
                                    norm_layer=self.norm_layer,
                                    activation_layer=self.activation_layer))

        mult = 2**(self.n_downsampling - 1)

        if self.opts.spatio_size == 32:
            encoder.add(Conv2D(filters=min(self.ngf * mult * 2, self.opts.mc),
                               kernel_size=self.k_size,
                               strides=2,
                               padding='same'))
            encoder.add(self.norm_layer())
            encoder.add(self.activation_layer())

        elif self.opts.spatio_size == 64:
            encoder.add(ResnetBlock(min(self.ngf * mult * 2, self.opts.mc),
                                    padding_type=self.padding_type,
                                    norm_layer=self.norm_layer,
                                    activation_layer=self.activation_layer))

        encoder.add(ResnetBlock(min(self.ngf * mult * 2, self.opts.mc),
                                padding_type=self.padding_type,
                                norm_layer=self.norm_layer,
                                activation_layer=self.activation_layer))

        if self.opts.feat_dim > 0:
            encoder.add(Conv2D(filters=self.opts.feat_dim, kernel_size=1))

        return encoder

    def _build_decoder(self, name):
        decoder = tf.keras.Sequential(name=name)

        mult = 2**(self.n_downsampling - 1)

        if self.opts.feat_dim > 0:
            decoder.add(Conv2D(filters=min(self.ngf * mult * 2, self.opts.mc), kernel_size=1))

        o_pad = 0 if self.k_size == 4 else 1
        mult = 2**self.n_downsampling

        decoder.add(ResnetBlock(min(self.ngf * mult, self.opts.mc),
                                padding_type=self.padding_type,
                                norm_layer=self.norm_layer,
                                activation_layer=self.activation_layer))

        if self.opts.spatio_size == 32:
            decoder.add(Conv2DTranspose(filters=min((self.ngf * mult) // 2, self.opts.mc),
                                        kernel_size=self.k_size,
                                        strides=2,
                                        padding='same',
                                        output_padding=o_pad))
            decoder.add(self.norm_layer())
            decoder.add(self.activation_layer())

        elif self.opts.spatio_size == 64:
            decoder.add(ResnetBlock(min(self.ngf * mult, self.opts.mc),
                                    padding_type=self.padding_type,
                                    norm_layer=self.norm_layer,
                                    activation_layer=self.activation_layer))

        for i in range(1, self.n_downsampling - self.opts.start_r):
            mult = 2**(self.n_downsampling - i)

            decoder.add(ResnetBlock(min(self.ngf * mult, self.opts.mc),
                                    padding_type=self.padding_type,
                                    norm_layer=self.norm_layer,
                                    activation_layer=self.activation_layer))
            decoder.add(ResnetBlock(min(self.ngf * mult, self.opts.mc),
                                    padding_type=self.padding_type,
                                    norm_layer=self.norm_layer,
                                    activation_layer=self.activation_layer))
            decoder.add(Conv2DTranspose(min((self.ngf * mult) // 2, self.opts.mc),
                                        kernel_size=self.k_size,
                                        strides=2,
                                        padding='valid',
                                        output_padding=o_pad))
            decoder.add(self.norm_layer())
            decoder.add(self.activation_layer())

        for i in range(self.n_downsampling - self.opts.start_r, self.n_downsampling):
            mult = 2**(self.n_downsampling - i)

            decoder.add(Conv2DTranspose(filters=min((self.ngf * mult) // 2, self.opts.mc),
                                        kernel_size=self.k_size,
                                        strides=2,
                                        padding='valid',
                                        output_padding=o_pad))
            decoder.add(self.norm_layer())
            decoder.add(self.activation_layer())

        decoder.add(ReflectionPadding2D(padding=3))
        decoder.add(Conv2D(filters=self.output_nc, kernel_size=7, padding='valid'))

        if not self.opts.use_segmentation_model:
            decoder.add(Lambda(lambda x: tanh(x)))

        return decoder


    def __init__(self, opts, flow, input_nc, output_nc, ngf=64, k_size=3, n_downsampling=8, padding_type='reflect',
                 norm_layer=BatchNormalization, activation_layer=ReLU, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.opts = opts
        self.flow = flow
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.k_size = k_size
        self.n_downsampling = n_downsampling
        self.padding_type = padding_type
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer

        if flow == 'enc':
            self.inner_layer = self._build_encoder(name='encoder')
        elif flow == 'dec':
            self.inner_layer = self._build_decoder(name='decoder')
        elif flow == 'enc_dec':
            self.inner_layer = tf.keras.Sequential([self._build_encoder(name='encoder'), self._build_decoder(name='decoder')])
        else:
            raise NotImplementedError("Unsupported flow specified!")

    def call(self, inputs):
        return self.inner_layer(inputs)