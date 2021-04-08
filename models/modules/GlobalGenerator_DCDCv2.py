import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU, Lambda
from tensorflow.keras.activations import tanh

from ..layers import *


class GlobalGenerator_DCDCv2(tf.keras.layers.Layer):
    def _build_encoder(self, name):
        encoder =\
        [
            ReflectionPadding2D(padding=3),
            Conv2D(filters=min(self.ngf, self.opts.mc), kernel_size=7, padding='valid'),
            self.norm_layer(),
            self.activation_layer()
        ]

        for i in range(self.opts.start_r):
            mult = 2**i

            encoder.extend([Conv2D(filters=min(self.ngf * mult * 2, self.opts.mc),
                                   kernel_size=self.k_size,
                                   strides=2,
                                   padding='valid'),
                            self.norm_layer(),
                            self.activation_layer()])

        for i in range(self.opts.start_r, self.n_downsampling - 1):
            mult = 2**i

            encoder.extend([Conv2D(filters=min(self.ngf * mult * 2, self.opts.mc),
                                   kernel_size=self.k_size,
                                   strides=2,
                                   padding='valid'),
                            self.norm_layer(),
                            self.activation_layer(),
                            ResnetBlock(min(self.ngf * mult * 2, self.opts.mc),
                                        padding_type=self.padding_type,
                                        norm_layer=self.norm_layer,
                                        activation_layer=self.activation_layer),
                            ResnetBlock(min(self.ngf * mult * 2, self.opts.mc),
                                        padding_type=self.padding_type,
                                        norm_layer=self.norm_layer,
                                        activation_layer=self.activation_layer)])

        mult = 2**(self.n_downsampling - 1)

        if self.opts.spatio_size == 32:
            encoder.extend([Conv2D(filters=min(self.ngf * mult * 2, self.opts.mc),
                                   kernel_size=self.k_size,
                                   strides=2,
                                   padding='valid'),
                            self.norm_layer(),
                            self.activation_layer()])

        elif self.opts.spatio_size == 64:
            encoder.append(ResnetBlock(min(self.ngf * mult * 2, self.opts.mc),
                                       padding_type=self.padding_type,
                                       norm_layer=self.norm_layer,
                                       activation_layer=self.activation_layer))

        encoder.append(ResnetBlock(min(self.ngf * mult * 2, self.opts.mc),
                                   padding_type=self.padding_type,
                                   norm_layer=self.norm_layer,
                                   activation_layer=self.activation_layer))

        if self.opts.feat_dim > 0:
            encoder.append(Conv2D(filters=self.opts.feat_dim, kernel_size=1))

        return tf.keras.Sequential(encoder, name=name)

    def _build_decoder(self, name):
        decoder = []

        mult = 2**(self.n_downsampling - 1)

        if self.opts.feat_dim > 0:
            decoder.append(Conv2D(filters=min(self.ngf * mult * 2, self.opts.mc), kernel_size=1, padding='valid'))

        o_pad = 0 if self.k_size == 4 else 1
        mult = 2**self.n_downsampling

        decoder.append(ResnetBlock(min(self.ngf * mult, self.opts.mc),
                                   padding_type=self.padding_type,
                                   norm_layer=self.norm_layer,
                                   activation_layer=self.activation_layer))

        if self.opts.spatio_size == 32:
            decoder.extend([Conv2DTranspose(filters=min((self.ngf * mult) // 2, self.opts.mc),
                                            kernel_size=self.k_size,
                                            strides=2,
                                            padding='valid',
                                            output_padding=o_pad),
                            self.norm_layer(),
                            self.activation_layer()])

        elif self.opts.spatio_size == 64:
            decoder.append(ResnetBlock(min(self.ngf * mult, self.opts.mc),
                                       padding_type=self.padding_type,
                                       norm_layer=self.norm_layer,
                                       activation_layer=self.activation_layer))

        for i in range(1, self.n_downsampling - self.opts.start_r):
            mult = 2**(self.n_downsampling - i)

            decoder.extend([ResnetBlock(min(self.ngf * mult, self.opts.mc),
                                        padding_type=self.padding_type,
                                        norm_layer=self.norm_layer,
                                        activation_layer=self.activation_layer),
                            ResnetBlock(min(self.ngf * mult, self.opts.mc),
                                        padding_type=self.padding_type,
                                        norm_layer=self.norm_layer,
                                        activation_layer=self.activation_layer),
                            Conv2DTranspose(min((self.ngf * mult) // 2, self.opts.mc),
                                            kernel_size=self.k_size,
                                            strides=2,
                                            padding='valid',
                                            output_padding=o_pad),
                            self.norm_layer(),
                            self.activation_layer()])

        for i in range(self.n_downsampling - self.opts.start_r, self.n_downsampling):
            mult = 2**(self.n_downsampling - i)

            decoder.extend([Conv2DTranspose(filters=min((self.ngf * mult) // 2, self.opts.mc),
                                        kernel_size=self.k_size,
                                        strides=2,
                                        padding='valid',
                                        output_padding=o_pad),
                            self.norm_layer(),
                            self.activation_layer()])

        decoder.extend([ReflectionPadding2D(padding=3),
                        Conv2D(filters=self.output_nc, kernel_size=7, padding='valid')])

        if not self.opts.use_segmentation_model:
            decoder.append(Lambda(lambda x: tanh(x)))

        return tf.keras.Sequential(decoder, name=name)


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
            self.inner_layers = self._build_encoder(name='encoder')
        elif flow == 'dec':
            self.inner_layers = self._build_decoder(name='decoder')
        elif flow == 'enc_dec':
            self.inner_layers = tf.keras.Sequential([self._build_encoder(name='encoder'), self._build_decoder(name='decoder')])
        else:
            raise NotImplementedError("Unsupported flow specified!")

    def call(self, inputs, training):
        return self.inner_layers(inputs, training=training)