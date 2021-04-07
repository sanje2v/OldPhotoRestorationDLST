import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Lambda, UpSampling2D, LeakyReLU, ZeroPadding2D
from tensorflow.keras.activations import tanh

from .layers import SPADEResnetBlock, SPADEResnetBlock_non_spade


class FaceEnhancer(tf.keras.Model):
    def _compute_latent_vector_size(self, num_upsampling_layers, crop_size, aspect_ratio):
        if num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif num_upsampling_layers == 'more':
            num_up_layers = 6
        elif num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError("Unsupported opts.num_upsampling-layers!")

        sw = crop_size // 2**num_up_layers
        sh = round(sw / aspect_ratio)
        return sw, sh

    def __init__(self, opts, *args, **kwargs):
        super().__init__(name='netG', *args, **kwargs)

        self.opts = opts

        opts.num_upsampling_layers = opts.num_upsampling_layers.casefold()
        self.sw, self.sh = self._compute_latent_vector_size(opts.num_upsampling_layers.casefold(),
                                                            opts.crop_size,
                                                            opts.aspect_ratio)

        # Build SPADEGenerator
        netG = []
        netG.append(tf.keras.Sequential([Lambda(lambda x: tf.image.resize(x, (self.sh, self.sw), method=tf.image.ResizeMethod.BILINEAR)),
                                         #ZeroPadding2D(padding=1),
                                         Conv2D(16 * opts.ngf, kernel_size=3, padding='same', name='fc')]))

        opts.injection_layer = opts.injection_layer.casefold()
        use_spade = opts.injection_layer in ['all' , '1']
        netG.append(tf.keras.Sequential([SPADEResnetBlock(opts, 16 * opts.ngf, 16 * opts.ngf, use_spade=use_spade, name='head_0'),
                                         UpSampling2D(size=2)]))

        name = ['G_middle_{:d}'.format(i) for i in range(2)]
        use_spade = opts.injection_layer in ['all', '2']
        netG.append(tf.keras.Sequential([SPADEResnetBlock(opts, 16 * opts.ngf, 16 * opts.ngf, use_spade=use_spade, name=name[0]),
                                         UpSampling2D(size=2) if opts.num_upsampling_layers in ['more', 'most'] else Lambda(lambda x: x)]))
        netG.append(tf.keras.Sequential([SPADEResnetBlock(opts, 16 * opts.ngf, 16 * opts.ngf, use_spade=use_spade, name=name[1]),
                                         UpSampling2D(size=2)]))

        use_spade = opts.injection_layer in ['all', '3']
        netG.append(tf.keras.Sequential([SPADEResnetBlock(opts, 16 * opts.ngf, 8 * opts.ngf, use_spade=use_spade, name='up_0'),
                                         UpSampling2D(size=2)]))

        use_spade = opts.injection_layer in ['all', '4']
        netG.append(tf.keras.Sequential([SPADEResnetBlock(opts, 8 * opts.ngf, 4 * opts.ngf, use_spade=use_spade, name='up_1'),
                                         UpSampling2D(size=2)]))

        use_spade = opts.injection_layer in ['all', '5']
        netG.append(tf.keras.Sequential([SPADEResnetBlock(opts, 4 * opts.ngf, 2 * opts.ngf, use_spade=use_spade, name='up_2'),
                                         UpSampling2D(size=2)]))

        use_spade = opts.injection_layer in ['all', '6']
        netG.append(tf.keras.Sequential([SPADEResnetBlock(opts, 2 * opts.ngf, 1 * opts.ngf, use_spade=use_spade, name='up_3'),
                                         UpSampling2D(size=2) if opts.num_upsampling_layers == 'most' else Lambda(lambda x: x)]))

        if opts.num_upsampling_layers == 'most':
            netG.append(SPADEResnetBlock(1 * opts.ngf, opts.ngf // 2, opts, name='up_4'))
            final_nc = opts.ngf // 2
        else:
            final_nc = opts.ngf

        netG.append(tf.keras.Sequential([LeakyReLU(alpha=2e-1),
                                         Conv2D(3, kernel_size=3, padding='valid', name='conv_img'),
                                         Lambda(lambda x: tanh(x))]))

        self.inner_layer = netG


    def call(self, inputs, training=False):
        if training:
            raise NotImplementedError("Training '{:s}' instance is NOT supported yet.".format(self.__class__.name))

        seg, degraded_image = inputs

        if len(seg.shape) != 4:
            seg = tf.expand_dims(seg, axis=0)

        if len(degraded_image.shape) != 4:
            degraded_image = tf.expand_dims(degraded_image, axis=0)

        x = self.inner_layer[0](degraded_image, training=training)

        for i in range(1, len(self.inner_layer) - 1):
            print(i)
            x = self.inner_layer[i]([x, seg, degraded_image], training=training)

        return self.inner_layer[-1](x, training=training)