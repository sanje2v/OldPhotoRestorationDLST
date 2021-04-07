import tensorflow as tf

from .modules import *


class ImageEnhancer(tf.keras.Model):
    def __init__(self, opts, *args, **kwargs):
        super().__init__(*args, **kwargs)

        netG_input_nc = opts.label_nc if opts.label_nc != 0 else opts.input_nc
        self.netG_A = GlobalGenerator_DCDCv2(opts,
                                             'enc',
                                             netG_input_nc,
                                             opts.output_nc,
                                             opts.ngf,
                                             opts.k_size,
                                             opts.n_downsample_global,
                                             norm_layer=opts.norm,
                                             name='netG_A')
        self.netG_B = GlobalGenerator_DCDCv2(opts,
                                             'dec',
                                             netG_input_nc,
                                             opts.output_nc,
                                             opts.ngf,
                                             opts.k_size,
                                             opts.n_downsample_global,
                                             norm_layer=opts.norm,
                                             name='netG_B')

        mapping_module_class = MappingModuleWithMask if opts.non_local == 'Settings_42' else MappingModule
        self.mapping_net = mapping_module_class(opts,
                                                min(opts.ngf * 2**opts.n_downsample_global, opts.mc),
                                                opts.map_mc,
                                                opts.mapping_n_block,
                                                norm_layer=opts.norm,
                                                name='mapping_net')

    def call(self, inputs, training=False):
        if training:
            raise NotImplementedError("Training '{:s}' instance is NOT supported yet.".format(self.__class__.name))

        input_concat, _ = inputs
        if len(input_concat.shape) != 4:
            input_concat = tf.expand_dims(input_concat, axis=0)

        label_feature = self.netG_A(input_concat, training=training)
        label_feature_map = self.mapping_net(label_feature, training=training)
        return self.netG_B(label_feature_map, training=training)