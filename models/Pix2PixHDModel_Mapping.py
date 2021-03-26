import tensorflow as tf

from .BaseModel import BaseModel
from .modules import *


class Pix2PixHDModel_Mapping(BaseModel):
    def __init__(self, opts):
        super().__init__(opts)

        netG_input_nc = opts.label_nc if opts.label_nc != 0 else opts.input_nc

        self.netG_A = GlobalGenerator_DCDCv2(opts,
                                             netG_input_nc,
                                             opts.output_nc,
                                             opts.ngf,
                                             opts.k_size,
                                             opts.n_downsample_global,
                                             norm_layer=opts.norm)
        self.netG_B = GlobalGenerator_DCDCv2(opts,
                                             netG_input_nc,
                                             opts.output_nc,
                                             opts.ngf,
                                             opts.k_size,
                                             opts.n_downsample_global,
                                             norm_layer=opts.norm)

        mapping_module_class = MappingModuleWithMask if opts.non_local == 'Settings_42' else MappingModule
        self.mapping_net = mapping_module_class(opts, min(opts.ngf * 2**opts.n_downsample_global, opts.mc), opts.map_mc, opts.mapping_n_block)

    def call(self, inputs, training=False):
        if training:
            raise NotImplementedError("Training 'Pix2PixHDModel_Mapping' instance is NOT supported yet.")

        label, instance = inputs

        label_feature = self.netG_A.predict({'input': label, 'flow': 'enc'})

