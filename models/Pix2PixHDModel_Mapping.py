import tensorflow as tf

import settings
from .BaseModel import BaseModel


class Pix2PixHDModel_Mapping(BaseModel):
    def __init__(self):
        super().__init__()

        self.netG_A = GlobalGenerator_DCDCv2(netG_input_nc, output_nc, ngf, k_size, n_downsample_global, settings.BATCH_NORM_CLASS)
        self.netG_B = GlobalGenerator_DCDCv2(netG_input_nc, output_nc, ngf, k_size, n_downsample_global, settings.BATCH_NORM_CLASS)

        mapping_model_class = Mapping_Model_with_mask if non_local == 'Settings_42' else Mapping_Model
        self.mapping_net = mapping_model_class(min(ngf * 2**n_downsample_global, mc), map_mc, mapping_n_block)

    def call(self, inputs, training=False):
        if training:
            raise NotImplementedError("Training 'Pix2PixHDModel_Mapping' instance is NOT supported yet.")

        label, inatance = inputs

        label_feature = self.netG_A.predict({'input': label, 'flow': 'enc'})

