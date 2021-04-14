import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU, LeakyReLU, Lambda
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.activations import tanh, sigmoid

from .layers import ReflectionPadding2D
from .modules import UNet
from utils import *


class ScratchDetector(tf.keras.Model):
    """Scratch detection using scratch mask prediction with UNet"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        scratch_mask_model = UNet(in_channels=1,
                                  out_channels=1,
                                  depth=4,
                                  conv_num=2,
                                  wf=6,
                                  padding=1,
                                  norm_layer=BatchNormalization,
                                  up_mode='upsample',
                                  with_tanh=False,
                                  antialiasing=True)

        self.inner_layers = [scratch_mask_model]


    def call(self, inputs, training=False):
        if training:
            raise NotImplementedError("Training '{:s}' instance is NOT supported yet.".format(self.__class__.name))

        if len(inputs.shape) != 4:
            inputs = tf.expand_dims(inputs, axis=0)

        x = sigmoid(iterative_call(self.inner_layers, inputs, training=training))
        x = tf.where(tf.greater_equal(x, 0.4), tf.ones_like(x, dtype=tf.dtypes.int32), tf.zeros_like(x, dtype=tf.dtypes.int32))
        return x