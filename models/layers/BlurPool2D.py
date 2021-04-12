import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D

from .ReflectionPadding2D import ReflectionPadding2D
from .ReplicationPadding2D import ReplicationPadding2D


# Original source: https://github.com/IsaacCorley/Making-Convolutional-Networks-Shift-Invariant-Again-Tensorflow
class BlurPool2D(tf.keras.layers.Layer):
    """
    Implementation of:
    https://arxiv.org/abs/1904.11486 https://github.com/adobe/antialiased-cnns
    """
    def __init__(self, pad_type='reflect', kernel_size=3, stride=2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pad_type = pad_type.casefold()
        self.kernel_size = kernel_size
        self.strides = (1, stride, stride, 1)
        self.paddings = ((int(1 * (kernel_size - 1) / 2), int(np.ceil(1 * (kernel_size - 1) / 2))))

        KERNEL_MAPPING =\
        {
            1: [1,],
            2: [1, 1],
            3: [1, 2, 1],
            4: [1, 3, 3, 1],
            5: [1, 4, 6, 4, 1],
            6: [1, 5, 10, 10, 5, 1],
            7: [1, 6, 15, 20, 15, 6, 1]
        }
        self.kernel = np.array(KERNEL_MAPPING[self.kernel_size], dtype=np.float32)


    def compute_output_shape(self, input_shape):
        height = input_shape[1] // self.strides[0]
        width = input_shape[2] // self.strides[1] 
        channels = input_shape[3]
        return (input_shape[0], height, width, channels)

    def call(self, inputs, training):
        k = self.kernel
        k = k[:, None] * k[None, :]
        k = k / tf.reduce_sum(k)
        k = tf.tile(k[:, :, None, None], (1, 1, inputs.shape[-1], 1))
        if self.pad_type == 'reflect':
            x = ReflectionPadding2D(padding=self.paddings)(inputs)
        elif self.pad_type == 'replicate':
            x = ReplicationPadding2D(padding=self.paddings)(inputs)
        else:
            x = ZeroPadding2D(padding=self.paddings)(inputs)
        x = tf.nn.depthwise_conv2d(x, k, strides=self.strides, padding='VALID')
        return x