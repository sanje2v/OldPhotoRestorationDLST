import tensorflow as tf


class GlobalGenerator_DCDCv2(tf.keras.Model):
    def __init__(self, input_nc, output_nc, ngf=64, k_size=3, n_downsampling=8, norm_layer=tf.keras.layers.BatchNormalization, activation_layer=tf.keras.activations.relu, padding_type='reflect'):
        super().__init__()

