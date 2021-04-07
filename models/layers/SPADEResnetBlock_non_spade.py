import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Lambda, UpSampling2D, LeakyReLU


class SPADEResnetBlock_non_spade(object):
    def __init__(self, fin, fout, opts, *args, **kwargs):
        super().__init__(*args, **kwargs)