import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Lambda, ReLU, BatchNormalization, ZeroPadding2D


class SPADE(tf.keras.layers.Layer):
    def __init__(self, opts, norm_nc, label_nc, norm_layer=BatchNormalization, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.opts = opts
        self.norm_nc = norm_nc
        self.label_nc = label_nc
        self.norm_layer = norm_layer

        ks = 3  # spadesyncbatch3x3
        nhidden = 128
        pw = ks // 2

        mlp_shared = tf.keras.Sequential([Lambda(lambda x: x[1] if opts.no_parsing_map else tf.concat([x[0], x[1]], axis=3)),
                                          ZeroPadding2D(padding=pw),
                                          Conv2D(nhidden, kernel_size=ks, padding='valid'),
                                          ReLU()])
        mlp_gamma = tf.keras.Sequential([ZeroPadding2D(padding=pw),
                                         Conv2D(norm_nc, kernel_size=ks, padding='valid')])
        mlp_beta = tf.keras.Sequential([ZeroPadding2D(padding=pw),
                                        Conv2D(norm_nc, kernel_size=ks, padding='valid')])
        param_free_norm = self.norm_layer(center=False, scale=False)

        self.inner_layer = [mlp_shared, mlp_gamma, mlp_beta, param_free_norm]

    def call(self, inputs, training):
        x, segmap, degraded_image = inputs

        segmap = tf.image.resize(segmap, size=x.shape[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        degraded_image = tf.image.resize(degraded_image, size=x.shape[1:3], method=tf.image.ResizeMethod.BILINEAR)

        actv = self.inner_layer[0]([segmap, degraded_image], training=training)
        gamma = self.inner_layer[1](actv, training=training)
        beta = self.inner_layer[2](actv, training=training)

        # Generate parameter free normalized activations
        normalized = self.inner_layer[3](x, training=training)

        return normalized * (1 + gamma) + beta