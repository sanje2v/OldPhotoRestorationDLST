import tensorflow as tf


# Ref: https://stackoverflow.com/questions/50677544/reflection-padding-conv2d
class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding, **kwargs):
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        if s[1] == None:
            return (None, None, None, s[3])
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = super(ReflectionPadding2D, self).get_config()
        return config