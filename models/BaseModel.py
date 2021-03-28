import tensorflow as tf


class BaseModel(tf.keras.Model):
    def __init__(self, opts, *args, **kwargs):
        super().__init__(*args, **kwargs)