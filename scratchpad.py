from models import FaceEnhancer
import test_options as opts
import numpy as np
import tensorflow as tf

FaceEnhancer(opts)([tf.convert_to_tensor(np.random.randint(0, 5, (1, 5, 5, 2), dtype=np.int32)), tf.convert_to_tensor(np.random.randn(1, 5, 5, 2))])