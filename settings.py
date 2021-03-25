import collections
import tensorflow_addons as tfa


version_tuple = collections.namedtuple("Row", ["major", "minor"])
MIN_PYTHON_VERSION = version_tuple(major=3, minor=7)
MIN_TENSORFLOW_VERSION = version_tuple(major=2, minor=4)
MIN_NUMPY_VERSION = version_tuple(major=1, minor=19)


# Default commandline values
PROJECT_DESCRIPTION = "Tensorflow implementation of the project 'Bringing Old Photos to Life'"
DEFAULT_OUTPUT_FOLDER = "outputs"
LOAD_SIZE = 256
LABEL_NC = 18

BATCH_NORM_CLASS = tfa.layers.InstanceNormalization