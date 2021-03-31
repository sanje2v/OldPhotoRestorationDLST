import os.path
import collections
import tensorflow as tf
import tensorflow_addons as tfa


version_tuple = collections.namedtuple("Row", ["major", "minor"])
MIN_PYTHON_VERSION = version_tuple(major=3, minor=7)
MIN_TENSORFLOW_VERSION = version_tuple(major=2, minor=4)
MIN_NUMPY_VERSION = version_tuple(major=1, minor=19)


# Filenames and Folder paths
WEIGHTS_DIR = 'weights'
FACE_DETECTION_WEIGHTS = 'shape_predictor_68_face_landmarks.dat'
IMAGE_ENHANCEMENT_SUBDIR = 'Image_Enhancement'
FACE_DETECTION_SUBDIR = 'Face_Detection'
FACE_ENHANCEMENT_SUBDIR = 'Face_Enhancement'


# Default commandline values
PROJECT_DESCRIPTION = "Tensorflow implementation of the project 'Bringing Old Photos to Life'"
DEFAULT_OUTPUT_FOLDER = "outputs"
LOAD_SIZE = 256
LABEL_NC = 18

# Model settings
BATCH_NORM_CLASS = tfa.layers.InstanceNormalization