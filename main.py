import sys
import argparse
import tensorflow as tf
import tensorboard as tb
import numpy as np

from models import Pix2PixHDModel_Mapping
import consts
import settings
import test_options
from utils import *


def main(args):
    # Create 'opts' object with attributes from both 'args' and 'test_options'
    opts = args
    for opt in [x for x in dir(test_options) if not x.startswith('__')]:
        setattr(opts, opt, getattr(test_options, opt))

    with tf.device("/device:GPU:{:d}".format(opts.gpu_id)):
        # Get all image files in 'args.input_folder'
        input_image_filenames = getFilesWithExtension(args.input_folder, consts.IMAGE_FILE_EXTENSIONS, with_path=True)

        # Load all images files
        input_images = []
        for input_image_filename in input_image_filenames:
            input_images.append(tf.image.decode_image(tf.io.read_file(input_image_filename),
                                                      channels=consts.NUM_RGB_CHANNELS,
                                                      dtype=tf.dtypes.float32,
                                                      expand_animations=False))  # NOTE: Image tensor range is 0.0-1.0

        #input_shapes = [tf.keras.Input(shape=(None, None, consts.NUM_RGB_CHANNELS)), tf.keras.Input(shape=(None, None, consts.NUM_RGB_CHANNELS))]
        #output_shapes = tf.keras.Input(shape=(None, None, consts.NUM_RGB_CHANNELS))
        model = Pix2PixHDModel_Mapping(opts)#, inputs=input_shapes)
        model.compile()

        # Preprocess and then batch predict
        output_images = []
        for i in range(len(input_images)):
            if opts.NL_use_mask:
                pass
            else:
                input_image = input_scale_transform(input_images[i], opts.test_mode.lower())
                mask = tf.zeros_like(input_image)

            input_image = input_normalize_transform(input_image)
            output_images.append(model([input_image, mask]))

        # Output images files
        for i in range(len(output_images)):
            output_image_filename = os.path.join(opts.input_folder, os.path.basename(input_images[i]))
            tf.keras.preprocessing.image.save_img(output_image_filename, tf.make_ndarray(output_image[i]), scale=True)


if __name__ == '__main__':
    assert check_version(sys.version_info, *settings.MIN_PYTHON_VERSION), \
        FATAL("This program needs at least Python {0:d}.{1:d} interpreter.".format(*settings.MIN_PYTHON_VERSION))
    assert check_version(tf.__version__, *settings.MIN_TENSORFLOW_VERSION), \
        FATAL("This program needs at least Tensorflow {0:d}.{1:d}.".format(*settings.MIN_TENSORFLOW_VERSION))
    assert check_version(np.__version__, *settings.MIN_NUMPY_VERSION), \
        FATAL("This program needs at least NumPy {0:d}.{1:d}.".format(*settings.MIN_NUMPY_VERSION))

    assert tf.executing_eagerly(), "Eager execution needs to be enabled for Tensorflow. Expected to be enabled by default."

    try:
        parser = argparse.ArgumentParser(description=settings.PROJECT_DESCRIPTION)
        parser.add_argument('--input_folder', required=True, type=lambda x: os.path.abspath(x), help="Folder with image files to process")
        parser.add_argument('--output_folder', type=lambda x: os.path.abspath(x), default=settings.DEFAULT_OUTPUT_FOLDER, help="Folder where to output processed images")
        parser.add_argument('--gpu_id', type=int, default=0, help="GPU device id")
        parser.add_argument('--checkpoint', type=str, default='Setting_9_epoch_100', help="Checkpoint weights to use")
        parser.add_argument('--with_scratch', action='store_true', help="Also remove scratches in input image")
        args = parser.parse_args()

        main(args)

    except KeyboardInterrupt:
        print(CAUTION("Caught 'Ctrl+c' SIGINT signal. Aborted operation."))