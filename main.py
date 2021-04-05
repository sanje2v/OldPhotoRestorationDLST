import os
import os.path
import sys
import argparse
import tensorflow as tf
import tensorboard as tb
import numpy as np

from models import Pix2PixHDModel_Mapping, FaceDetector
import consts
import settings
import test_options
from utils import *


def main(args):
    # Create 'opts' object with attributes from both 'args' and 'test_options'
    opts = args
    for opt in [x for x in dir(test_options) if not x.startswith('__')]:
        setattr(opts, opt, getattr(test_options, opt))

    if not opts.gpu_id < 0:
        # Set GPU memory usage growth policy to prevent CUBLAS error
        try:
            tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[opts.gpu_id], True)
        except RuntimeError:
            pass

    with tf.device("/CPU" if opts.gpu_id < 0 else "/device:GPU:{:d}".format(opts.gpu_id)):
        # Get all image files in 'args.input_folder'
        input_image_filenames = getFilesWithExtension(args.input_folder, consts.IMAGE_FILE_EXTENSIONS, with_path=True)

        # Load all images files to tensor
        input_images = []
        for input_image_filename in input_image_filenames:
            input_images.append(tf.image.decode_image(tf.io.read_file(input_image_filename),
                                                      channels=consts.NUM_RGB_CHANNELS,
                                                      dtype=tf.dtypes.float32,
                                                      expand_animations=False))  # NOTE: Image tensor range is 0.0-1.0

        # Create image enhancement model for stage 1 and load weights
        model = Pix2PixHDModel_Mapping(opts)
        model([np.empty((1, 256, 256, consts.NUM_RGB_CHANNELS), dtype=np.float32),
               np.empty((1, 256, 256, consts.NUM_RGB_CHANNELS), dtype=np.float32)])    # Need to eagerly create layer to apply weights
        model.load_weights(opts.checkpoint).assert_consumed()

        # Preprocess and then run each stage
        for i in range(len(input_images)):
            if opts.NL_use_mask:
                pass
            else:
                input_image = input_scale_transform(input_images[i], opts.test_mode.lower(), settings.LOAD_SIZE)
                mask = None
            input_image = input_normalize_transform(input_image)

            ######### Step 1: Image enhancement
            print(INFO("Running Image Enhancement stage..."))

            output_image = model([input_image, mask])
            output_image = np.squeeze(rescale_model_image_output_for_opencv(output_image.numpy()), axis=0)

            # Save stage 1 output to file
            output_image_dir = os.path.join(opts.output_folder, settings.IMAGE_ENHANCEMENT_SUBDIR)
            os.makedirs(output_image_dir, exist_ok=True)
            output_image_filename = os.path.join(output_image_dir, os.path.basename(input_image_filenames[i]))
            tf.keras.preprocessing.image.save_img(output_image_filename, output_image, scale=False)
            print(INFO("Image Enhancement stage output saved to {:s}.".format(output_image_filename)))

            ########## Step 2: Face detection
            face_detector = FaceDetector(os.path.join(settings.WEIGHTS_DIR, settings.FACE_DETECTION_SUBDIR, settings.FACE_DETECTION_WEIGHTS))
            faces = face_detector(output_image)

            for i, face in enumerate(faces):
                # Save stage 2 output to file
                output_image_dir = os.path.join(opts.output_folder, settings.FACE_DETECTION_SUBDIR)
                os.makedirs(output_image_dir, exist_ok=True)
                output_image_filename = os.path.join(output_image_dir, "{:d}_{:s}".format(i, os.path.basename(input_image_filenames[i])))
                tf.keras.preprocessing.image.save_img(output_image_filename, face, scale=False)
                print(INFO("Face detection stage output saved to {:s}.".format(output_image_filename)))

                ########## Step 3: Face enhancement
                


if __name__ == '__main__':
    assert check_version(sys.version_info, *settings.MIN_PYTHON_VERSION), \
        FATAL("This program needs at least Python {0:d}.{1:d} interpreter.".format(*settings.MIN_PYTHON_VERSION))
    assert check_version(tf.__version__, *settings.MIN_TENSORFLOW_VERSION), \
        FATAL("This program needs at least Tensorflow {0:d}.{1:d}.".format(*settings.MIN_TENSORFLOW_VERSION))
    assert check_version(np.__version__, *settings.MIN_NUMPY_VERSION), \
        FATAL("This program needs at least NumPy {0:d}.{1:d}.".format(*settings.MIN_NUMPY_VERSION))

    assert tf.executing_eagerly(), "Eager execution needs to be enabled for Tensorflow. Expected it was enabled by default."

    try:
        parser = argparse.ArgumentParser(description=settings.PROJECT_DESCRIPTION)
        parser.add_argument('--input_folder', required=True, type=lambda x: os.path.abspath(x), help="Folder with image files to process")
        parser.add_argument('--output_folder', type=lambda x: os.path.abspath(x), default=settings.DEFAULT_OUTPUT_FOLDER, help="Folder where to output processed images")
        parser.add_argument('--gpu_id', required=False, type=int, default=-1, help="GPU device id OR integer less than 0 for CPU device")
        parser.add_argument('--checkpoint', required=True, type=str, help="Checkpoint weights to use")
        parser.add_argument('--with_scratch', action='store_true', help="Also remove scratches in input image")
        args = parser.parse_args()

        main(args)

    except KeyboardInterrupt:
        print(CAUTION("Caught 'Ctrl+c' SIGINT signal. Aborted operation."))