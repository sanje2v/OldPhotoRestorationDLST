import os
import os.path
import sys
import argparse
import tensorflow as tf
import numpy as np

from models import *
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

        # Create image enhancement model for stages and load their weights
        if opts.with_scratch:
            scratch_detector = ScratchDetector()
        image_enhancer = ImageEnhancer(opts)
        face_detector = FaceDetector(os.path.join(settings.WEIGHTS_DIR, settings.FACE_DETECTION_SUBDIR, settings.FACE_DETECTION_WEIGHTS))
        opts.label_nc = 18
        face_enhancer = FaceEnhancer(opts)
        face_blender = FaceBlender()

        # Load model weights
        # CAUTION: Need to eagerly inference once to create all model layer to apply weights to
        if opts.with_scratch:
            scratch_detector(np.empty((1, 256, 256, 1), dtype=np.float32))
        #image_enhancer([np.empty((1, 256, 256, consts.NUM_RGB_CHANNELS), dtype=np.float32),
        #                np.empty((1, 256, 256, 18), dtype=np.float32)])
        #image_enhancer.load_weights(opts.checkpoint[0]).assert_consumed()

        face_enhancer([np.empty((1, 256, 256, consts.NUM_RGB_CHANNELS), dtype=np.float32),
                       np.empty((1, 256, 256, consts.NUM_RGB_CHANNELS), dtype=np.float32)])
        face_enhancer.load_weights(opts.checkpoint[1]).assert_consumed()

        # Preprocess and then run each stage
        for i in range(len(input_images)):
            default_output_image_filename = replaceExtension(os.path.basename(input_image_filenames[i]), 'png')

            ######### Step 1: Optional scratch detection and then Image enhancement
            if opts.with_scratch:
                print(INFO("Running Scratch Detection stage...", prefix='\n'))
                input_image_size = input_images[i].shape[0:2]
                input_image = input_scale_transform(input_images[i], 'scale', settings.LOAD_SIZE, 16)
                input_image = input_grayscale_transform(input_image)
                input_image = input_normalize_transform(input_image)
                scratch_mask = scratch_detector(input_image)
                scratch_mask = tf.image.resize(scratch_mask, size=input_image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                scratch_mask = np.squeeze(scratch_mask.numpy(), axis=0)

                output_image_dir = os.path.join(opts.output_folder, settings.IMAGE_ENHANCEMENT_SUBDIR)
                os.makedirs(output_image_dir, exist_ok=True)
                output_image_filename = os.path.join(output_image_dir, "scratchmask_" + default_output_image_filename)
                tf.keras.preprocessing.image.save_img(output_image_filename, scratch_mask, scale=True)
            else:
                scratch_mask = None

            print(INFO("Running Image Enhancement stage...", prefix='\n'))
            input_image = input_scale_transform(input_images[i], opts.test_mode.lower(), settings.LOAD_SIZE, 4)
            input_image = input_normalize_transform(input_image)
            enhanced_image = image_enhancer([input_image, scratch_mask])
            enhanced_image = np.squeeze(rescale_model_image_output_for_opencv(enhanced_image.numpy()), axis=0)

            # Save stage 1 output to file
            output_image_dir = os.path.join(opts.output_folder, settings.IMAGE_ENHANCEMENT_SUBDIR)
            os.makedirs(output_image_dir, exist_ok=True)
            output_image_filename = os.path.join(output_image_dir, default_output_image_filename)
            tf.keras.preprocessing.image.save_img(output_image_filename, enhanced_image, scale=False)
            print(INFO("Image Enhancement stage output saved to {:s}.".format(output_image_filename)))

            ######### Step 2: Face detection
            print(INFO("Running Face Detection stage...", prefix='\n'))
            faces_with_affines = face_detector(enhanced_image)
            print(INFO("Found {:d} face(s).".format(len(faces_with_affines))))
            enhanced_faces = []

            for face_id, (face_image, _, _) in enumerate(faces_with_affines):
                # Save stage 2 output to file
                output_image_dir = os.path.join(opts.output_folder, settings.FACE_DETECTION_SUBDIR)
                os.makedirs(output_image_dir, exist_ok=True)
                output_image_filename = os.path.join(output_image_dir, "{:d}_{:s}".format(face_id, default_output_image_filename))
                tf.keras.preprocessing.image.save_img(output_image_filename, face_image, scale=False)
                print(INFO("Face detection stage output saved to {:s}.".format(output_image_filename)))

                ########## Step 3: Face enhancement
                print(INFO("Running Face Enhancement stage...", prefix='\n'))
                degraded_image = input_scale_transform((face_image.astype(np.float32) / 255.), opts.test_mode.lower(), settings.LOAD_SIZE)
                degraded_image = tf.expand_dims(tf.convert_to_tensor(input_normalize_transform(degraded_image)), axis=0)
                seg = tf.zeros([1, settings.LOAD_SIZE, settings.LOAD_SIZE, opts.label_nc], dtype=degraded_image.dtype)
                enhanced_face = face_enhancer([seg, degraded_image])
                enhanced_face = np.squeeze(rescale_model_image_output_for_opencv(enhanced_face.numpy()), axis=0)
                enhanced_faces.append(enhanced_face)
                output_image_dir = os.path.join(opts.output_folder, settings.FACE_ENHANCEMENT_SUBDIR)
                os.makedirs(output_image_dir, exist_ok=True)
                output_image_filename = os.path.join(output_image_dir, "{:d}_{:s}".format(face_id, default_output_image_filename))
                tf.keras.preprocessing.image.save_img(output_image_filename, enhanced_face, scale=False)
                print(INFO("Face enhancement stage outputs saved to {:s}.".format(output_image_filename)))

            ########## Step 4: Blending face back and histogram color matching
            if len(faces_with_affines) > 0:
                print(INFO("Running blending stage...", prefix='\n'))
                blended_image = face_blender(enhanced_image, faces_with_affines, enhanced_faces)
                output_image_dir = os.path.join(opts.output_folder, settings.BLENDING_SUBDIR)
                os.makedirs(output_image_dir, exist_ok=True)
                output_image_filename = os.path.join(output_image_dir, default_output_image_filename)
                tf.keras.preprocessing.image.save_img(output_image_filename, blended_image, scale=False)
                print(INFO("Blending stage output saved to {:s}.".format(output_image_filename)))




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
        parser.add_argument('--checkpoint', required=True, nargs=2, metavar=("IMAGE_ENHANCEMENT_WEIGHTS", "FACE_ENHANCEMENT_WEIGHTS"), type=str, help="Checkpoint weights to use")
        parser.add_argument('--with_scratch', action='store_true', help="Also remove scratches in input image")
        args = parser.parse_args()

        main(args)

    except KeyboardInterrupt:
        print(CAUTION("Caught 'Ctrl+c' SIGINT signal. Aborted operation."))