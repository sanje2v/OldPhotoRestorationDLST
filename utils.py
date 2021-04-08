import os
import os.path
import argparse
import termcolor
import tensorflow as tf
import numpy as np


def INFO(text, prefix=''):
    return termcolor.colored(prefix + "INFO: {:}".format(text), 'green')

def CAUTION(text, prefix=''):
    return termcolor.colored(prefix + "CAUTION: {:}".format(text), 'yellow')

def FATAL(text, prefix=''):
    return termcolor.colored(prefix + "FATAL: {:}".format(text), 'red', attrs=['reverse', 'blink'])


def check_version(version, major, minor):
    if type(version) == str:
        version = tuple(int(x) for x in version.split('.'))
    return version[0] >= major and version[1] >= minor

def getFilesWithExtension(dir, extension_or_tuple, with_path=False):
    if not type(extension_or_tuple) is tuple:
        extension_or_tuple = (extension_or_tuple,)
    extension_or_tuple = tuple(x.casefold() for x in extension_or_tuple)
    return [(os.path.join(dir, f) if with_path else f) for f in os.listdir(dir) if f.casefold().endswith(extension_or_tuple)]

class ValidateLayerNamesAndWeightsFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) % 2 != 0:
            raise ValueError("'--input_weights' should have layer name followed by weight filenames!")

        input_weights_dict = {}
        for i in range(0, len(values), 2):
            layer_name, weights_filename = values[i], values[i+1]

            if not os.path.isfile(weights_filename):
                raise ValueError("{:s} weights file specified in '--input_weights' doesn't exists!".format(weights_filename))
            input_weights_dict[layer_name] = os.path.abspath(weights_filename)
        setattr(namespace, self.dest, input_weights_dict)


def input_scale_transform(input_image, test_mode, load_size):
    h, w = input_image.shape[0:2]

    if test_mode == 'scale':
        if w < h:
            w = load_size
            h = (h * load_size) // input_image.shape[1]
        else:
            h = load_size
            w = (w * load_size) // input_image.shape[0]

    if test_mode in ['scale', 'full']:
        if any([x % 4 != 0 for x in (h, w)]):
            h = int(round(h / 4) * 4)
            w = int(round(w / 4) * 4)
            input_image = tf.image.resize(input_image, (h, w), tf.image.ResizeMethod.BILINEAR)
    else:
        input_image = tf.image.resize_with_crop_or_pad(input_image, load_size, load_size)
    return input_image

def input_normalize_transform(input_image):
    return (input_image - 0.5) / 0.5

def rescale_model_image_output_for_opencv(img):
    return np.clip(np.floor((img + 1.0) / 2.0 * 255.0), a_min=0., a_max=255.).astype(np.uint8)


def iterative_call(funcs_list, initial_input, *args, **kwargs):
    x = initial_input
    for func in funcs_list:
        x = func(x, *args, **kwargs)
    return x