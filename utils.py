import os
import os.path
import glob
import termcolor
import tensorflow as tf


def INFO(text):
    return termcolor.colored("INFO: {:}".format(text), 'green')

def CAUTION(text):
    return termcolor.colored("CAUTION: {:}".format(text), 'yellow')

def FATAL(text):
    return termcolor.colored("FATAL: {:}".format(text), 'red', attrs=['reverse', 'blink'])


def check_version(version, major, minor):
    if type(version) == str:
        version = tuple(int(x) for x in version.split('.'))

    return version[0] >= major and version[1] >= minor

def getFilesWithExtension(dir, extension_or_tuple, with_path=False):
    if not type(extension_or_tuple) is tuple:
        extension_or_tuple = (extension_or_tuple,)
    extension_or_tuple = tuple(x.casefold() for x in extension_or_tuple)
    return [(os.path.join(dir, f) if with_path else f) for f in os.listdir(dir) if f.casefold().endswith(extension_or_tuple)]


def input_scale_transform(input_image, test_mode):
    h, w = input_image.shape[0:2]

    if test_mode == 'scale':
        if w < h:
            w = 256
            h = (h * 256) // input_image.shape[1]
        else:
            h = 256
            w = (w * 256) // input_image.shape[0]

    if test_mode in ['scale', 'full']:
        h = int(round(h / 4) * 4)
        w = int(round(w / 4) * 4)
        input_image = tf.image.resize(input_images[i], (h, w), tf.image.ResizeMethod.BILINEAR)
    else:
        input_image = tf.image.resize_with_crop_or_pad(input_image, 256, 256)

    return input_image

def input_normalize_transform(input_image):
    return (input_image - 0.5) / 0.5