import os
import argparse
import torch as t
import tensorflow as tf
from tensorflow_addons.layers import SpectralNormalization
import numpy as np

from models import *
import consts
import settings
from utils import *
import test_options


def main(args):
    # Create 'opts' object with attributes from both 'args' and 'test_options'
    opts = args
    for opt in [x for x in dir(test_options) if not x.startswith('__')]:
        setattr(opts, opt, getattr(test_options, opt))

    # Load the weights file into dictionary
    for key, value in opts.input_weights.items():
        opts.input_weights[key] = t.load(value, map_location=t.device('cpu'))

    with tf.device('/CPU'):
        if opts.stage == 1:
            # Create the model instance and run it eagerly so that all its layers are constructed
            model = ImageEnhancer(opts)
            model([np.empty((1, 256, 256, consts.NUM_RGB_CHANNELS), dtype=np.float32),
                   np.empty((1, 256, 256, 1), dtype=np.float32)])

            # CAUTION: This code assumes that PyTorch saves weights in the order of layers
            for layer in model.layers:
                input_weights = opts.input_weights[layer.name]
                if 'state_dict' in input_weights:
                    input_weights = input_weights['state_dict']

                inner_layer_list = list(filter(lambda x: x.startswith(layer.inner_layers.name), input_weights.keys()))
                inner_layer_index = 0
                conv2d_variables = [x for x in layer.variables if 'conv2d' in x.name]

                for i, variable in enumerate(conv2d_variables):
                    if 'kernel' in variable.name:
                        weights = np.transpose(input_weights[inner_layer_list[inner_layer_index]].numpy(), (2, 3, 1, 0))
                        assert weights.shape == variable.shape, "Weights for '{:s}' being assigned is of a different shape than of variable!".format(variable.name)
                        variable.assign(weights)
                        inner_layer_index += 1

                    elif 'bias' in variable.name:
                        bias = input_weights[inner_layer_list[inner_layer_index]].numpy()
                        assert bias.shape == variable.shape, "Bias for '{:s}' being assigned is of a different shape than of variable!".format(variable.name)
                        variable.assign(bias)
                        inner_layer_index += 1

                if len(inner_layer_list) == inner_layer_index:
                    print(INFO("All {:d} weights values were used for '{:s}'.".format(len(inner_layer_list), layer.name), prefix='\n'))
                else:
                    print(CAUTION("Not all {:d} weights values were used for '{:s}'.".format(len(inner_layer_list), layer.name), prefix='\n'))
        elif opts.stage == 3:
            # Create the model instance and run it eagerly so that all its layers are constructed
            model = FaceEnhancer(opts)
            model([np.empty((1, 256, 256, consts.NUM_RGB_CHANNELS), dtype=np.float32),
                   np.empty((1, 256, 256, consts.NUM_RGB_CHANNELS), dtype=np.float32)])

            input_weights = opts.input_weights[model.name]
            unused_keys = list(input_weights.keys())

            for variable in model.variables:
                name = to_pytorch_like_name(variable.name, prefix_len_to_remove=len(model.name) + 1)

                if name in input_weights:
                    weights = input_weights[name].numpy()
                    if len(weights.shape) == 4:
                        weights = np.transpose(weights, (2, 3, 1, 0))
                    elif len(variable.shape) == 2:
                        weights = weights.reshape((1, weights.size))

                    assert weights.shape == variable.shape, "Weights for '{:s}' being assigned is of a different shape than of variable!".format(variable.name)
                    variable.assign(weights)

                    unused_keys.remove(name)
                else:
                    raise ValueError("Couldn't find weight for variable '{:s}'!".format(variable.name))

            # CAUTION: It is necessary to call 'normalize()' on all 'SpectralNormalization' layers as we
            # only loaded original weights and 'u' values to their variables
            for module in model.submodules:
                if isinstance(module, SpectralNormalization):
                    module.normalize_weights()

            if len(unused_keys) > 0:
                print(CAUTION("The following weights key(s) were unused:\n{:}.".format(unused_keys)))

        # Create all the intermediate directories and then save weights as TensorFlow checkpoint
        os.makedirs(os.path.dirname(opts.output_weights), exist_ok=True)
        model.save_weights(opts.output_weights, save_format='tf')


def to_pytorch_like_name(name, prefix_len_to_remove=0):
    name = name[prefix_len_to_remove:].replace('/', '.')
    replacewith = 'weight'
    if 'mlp_shared' in name:
        replacewith = '0.' + replacewith
    elif any(name.startswith(x) for x in ['head_0.conv_',
                                          'G_middle_0.conv_',
                                          'G_middle_1.conv_',
                                          'up_0.conv_',
                                          'up_1.conv_',
                                          'up_2.conv_',
                                          'up_3.conv_']):
        replacewith = 'weight_orig'
    name = name.replace('kernel:0', replacewith)
    replacewith = 'bias'
    if 'mlp_shared' in name:
        replacewith = '0.' + replacewith
    name = name.replace('bias:0', replacewith)
    name = name.replace('sn_u:0', 'weight_u')
    name = name.replace('moving_mean:0', 'running_mean')
    name = name.replace('moving_variance:0', 'running_var')
    return name


if __name__ == '__main__':
    try:
        print("This tool converts PyTorch weights to Tensorflow+Keras weights for the project 'https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life'.\n")

        parser = argparse.ArgumentParser(description=settings.PROJECT_DESCRIPTION)
        parser.add_argument('--input_weights', required=True, nargs='+', action=ValidateLayerNamesAndWeightsFile, help="Layer names followed by weights files to load and convert")
        parser.add_argument('--output_weights', required=True, type=lambda x: os.path.abspath(x), help="Tensorflow + Keras weights file")
        parser.add_argument('--stage', type=int, required=True, choices=[1, 3], help="Stage 1: Image enhancement, Stage 3: Face enhancement")
        parser.add_argument('--with_scratch', action='store_true', help="Also remove scratches in input image")
        args = parser.parse_args()

        main(args)

    except KeyboardInterrupt:
        print(CAUTION("Caught 'Ctrl+c' SIGINT signal. Aborted operation."))