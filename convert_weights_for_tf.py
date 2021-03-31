import argparse
import torch as t
import tensorflow as tf
import numpy as np

from models import Pix2PixHDModel_Mapping
import consts
import settings
from utils import *
import test_options


def main(args):
    # Create 'opts' object with attributes from both 'args' and 'test_options'
    opts = args
    for opt in [x for x in dir(test_options) if not x.startswith('__')]:
        setattr(opts, opt, getattr(test_options, opt))

    with tf.device('/CPU'):
        # Load the weights file into dictionary
        for key, value in opts.input_weights.items():
            opts.input_weights[key] = t.load(value, map_location=t.device('cpu'))

        if opts.stage == 1:
            # Create the model instance and run it eagerly so that all its layers are constructed
            model = Pix2PixHDModel_Mapping(opts)
            model([np.empty((1, 256, 256, consts.NUM_RGB_CHANNELS), dtype=np.float32),
                   np.empty((1, 256, 256, consts.NUM_RGB_CHANNELS), dtype=np.float32)])

            for layer in model.layers:
                input_weights = opts.input_weights[layer.name]
                print(layer.name)

                inner_layer_list = list(filter(lambda x: x.startswith(layer.inner_layer.name), input_weights.keys()))
                inner_layer_index = 0
                conv2d_variables = [x for x in layer.inner_layer.weights if x.name.startswith('conv2d')]

                for variable in conv2d_variables:
                    if 'kernel' in variable.name:
                        weights = np.transpose(input_weights[inner_layer_list[inner_layer_index]].numpy(), (2, 3, 1, 0))
                        variable.assign(weights)
                        inner_layer_index += 1
                    elif 'bias' in variable.name:
                        bias = input_weights[inner_layer_list[inner_layer_index]].numpy()
                        variable.assign(bias)
                        inner_layer_index += 1

                print(INFO("All weights values {:s} used.".format("were" if len(conv2d_variables) == inner_layer_index else "weren't"), prefix='\n'))
        else:
            pass

        model.save_weights(opts.output_weights, save_format='tf')


if __name__ == '__main__':
    try:
        print("This tool converts PyTorch weights to Tensorflow+Keras weights for the project 'https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life'.\n")

        parser = argparse.ArgumentParser(description=settings.PROJECT_DESCRIPTION)
        parser.add_argument('--input_weights', required=True, nargs='+', action=ValidateLayerNamesAndWeightsFile, help="Layer names followed by weights files to load and convert")
        parser.add_argument('--output_weights', required=True, type=lambda x: os.path.abspath(x), help="Tensorflow + Keras weights file")
        parser.add_argument('--stage', type=int, required=True, choices=[1, 3], help="Stage 1: Image enhancement, Stage 3: Face enhancement")
        args = parser.parse_args()

        main(args)

    except KeyboardInterrupt:
        print(CAUTION("Caught 'Ctrl+c' SIGINT signal. Aborted operation."))