# TensorFlow v2+Keras re-implementation of 'Old Photo Restoration via Deep Latent Space Translation', CVPR 2020 paper
This is a TensorFlow v2+Keras inference ONLY implemention of a [CVPR 2020 paper](https://arxiv.org/abs/2004.09484) that restores old photos sufferring from degradations like faded colors, scratches and color spots by jointly learning from the latent spaces of paired artificially degraded images and real degraded photos.

The whole process consists of the following steps:
* Stage 1 - Image enchancement
    1. Quality restoration OR
    2. Quality restoration with scratch mask
* Stage 2 - Face detection
* Stage 3 - Face enhancement
* Stage 4 - Blend enhanced face image back into enhanced image

Official PyTorch implementation can be found [here](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life).

*[25th March 2021] The official repository has been updated with training code for the model. It is unlikely that this new changes will be re-implemented here.*

# Results
NOTE: Images shown below have been taken from [The United States Library of Congress](https://www.loc.gov/free-to-use/)'s 'free to use and reuse' photo archives.

TODO

# To improve
The output images have a drop shadow-like effect around their edges. Have yet to investigate the source of this problem. Inconsistent padding could causing this.


# Requirements
The following software versions were used for testing the code in this repo.
* Python 3.7
* PyTorch 1.8*
* Tensorflow 2.4.1
* CUDA 11.1
* Pip 21.0.1
* Microsoft Visual Studio 2019 (if using .sln file)
* Other required python libraries are in 'requirements.txt'

NOTE: As of the date of the publication of this repo, DLib is only available for Python 3.6 or lower with Pip. You could either use Python 3.6 or recompile DLib yourselves.

\**Required for weights conversion*

# Getting started
Download all PyTorch and DLib weights from official repo (see page top). Then convert PyTorch weights to Tensorflow checkpoint format weights using:

1. `python convert_weights_for_tf.py --input_weights netG_A <Path to VAE_A weights folder>/latest_net_G.pth netG_B <Path to VAE_B weights folder>/latest_net_G.pth mapping_net <Path to mapping net weights folder>/latest_net_mapping_net.pth --stage 1 --output_weights ./weights/Photo_Enhancement/tf_keras/out.weights`
2. `python convert_weights_for_tf.py --input_weights netG_A <Path to VAE_A weights folder>/latest_net_G.pth netG_B <Path to VAE_B weights folder>/latest_net_G.pth mapping_net <Path to mapping net weights folder>/latest_net_mapping_net.pth --stage 3 --output_weights ./weights/Image_Enhancement/tf_keras/out.weights`

Then inference can be done in a folder of input images using:

`python main.py --input_folder <Folder with images> --checkpoint .\weights\Photo_Enhancement\tf_keras\out.weights --gpu_id 0`

# License
>Any part of this source code can ONLY be reused for research purposes with citation. This repo contains some modified source code from third-party sources who have been credited in files where they were used. Any commercial use of any part of this code requires a formal request to this repo's author.


# References
>Wan, Ziyu and Zhang, Bo and Chen, Dongdong and Zhang, Pan and Chen, Dong and Liao, Jing and Wen, Fang, 2020. Bringing Old Photos Back to Life. In proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 2747-2757)