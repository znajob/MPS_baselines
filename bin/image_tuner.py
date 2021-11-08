from qmltn.torchmps.tuner.tuner_cli import TunerCLI
import torch
from torchvision import transforms, datasets
import numpy as np
import elasticdeform.torch as etorch
import random
from sklearn.model_selection import KFold

from qmltn.utils.augmentation import image_transformations
from qmltn.utils.dataset import get_datasets, init_loaders
from qmltn import __version__


def init_loaders_all_folds(hp, **kwargs):
    # Random ColorJitter
    if kwargs["disable_color_jitter"]:
        color_jitter_prob = hp.Fixed("aug_color_jitter_prob", 0)
    else:
        color_jitter_prob = abs(
            hp.Float("aug_color_jitter_prob", 0.0001, 1.0, default=0.1))
        brightness = abs(hp.Float("aug_brightness", 0.01, 1.0, default=0.01))
        contrast = abs(hp.Float("aug_contrast", 0.01, 1.0, default=0.01))
        saturation = abs(hp.Float("aug_saturation", 0.01, 1.0, default=0.01))
        hue = abs(hp.Float("aug_hue", 0.0001, 0.5, default=0.01))

    # Random Sharpness
    if kwargs["disable_sharpness"]:
        sharp_prob = hp.Fixed("aug_sharpness_prob", 0)
    else:
        sharp_prob = abs(
            hp.Float("aug_sharpness_prob", 0.0001, 1.0, default=0.1))
        sharp_min = abs(hp.Float("aug_sharp_min", 0.01, 1.0, default=0.01))
        sharp_max = abs(hp.Float("aug_sharp_max", 1.0, 10.0, default=0.01))

    # Random GaussianBlur
    if kwargs["disable_blur"]:
        blur_prob = hp.Fixed("aug_gblur_prob", 0)
    else:
        blur_prob = abs(hp.Float("aug_gblur_prob", 0.0001, 1.0, default=0.1))
        blur_kernel_size = abs(
            hp.Int("aug_gblur_kernel", 1, kwargs['crop'], default=10))

    # Random flip
    if kwargs["disable_hflip"]:
        hflip = hp.Fixed("aug_horizontal_flip", False)
    elif kwargs["hflip"]:
        hflip = hp.Fixed("aug_horizontal_flip", True)
    else:
        hflip = hp.Choice("aug_horizontal_flip", [True, False], default=True)

    # Random affine transformation
    if kwargs["disable_affine"]:
        affine_prob = hp.Fixed("aug_affine_prob", 0)
    else:
        affine_prob = abs(hp.Float("aug_affine_prob", 0.0001, 1.0, default=0.1))
        txy = abs(hp.Float("aug_translate", 1e-5, 0.2, default=0.1))
        rotate = abs(hp.Float("aug_rotate", 1e-5, 20, default=10))
        scale_min = hp.Float("aug_scale_min", 0.8, 1, default=0.9)
        scale_max = hp.Float("aug_scale_max", 1., 1.2, default=1.1)

    # Random perspective transformation
    if kwargs["disable_perspective"]:
        perspective_prob = hp.Fixed("aug_perspective_prob", 0)
    else:
        perspective_prob = hp.Float("aug_perspective_prob", 0.01, 1, default=0.7)
        perspective_scale = hp.Float(
            "aug_perspective_scale", 0.001, 1, default=0.5)

    # Random elastic transformation
    if kwargs["disable_elastic"]:
        elastic_prob = hp.Fixed("aug_elastic_prob", 0)
    else:
        elastic_prob = hp.Float("aug_elastic_prob", 0.01, 1, default=0.7)
        elastic_scale = hp.Float("aug_elastic_strength", 0.01, 8, default=1.0)

    # Random erasing
    if kwargs["disable_erasing"]:
        erasing_prob = hp.Fixed("aug_erasing_prob", 0)
    else:
        erasing_prob = abs(hp.Float("aug_erasing_prob", 1e-6, 0.7, default=0.1))
        erasing_scale_min = abs(
            hp.Float("aug_erasing_scale_min", 1e-5, 0.02, default=0.01))
        erasing_scale_max = hp.Float("aug_erasing_scale_max",
                                    0.05, 0.33, default=0.1)

    kwargs.update(hp.values)

    return init_loaders(all_folds=True, **kwargs)


class ImageTuner(TunerCLI):
    def __init__(self):
        super(ImageTuner, self).__init__(
            init_loaders_all_folds=init_loaders_all_folds)

    def arg_parser(self):
        parser = super(ImageTuner, self).arg_parser()
        parser.add_argument('--dataset',
                            default="MNIST",
                            type=str,
                            help='The name of the dataset')
        parser.add_argument('--savedir',
                            default="../data/tuner/",
                            type=str,
                            help='A full path to where the files should be stored')
        parser.add_argument('--hflip',
                            action='store_true',
                            help='If enabled horizontal flip will always be applied.')

        return parser


if __name__ == "__main__":
    ImageTuner().search()
