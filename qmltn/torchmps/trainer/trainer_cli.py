#!/usr/bin/env python3
import argparse

from qmltn.torchmps.trainer.trainer import Trainer
from qmltn.utils.dataset import init_loaders


class TrainerCLI(Trainer):
    def __init__(self):
        parse_args = self.arg_parser().parse_args()
        super(TrainerCLI, self).__init__(**vars(parse_args))

    def arg_parser(self):
        parser = argparse.ArgumentParser()
        # parser.add_argument('--dataset',
        #                     default="TEST",
        #                     type=str,
        #                     help='The name of the dataset.')
        # parser.add_argument('--savedir',
        #                     default="data/test",
        #                     type=str,
        #                     help='A full path to where the files should be stored.')
        parser.add_argument('--embedding',
                            default="linear",
                            type=str,
                            help='Embedding type of the initial vector. angle(default), linear, auto')
        parser.add_argument('--nfolds',
                            default=5,
                            type=int,
                            help='Number of folds for crossvalidation: 5 (default).')
        parser.add_argument('--fold',
                            default=0,
                            type=int,
                            help='Fold used for training: 0 (default).')
        parser.add_argument('--wandb',
                            default=0,
                            type=int,
                            help='If enabled the results will be send to wandb. 0 (disabled-default), 1 (enabled).')
        parser.add_argument('--wandb_offline',
                            default=0,
                            type=int,
                            help='Used only if wandb == 1. If enabled the logging is done offline. 0 (disabled-default), 1 (enabled).')
        parser.add_argument('--entity',
                            default="",
                            type=str,
                            help='WANDB entity where to send the runs data. Default is empty meaning the runs will be sent to default entity. Only used if wandb is enabled.')
        parser.add_argument('--profile',
                            default=0,
                            type=int,
                            help='If enabled the model profiling will be performed before the start of the training. 0 (disabled-default), 1 (enabled).')
        parser.add_argument('--monitor_ent',
                            default=0,
                            type=int,
                            help='If enabled entropy is calculated at each epoch end. 0 (disabled-default), 1 (enabled).')
        parser.add_argument('--max_training_hours',
                            default=44,
                            type=float,
                            help='Maximal training duration in hours. 44 (default), -1 => no simulation time limit.')
        parser.add_argument('--datadir',
                            default="../dataset/",
                            type=str,
                            help='A full path to where the files should be stored.')
        parser.add_argument('--prefix',
                            default="",
                            type=str,
                            help='Prefix for saved models. Empty string by default.')
        parser.add_argument('--D',
                            default=30,
                            type=int,
                            help='Bond dimension for the mps.')
        parser.add_argument('--bs',
                            default=500,
                            type=int,
                            help='Batch size.')
        parser.add_argument('--ntrain',
                            default=60000,
                            type=int,
                            help='Number of training examples. Defaults to MNIST size.')
        parser.add_argument('--ntest',
                            default=10000,
                            type=int,
                            help='Number of test examples. Defaults to MNIST size.')
        parser.add_argument('--train_ratio',
                            default=1.0,
                            type=float,
                            help='Ratio of the training set examples to be used in training.')
        parser.add_argument('--lr',
                            default=0.0002,
                            type=float,
                            help='Learning rate.')
        parser.add_argument('--l2',
                            default=0.0,
                            type=float,
                            help='L2 regularization of the MPS parameters. Default is 0.')
        parser.add_argument('--seed',
                            type=int,
                            default=42,
                            help='Seed for the optimization and initialization.')
        parser.add_argument('--nepoch',
                            type=int,
                            default=300,
                            help='Number of epoch.')
        parser.add_argument('--step',
                            default=50,
                            type=int,
                            help='Step size for the learning rate decay in the number of epochs.')
        parser.add_argument('--gamma',
                            default=0.5,
                            type=float,
                            help='Learning rate decay: lr = lr * gamma.')
        parser.add_argument('--aug_phi',
                            default=0.0,
                            type=float,
                            help='Maximum size of the random shift of the input vector elements.')
        parser.add_argument('--crop',
                            default=28,
                            type=int,
                            help='Image crop size.')
        parser.add_argument('--permute',
                            default=0,
                            type=int,
                            help='Enable permutation of the input.')
        parser.add_argument('--spiral',
                            default=0,
                            type=int,
                            help='Spiral order of mps sites in the image.')
        parser.add_argument('--periodic',
                            default=1,
                            type=int,
                            help='Enable periodic boundary conditions.')
        parser.add_argument('--ti',
                            default=0,
                            type=int,
                            help='Translationary invariant MPS.')
        parser.add_argument('--cuda',
                            default=0,
                            type=int,
                            help='Use cuda GPU.')
        parser.add_argument('--nclass',
                            default=10,
                            type=int,
                            help='Number of classes.')
        parser.add_argument('--verbose',
                            default=0,
                            type=int,
                            help='Determines the logger output.')
        parser.add_argument('--savemodel',
                            default=0,
                            type=int,
                            help='If enabled the model will be saved.')
        parser.add_argument('--monitoring',
                            default=0,
                            type=int,
                            help='If enabled the model will be evaluated on the test set after each epoch.')
        parser.add_argument('--optimizer',
                            default="adam",
                            type=str,
                            help='Optimizer used for training. Possible options: "adam", "adadelta", "adamw", "adamax", "lbfgs", "rmsprop", "rprop", "sgd"')
        parser.add_argument('--stop_patience',
                            default=50,
                            type=int,
                            help='Early stopping patience in epochs. 50 (default)')
        parser.add_argument('--reset_early_stopping',
                            default=0,
                            type=int,
                            help='If enabled resets the early stopping difference to 0.')
        parser.add_argument('--continue_training',
                            default=0,
                            type=int,
                            help='If enabled a new job will be submitted to the queue after the job is killed before completion: 0 (default), 1')
        parser.add_argument('--checkpoint',
                            default="",
                            type=str,
                            help='A path to the model for from which we start training. If not specified we use a standard random initialization. If a model already exists, the checkpoint is ignored.')
        return parser


class ImageTrainerCLI(TrainerCLI):
    def __init__(self):
        super(ImageTrainerCLI, self).__init__()

    def init_loaders(self, *args, **kwargs):
        self.loaders, self.num_batches = init_loaders(*args, **kwargs)

    def get_feature_dim(self, *args, **kwargs):
        # The standard embedding has a local dimension nchan+1.
        feature_dim = 2
        if kwargs["dataset"] in ["CIFAR10", "CIFAR100"] and not kwargs['use_grayscale']:
            # For color datasets we use a different embedding with local hilbert space of dimension 4
            feature_dim = 4

        # We can add additional higher order features for linear embeddings
        nchan = feature_dim - 1
        if kwargs["embedding"] == "linear":
            emb_ord = kwargs["embedding_order"]
            if emb_ord == 2:
                feature_dim += nchan**2
            elif emb_ord == 3:
                feature_dim += nchan**2 + nchan**3

        return feature_dim

    def get_input_dim(self, *args, **kwargs):
        return kwargs['crop']**2

    def arg_parser(self):
        parser = super(ImageTrainerCLI, self).arg_parser()
        parser.add_argument('--dataset',
                            default="MNIST",
                            type=str,
                            help='The name of the dataset')
        parser.add_argument('--savedir',
                            default="../data",
                            type=str,
                            help='A full path to where the files should be stored')
        parser.add_argument('--embedding_order',
                            default=1,
                            type=int,
                            help='Order of features in the embedding. Used only for linear embedding. Options: 1 (default), 2 , 3')
        parser.add_argument('--use_grayscale',
                            default=0,
                            type=int,
                            help='Used in colored images. If enabled a colored image is transformed into the grayscale image after the augmentation process. Options: 0 (disabled-default), 1 (enabled)')
        # Data augmentation options
        parser.add_argument('--aug_random_crop',
                            default=0,
                            type=int,
                            help='Random cropping the image. Can only be used in the TI_MPS model.')
        parser.add_argument('--aug_horizontal_flip',
                            default=0,
                            type=int,
                            help='Random horizontal flipping of the image.')
        parser.add_argument('--aug_color_jitter_prob',
                            default=0,
                            type=float,
                            help='Probability of the random jitter transformation')
        parser.add_argument('--aug_brightness',
                            default=0.9,
                            type=float,
                            help='Scale of the random brightness transform.')
        parser.add_argument('--aug_contrast',
                            default=0.85,
                            type=float,
                            help='Scale of the random contrast transform')
        parser.add_argument('--aug_saturation',
                            default=0.85,
                            type=float,
                            help='Scale of the random saturation transform')
        parser.add_argument('--aug_hue',
                            default=0.45,
                            type=float,
                            help='Scale of the random hue transform')
        parser.add_argument('--aug_sharpness_prob',
                            default=0,
                            type=float,
                            help='Probability for the sharpe transformation')
        parser.add_argument('--aug_sharp_min',
                            default=0.1,
                            type=float,
                            help='Minimum for the random sharpness')
        parser.add_argument('--aug_sharp_max',
                            default=10.0,
                            type=float,
                            help='Maximum for the random sharpness')
        parser.add_argument('--aug_gblur_prob',
                            default=0,
                            type=float,
                            help='Probability of the random blur.')
        parser.add_argument('--aug_gblur_kernel',
                            default=7,
                            type=int,
                            help='Kernel size of the random Gaussian blur')
        parser.add_argument('--aug_affine_prob',
                            default=0,
                            type=float,
                            help='Probability of affine transformation of an image. 0.85 (default)')
        parser.add_argument('--aug_translate',
                            default=0.02,
                            type=float,
                            help='Relative random translation of the image.')
        parser.add_argument('--aug_rotate',
                            default=3.1,
                            type=float,
                            help='Maximum angle for a random rotation of the image.')
        parser.add_argument('--aug_scale_min',
                            default=0.8,
                            type=float,
                            help='Minimum relative size for a random scaling of the image.')
        parser.add_argument('--aug_scale_max',
                            default=1.06,
                            type=float,
                            help='Maximum relative size for a random scaling of the image.')
        parser.add_argument('--aug_elastic_prob',
                            default=0,
                            type=float,
                            help='Probability of elastic deformation of an image. 0.85 (default)')
        parser.add_argument('--aug_elastic_strength',
                            default=1.2,
                            type=float,
                            help='Strength of the elastic deformation in pixels. 0.8 (default)')
        parser.add_argument('--aug_erasing_prob',
                            default=0,
                            type=float,
                            help='Probability of random erasing of an image patch.')
        parser.add_argument('--aug_erasing_scale_min',
                            default=0.001,
                            type=float,
                            help='Minimum relative size of the randomly erased patch.')
        parser.add_argument('--aug_erasing_scale_max',
                            default=0.325,
                            type=float,
                            help='Maximum relative size of the randomly erased patch.')
        parser.add_argument('--aug_perspective_prob',
                            default=0,
                            type=float,
                            help='Probability of the perspective transformation.')
        parser.add_argument('--aug_perspective_scale',
                            default=0.2,
                            type=float,
                            help='Maximum distortion scale of the random perspective transformation.')

        return parser
