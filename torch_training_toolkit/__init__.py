# -*- coding: utf-8 -*-
"""
* Torch Training Toolkit - utility functions & classes to ease the process of training,
*   evaluating & testing Pytorch models.
*
* @author: Manish Bhobe
* Inspired by Keras and Pytorch Lightening
* My experiments with Python, Data Science & Deep Learning
* The code is made available for illustration purposes only.
* Use at your own risk!!
"""
import warnings

warnings.filterwarnings('ignore')

import sys

if sys.version_info < (2,):
    raise Exception(
        "torch_training_toolkit does not support Python 1. Please use a Python 3+ interpreter!"
    )
import logging

version_info = (1, 0, 0, "dev0")

__version__ = '.'.join(map(str, version_info))
__installer_version__ = __version__
__title__ = "Torch Training Toolkit (t3)"
__author__ = "Manish Bhobé"
# Nämostuté -> means "May our Minds Meet"
__organization__ = "Nämostuté Ltd."
__org_domain__ = "namostute.pytorch.in"
__license__ = __doc__
__project_url__ = "https://github.com/mjbhobe/dl_pytorch"

T3_FAV_SEED = 41

import torch

# bring in our functions & classes
from utils import get_logger, seed_all, plot_confusion_matrix
from layers import Linear, Conv2d
from dataset_utils import split_dataset
from training import Trainer, load_model, save_model

_logger = get_logger(__name__)