# -*- coding: utf-8 -*-
"""
* pytorch_enlightning: Pytorch En(hanced)-Lightning Library
*   functions and procedures to address some 'limitations' of the Pytorch 
*   Lightning library. The issues addressed here are:
*   1. You normally land up writing the same code in the training_step,
       validation_step and test_step functions of class derived from LightningModule
       This class provides a `process_batch` function where you can write this
       common code.
    2. Custom progress bars that report Keras-like metrics and display epoch-wise
       progress. Lightning shows only one bar & previous epoch metrics are not displayed
    3. Logging & displaying plots of metrics across epochs
*
* @author: Manish Bhobe for Nämostuté Ltd.
* My experiments with Python, Data Science & Deep Learning
* The code is made available for illustration purposes only.
* Use at your own risk!!
"""
import warnings

warnings.filterwarnings("ignore")

import sys

if sys.version_info < (2,):
    raise Exception("These samples require Python 3+, please upgrade Python")

import logging

version_info = (1, 0, 0, "dev0")

__version__ = ".".join(map(str, version_info))
__installer_version__ = __version__
__title__ = "Pytorch Enlightning Library"
__author__ = "Manish Bhobé"
# Nämostuté -> means "May our Minds Meet"
__organization__ = "Nämostuté Ltd."
__org_domain__ = "namostute.pytorch.in"
__license__ = __doc__
__project_url__ = "https://github.com/mjbhobe/dl_pytorch"

T3_FAV_SEED = 41

import torch
import pytorch_lightning as pl

from .pel_module import EnLitModule
from .metrics_logger import MetricsLogger
from .pel_progbar import EnLitProgressBar
from .dataset_utils import split_dataset
from .utils import load_model, save_model, predict_module, predict_array
from .cl_options import TrainingArgsParser, parse_command_line
