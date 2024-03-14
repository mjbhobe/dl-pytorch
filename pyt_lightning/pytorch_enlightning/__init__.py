# -*- coding: utf-8 -*-
"""
* pytorch_enlightning: Pytorch En(hanced)-Lightning Library
*   functions and procedures to further enhance Pytorch Lightning experience
*
* @author: Manish Bhobe
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
from .utils import load_model, save_model, predict_module, predict_array
from .cl_options import TrainingArgsParser, parse_command_line
