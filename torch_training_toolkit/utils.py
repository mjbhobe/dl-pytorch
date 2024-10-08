# -*- coding: utf-8 -*-
""" utils.py - utility functions """
import warnings

warnings.filterwarnings("ignore")

import sys

if sys.version_info < (2,):
    raise Exception(
        "torch_training_toolkit does not support Python 1. Please use a Python 3+ interpreter!"
    )

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pathlib
from datetime import datetime

# Pytorch imports
import torch


def seed_all(seed=None, logger: logging.Logger = None):
    """seed all random number generators to ensure that you get consistent results
    across multiple runs ON SAME MACHINE - you may get different results
    on a different machine (architecture) & that is to be expected
    @see: https://pytorch.org/docs/stable/notes/randomness.html
    @see: https://discuss.pytorch.org/t/reproducibility-over-different-machines/63047

    @params:
         - seed (optional): seed value that you choose to see everything. Can be None
           (default value). If None, the code chooses a random uint between np.uint32.min
           & np.unit32.max
     @returns:
         - if parameter seed=None, then function returns the randomly chosen seed, else it
           returns value of the parameter passed to the function
    """
    if seed is None:
        # pick a random uint32 seed
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    if logger is not None:
        log_str = f"Initialized random seed {' and torch.cuda' if torch.cuda.is_available() else ''} with {seed}."
        logger.debug(log_str)

    return seed


def get_logger(file_path: pathlib.Path, level: int = logging.WARNING) -> logging.Logger:
    assert (
        file_path.is_file()
    ), f"FATAL ERROR: get_logger() -> file_path parameter must be a valid path to existing file!"
    name = file_path.stem
    # replace extension of file_path with .log
    # log_path = file_path.with_suffix(".log")
    log_dir = file_path.parent / "logs"
    # log_dir does not exist, create it
    log_dir.mkdir(exist_ok=True)
    now = datetime.now()
    now_str = datetime.strftime(now, "%Y%m%d-%H%M%S")
    log_path = f"{log_dir}/{name}_{now_str}.log"

    logger = logging.getLogger(name)
    # create handlers
    stream_handler = logging.StreamHandler()
    # log_path = f"{os.getcwd()}/{name}.log"
    file_handler = logging.FileHandler(log_path)

    stream_handler.setLevel(level)
    file_handler.setLevel(logging.DEBUG)

    stream_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(stream_formatter)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def plot_confusion_matrix(
    cm,
    class_names=None,
    title="Confusion Matrix",
    cmap=plt.cm.Purples,
    fig_size=(8, 6),
):
    """graphical plot of the confusion matrix
    @params:
        cm - the confusion matrix (value returned by the sklearn.metrics.confusion_matrix(...) call)
        class_names (list) - names (text) of classes you want to use (list of strings)
        title (string, default='Confusion Matrix') - title of the plot
        cmap (matplotlib supported palette, default=plt.cm.Blues) - color palette you want to use
    """

    class_names = ["0", "1"] if class_names is None else class_names
    df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=fig_size)
    with sns.axes_style("darkgrid"):
        # sns.set_context("notebook")  # , font_scale = 1.1)
        sns.set_style(
            {
                "font.sans-serif": [
                    "Segoe UI",
                    "Calibri",
                    "SF Pro Display",
                    "Arial",
                    "DejaVu Sans",
                    "Sans",
                ]
            }
        )
        hmap = sns.heatmap(df, annot=True, fmt="d", cmap=cmap)
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha="right")
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha="right")

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.title(title)
    plt.show()
    plt.close()
