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
from typing import Dict, Tuple

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


def get_logger(file_path: pathlib.Path, level: int = logging.INFO) -> logging.Logger:
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
    logger.setLevel(logging.DEBUG)

    # stream_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    # file_formatter = logging.Formatter(
    #     "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # )
    formatter = logging.Formatter(
        "[%(name)s] [%(levelname)s] [%(asctime)s] %(message)s",
        datefmt="%Y-%b-%d %H:%M:%S",
    )

    # create handlers
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # Avoid duplicate logs by ensuring handlers are not added multiple times
    logger.propagate = False

    return logger


def diff_datetime(start: datetime, end: datetime) -> str:
    """
    Calculate the difference between two datetime objects and report it in days, hours, minutes, seconds, and milliseconds.

    Parameters:
        start (datetime): The starting datetime.
        end (datetime): The ending datetime.

    Returns:
        str: A formatted string reporting the difference.
    """
    # Ensure `end` is after `start`
    if end < start:
        start, end = end, start

    # Calculate the time delta
    delta = end - start

    # Extract days, seconds, and microseconds from timedelta
    days = delta.days
    seconds = delta.seconds
    microseconds = delta.microseconds

    # Calculate hours, minutes, seconds, and milliseconds
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    milliseconds = microseconds // 1000

    # Build the output string
    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days > 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
    if seconds > 0:
        parts.append(f"{seconds} second{'s' if seconds > 1 else ''}")
    if milliseconds > 0:
        parts.append(f"{milliseconds} millisecond{'s' if milliseconds > 1 else ''}")

    # Join the parts with commas
    return ", ".join(parts) if parts else "0 milliseconds"


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


def denormalize_and_permute_images(
    images: torch.Tensor, mean: tuple, std: tuple
) -> torch.Tensor:
    """
    denormalizes a batch of images (Tensors) using provided mean & std-deviation
    and clamps the output between (0, 1).
    Executes operation (images * std) + mean

    @params:
        images - batch of images as a torch.Tensor with shape (batch_size, channels, width, height)
        mean - mean to be applied - tuple of floats or ints [e.g. (0.5, 0.5, 0.5)]
        std - std deviation to be applied - tuple of floats or ints [e.g. (0.5, 0.5, 0.5)]
    @returns:
        images - denormalized batch of image tensors of same shape as input parameter
    """
    mu_t = torch.Tensor(mean)
    std_t = torch.Tensor(std)
    images_x = images * std_t[:, None, None] + mu_t[:, None, None]
    images_x = images_x.clamp(0, 1)
    images_x = images_x.permute(0, 2, 3, 1)
    return images_x


def display_images_grid(
    sample_images: np.ndarray,  # de-normalized & transposed images!!
    sample_labels: np.ndarray,
    labels_dict: Dict[int, str] = None,
    grid_shape: tuple = (10, 10),
    plot_title: str = None,
    fig_size: Tuple[int, int] = (14, 10),
    sample_predictions: np.ndarray = None,
):
    """
    displays grid of images with labels on top of each image (if provided) and plot tite (if provided)

    @params:
        sample_images : Numpy array of DENORMALIZED images [shape: (batch_size, c, w, h)]
        sample_labels : Numpy array of ints
    """
    # just in case these are not imported!
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("seaborn-v0_8")

    num_rows, num_cols = grid_shape
    assert sample_images.shape[0] == num_rows * num_cols

    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=1.0)
        sns.set_style(
            {
                "font.sans-serif": [
                    "SF Pro Rounded",
                    "Calibri",
                    "Arial",
                    "DejaVu Sans",
                    "sans",
                ]
            }
        )

        f, ax = plt.subplots(
            num_rows,
            num_cols,
            figsize=fig_size,
            gridspec_kw={"wspace": 0.3, "hspace": 0.5},
            squeeze=True,
        )  # 0.03, 0.25
        # fig = ax[0].get_figure()
        f.tight_layout()
        f.subplots_adjust(top=0.90)  # 0.93

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")
                image = sample_images[image_index]

                # show selected image
                ax[r, c].imshow(
                    image,
                    cmap="Greys",
                    interpolation="nearest",
                )

                if sample_predictions is None:
                    # show the text label as image title
                    title = (
                        ""
                        if labels_dict is None
                        else f"{labels_dict[sample_labels[image_index]]}"
                    )
                    title = ax[r, c].set_title(title)
                else:
                    pred_matches_actual = (
                        sample_labels[image_index] == sample_predictions[image_index]
                    )
                    # show prediction from model as image title
                    title = (
                        ""
                        if labels_dict is None
                        else f"{labels_dict[sample_predictions[image_index]]}"
                    )
                    if pred_matches_actual:
                        # if matches, title color is green
                        title_color = "g"
                    else:
                        # else title color is red
                        title_color = "r"

                    # but show the prediction in the title
                    title = ax[r, c].set_title(title, fontsize=8)
                    # if prediction is incorrect title color is red, else green
                    plt.setp(title, color=title_color)

        if plot_title is not None:
            plt.suptitle(plot_title)
        plt.show()
        plt.close()
