# -*- coding: utf-8 -*-
"""
dataset.py : utility functions to download & display datasets (Fashion MNIST)

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import sys
import warnings
import pathlib
import logging
import logging.config

BASE_PATH = pathlib.Path(__file__).parent.parent
sys.path.append(str(BASE_PATH))

warnings.filterwarnings("ignore")
logging.config.fileConfig(fname=BASE_PATH / "logging.config")

import pathlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
plt.style.use("seaborn")
sns.set(style="whitegrid", font_scale=1.1, palette="muted")

# Pytorch imports
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


def get_datasets(download_path):
    xforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.FashionMNIST(
        root=download_path,
        train=True,
        download=True,
        transform=xforms,
    )

    test_dataset = datasets.FashionMNIST(
        root=download_path,
        train=False,
        download=True,
        transform=xforms,
    )
    val_dataset, test_dataset = torch.utils.data.random_split(
        test_dataset,
        [8000, 2000],
    )
    logger.info(
        f"get_datasets() -> train_dataset: {len(train_dataset)} recs - val_dataset: {len(val_dataset)} recs - test_dataset: {len(test_dataset)} recs"
    )
    return train_dataset, val_dataset, test_dataset


def display_sample(
    sample_images,
    sample_labels,
    grid_shape=(8, 8),
    plot_title=None,
    sample_predictions=None,
):
    # just in case these are not imported!
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("seaborn")

    num_rows, num_cols = grid_shape
    assert sample_images.shape[0] == num_rows * num_cols

    # a dict to help encode/decode the labels
    FASHION_LABELS = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }

    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=1.10)
        sns.set_style({"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})

        f, ax = plt.subplots(
            num_rows, num_cols, figsize=(14, 10), gridspec_kw={"wspace": 0.05, "hspace": 0.35}, squeeze=True
        )  # 0.03, 0.25
        # fig = ax[0].get_figure()
        f.tight_layout()
        f.subplots_adjust(top=0.90)  # 0.93

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")
                # show selected image
                ax[r, c].imshow(sample_images[image_index].squeeze(), cmap="Greys", interpolation="nearest")

                if sample_predictions is None:
                    # but show the prediction in the title
                    title = ax[r, c].set_title(f"{FASHION_LABELS[sample_labels[image_index]]}")
                else:
                    pred_matches_actual = sample_labels[image_index] == sample_predictions[image_index]
                    if pred_matches_actual:
                        # show title from prediction or actual in green font
                        title = "%s" % FASHION_LABELS[sample_predictions[image_index]]
                        title_color = "g"
                    else:
                        # show title as actual/prediction in red font
                        title = "%s/%s" % (
                            FASHION_LABELS[sample_labels[image_index]],
                            FASHION_LABELS[sample_predictions[image_index]],
                        )
                        title_color = "r"

                    # but show the prediction in the title
                    title = ax[r, c].set_title(title)
                    # if prediction is incorrect title color is red, else green
                    plt.setp(title, color=title_color)

        if plot_title is not None:
            plt.suptitle(plot_title)
        plt.show()
        plt.close()
