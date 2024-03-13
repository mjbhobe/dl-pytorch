# -*- coding: utf-8 -*-
"""
histo_dataset.py : utility functions to download & display datasets 
    (Histopathological Cancer detection dataset)

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

import os, pathlib, shutil
import opendatasets as od
from zipfile import ZipFile
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# tweaks for libraries
plt.style.use("seaborn-v0_8")
sns.set(style="whitegrid", font_scale=1.1, palette="muted")

# Pytorch imports
import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


def get_datasets(download_path, force_download=False, force_recreate=False):
    dataset_url = "https://www.kaggle.com/c/histopathologic-cancer-detection"
    # download dataset from Kaggle using opendatasets
    # NOTE: save your kaggle.json file to local folder
    od.download(dataset_url, data_dir=download_path, force=force_download)

    download_base_path = (
        pathlib.Path(download_path) / "histopathologic-cancer-detection"
    )
    downsample_base_path = download_base_path / "downsample"
    downsample_train_path = downsample_base_path / "train"
    downsample_eval_path = downsample_base_path / "eval"
    downsample_test_path = downsample_base_path / "test"
    num_benign, num_malignant = None, None

    if (not downsample_base_path.exists()) or force_recreate:
        train_data_path = download_base_path / "train"
        test_data_path = download_base_path / "test"
        train_labels_file = download_base_path / "train_labels.csv"
        # we will downsample 25,000 images from a
        downsample_size = 25000

        assert (
            train_data_path.exists()
        ), f"FATAL ERROR: {train_data_path} does not exist!"
        assert test_data_path.exists(), f"FATAL ERROR: {test_data_path} does not exist!"
        assert (
            train_labels_file.exists()
        ), f"FATAL ERROR: {train_labels_file} does not exist!"

        np.random.seed(42)
        labels_df = pd.read_csv(str(train_labels_file))
        for _ in range(5):
            labels_df.sample(frac=1)

        # downsample a subset of downsample_size using stratified sampling
        # downsample_df = labels_df.sample(n=downsample_size)
        f = (downsample_size * 1.0) / len(labels_df)
        downsample_df = labels_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(frac=f)
        )
        # split the subset into train: 15,000, eval: 8,000 and test: 2,000
        # downsample_train_df = downsample_df.sample(n=15000)
        f = (15000 * 1.0) / len(downsample_df)
        downsample_train_df = downsample_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(frac=f)
        )
        assert (
            len(downsample_train_df) == 15000
        ), f"FATAL: something wrong, expecting 15,000 records in train dataset, sampled {len(downsample_train_df)}"
        downsample_rest_df = downsample_df.drop(downsample_train_df.index)
        f = (8000 * 1.0) / len(downsample_rest_df)
        downsample_eval_df = downsample_rest_df.groupby(
            "label", group_keys=False
        ).apply(lambda x: x.sample(frac=f))
        assert (
            len(downsample_eval_df) == 8000
        ), f"FATAL: something wrong, expecting 8,000 records in eval dataset, sampled {len(downsample_eval_df)}"
        downsample_test_df = downsample_rest_df.drop(downsample_eval_df.index)
        assert (
            len(downsample_test_df) == 2000
        ), f"FATAL: something wrong, expecting 2,000 records in test dataset, sampled {len(downsample_test_df)}"

        logger.info(
            f"Sample sizes -> train: {len(downsample_train_df)} - "
            f"eval: {len(downsample_eval_df)} - "
            f"test: {len(downsample_test_df)}"
        )

        if downsample_train_path.exists():
            logger.info(
                f"Deleting existing training images from {downsample_train_path}..."
            )
            shutil.rmtree(downsample_train_path)
        (downsample_train_path / "benign").mkdir(parents=True, exist_ok=True)
        (downsample_train_path / "malignant").mkdir(parents=True, exist_ok=True)

        if downsample_eval_path.exists():
            logger.info(
                f"Deleting existing cross-val images from {downsample_eval_path}..."
            )
            shutil.rmtree(downsample_eval_path)
        (downsample_eval_path / "benign").mkdir(parents=True, exist_ok=True)
        (downsample_eval_path / "malignant").mkdir(parents=True, exist_ok=True)

        if downsample_test_path.exists():
            logger.info(f"Deleting existing test images from {downsample_test_path}...")
            shutil.rmtree(downsample_test_path)
        (downsample_test_path / "benign").mkdir(parents=True, exist_ok=True)
        (downsample_test_path / "malignant").mkdir(parents=True, exist_ok=True)

        # recreate the train, eval & test sets
        outcomes = ["benign", "malignant"]
        datasets = ["train", "eval", "test"]
        dataframes = [downsample_train_df, downsample_eval_df, downsample_test_df]
        folder_paths = [
            downsample_train_path,
            downsample_eval_path,
            downsample_test_path,
        ]

        from tqdm import trange
        from time import sleep

        for dataset, df, folder_path in zip(datasets, dataframes, folder_paths):
            # iterate over the dataframe & copy files
            logger.info(f"Creating {dataset} images")
            t = trange(len(df), desc=f"Creating {dataset} images", leave=True)
            # for i in range(len(df)):
            for i in t:
                image_file_name, outcome = df.iloc[i, 0], outcomes[df.iloc[i, 1]]
                image_source_path = train_data_path / (image_file_name + ".tif")
                assert (
                    image_source_path.exists()
                ), f"FATAL: Source image {image_source_path} does not exist!"
                image_target_path = folder_path / outcome / (image_file_name + ".tif")
                t.set_postfix({"Copying": f"{(image_file_name + '.tif')}"})
                shutil.copy(image_source_path, image_target_path)
                sleep(0.01)
                # logger.info(f"  shutil.copy('{image_source_path}', '{image_target_path}')")

    # display counts
    # fmt:off
    benign_image_count, malignant_image_count = \
        len(os.listdir(downsample_train_path / "benign")), len(
            os.listdir(downsample_train_path / "malignant")
        )
    logger.info(
        f"Training images -> benign: {benign_image_count} - malignant: {malignant_image_count} - total: {benign_image_count + malignant_image_count}"
    )
    benign_image_count, malignant_image_count = \
        len(os.listdir(downsample_eval_path / "benign")), len(
            os.listdir(downsample_eval_path / "malignant")
        )
    logger.info(
        f"Cross-val images -> benign: {benign_image_count} - malignant: {malignant_image_count} - total: {benign_image_count + malignant_image_count}"
    )
    benign_image_count, malignant_image_count = \
        len(os.listdir(downsample_test_path / "benign")), len(
            os.listdir(downsample_test_path / "malignant")
        )
    logger.info(
        f"Test images -> benign: {benign_image_count} - malignant: {malignant_image_count} - total: {benign_image_count + malignant_image_count}"
    )
    # fmt:on

    train_xforms = transforms.Compose(
        [
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(25),
            transforms.ToTensor(),
        ]
    )
    test_xforms = transforms.Compose(
        [
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ]
    )

    # check bincounts of labels for imbalance - the loss fuction will have
    # to be weighted accordingly
    np.random.seed(42)
    train_labels_file = download_base_path / "train_labels.csv"
    labels_df = pd.read_csv(str(train_labels_file))
    num_benign, num_malignant = np.bincount(labels_df["label"])
    logger.info(
        f"Label distribution -> benign: {num_benign} - malignant: {num_malignant}"
    )

    train_dataset = ImageFolder(str(downsample_train_path), transform=train_xforms)
    val_dataset = ImageFolder(str(downsample_eval_path), transform=test_xforms)
    test_dataset = ImageFolder(str(downsample_test_path), transform=test_xforms)

    return num_benign, num_malignant, train_dataset, val_dataset, test_dataset


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
    LABELS = {
        0: "Benign",
        1: "Malignant",
    }

    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=0.75)
        sns.set_style(
            {
                "font.sans-serif": [
                    "SF Pro Display",
                    "Segoe UI",
                    "Calibri",
                    "Arial",
                    "DejaVu Sans",
                    "Sans",
                ],
            }
        )

        f, ax = plt.subplots(
            num_rows,
            num_cols,
            figsize=(14, 10),
            gridspec_kw={"wspace": 0.05, "hspace": 0.55},
            squeeze=True,
        )
        f.tight_layout()
        f.subplots_adjust(top=0.90)

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")
                # show selected image (convert CHW to HWC)
                sample_image = sample_images[image_index].transpose(1, 2, 0)
                ax[r, c].imshow(
                    sample_image,
                    cmap="Greys",
                    interpolation="nearest",
                )

                if sample_predictions is None:
                    # but show the prediction in the title
                    title = ax[r, c].set_title(f"{LABELS[sample_labels[image_index]]}")
                else:
                    pred_matches_actual = (
                        sample_labels[image_index] == sample_predictions[image_index]
                    )
                    if pred_matches_actual:
                        # show title from prediction or actual in green font
                        title = "%s" % LABELS[sample_predictions[image_index]]
                        title_color = "g"
                    else:
                        # show title as actual/prediction in red font
                        title = "%s/%s" % (
                            LABELS[sample_labels[image_index]],
                            LABELS[sample_predictions[image_index]],
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
