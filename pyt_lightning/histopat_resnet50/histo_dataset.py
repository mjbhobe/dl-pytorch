# -*- coding: utf-8 -*-
"""
histo_dataset.py : utility functions to download & display datasets 
    (Histopathological Cancer detection dataset)
    NOTE: we are using the same dataset as used in the histopat model

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import sys
import warnings
import pathlib
import logging.config

BASE_PATH = pathlib.Path(__file__).parent.parent
sys.path.append(str(BASE_PATH))

warnings.filterwarnings("ignore")
logging.config.fileConfig(fname=BASE_PATH / "logging.config")

import os, pathlib, shutil
import opendatasets as od
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
plt.style.use("seaborn-v0_8")
sns.set(style="whitegrid", font_scale=1.1, palette="muted")

# Pytorch imports
from torchvision import transforms
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)


def get_datasets(
    download_path, force_download=False, force_recreate=False, random_state=42
):
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

        assert (
            train_data_path.exists()
        ), f"FATAL ERROR: {train_data_path} does not exist!"
        assert test_data_path.exists(), f"FATAL ERROR: {test_data_path} does not exist!"
        assert (
            train_labels_file.exists()
        ), f"FATAL ERROR: {train_labels_file} does not exist!"

        np.random.seed(random_state)
        labels_df = pd.read_csv(str(train_labels_file))
        for _ in range(5):
            labels_df.sample(frac=1)

        # dataset has 220,025 records
        # ** old code **
        # we'll downsample 25,000 records and use 15,000 train, 8,000 cross-val and 2,000 test
        # downsample_size, num_train, num_valid, num_test = 25000, 15000, 8000, 2000

        # ** new code **
        # Since we are training on Colab (no data restrictions), we'll use all records
        downsample_size = 75000  # len(labels_df)
        num_test = int(10 * downsample_size / 100.0)
        num_valid = int(20 * downsample_size / 100.0)
        num_train = downsample_size - (num_valid + num_test)
        logger.info(
            f"Trying to split data into {num_train} training, {num_valid} cross-val and {num_test} testing records"
        )

        # downsample a subset of downsample_size using stratified sampling
        # downsample_df = labels_df.sample(n=downsample_size)
        f = (downsample_size * 1.0) / len(labels_df)
        downsample_df = labels_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(frac=f)
        )

        # split the subset into train, cross-val & test sets using stratified sampling
        # calculate correct fraction
        f = (num_train * 1.0) / len(downsample_df)
        downsample_train_df = downsample_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(frac=f)
        )
        assert (
            len(downsample_train_df) == num_train
        ), f"FATAL: something wrong, expecting {num_train} records in train dataset, sampled {len(downsample_train_df)}"

        downsample_rest_df = downsample_df.drop(downsample_train_df.index)
        f = (num_valid * 1.0) / len(downsample_rest_df)
        downsample_eval_df = downsample_rest_df.groupby(
            "label", group_keys=False
        ).apply(lambda x: x.sample(frac=f))
        assert (
            len(downsample_eval_df) == num_valid
        ), f"FATAL: something wrong, expecting {num_valid} records in eval dataset, sampled {len(downsample_eval_df)}"

        downsample_test_df = downsample_rest_df.drop(downsample_eval_df.index)
        assert (
            len(downsample_test_df) == num_test
        ), f"FATAL: something wrong, expecting {num_test} records in test dataset, sampled {len(downsample_test_df)}"

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
        # re-create the folders
        (downsample_train_path / "benign").mkdir(parents=True, exist_ok=True)
        (downsample_train_path / "malignant").mkdir(parents=True, exist_ok=True)

        if downsample_eval_path.exists():
            logger.info(
                f"Deleting existing cross-val images from {downsample_eval_path}..."
            )
            shutil.rmtree(downsample_eval_path)
        # re-create the folders
        (downsample_eval_path / "benign").mkdir(parents=True, exist_ok=True)
        (downsample_eval_path / "malignant").mkdir(parents=True, exist_ok=True)

        if downsample_test_path.exists():
            logger.info(f"Deleting existing test images from {downsample_test_path}...")
            shutil.rmtree(downsample_test_path)
        # re-create the folders
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
        len(os.listdir(downsample_train_path / "benign")), len(os.listdir(downsample_train_path / "malignant"))
    logger.info(
        f"Training images -> benign: {benign_image_count} - malignant: {malignant_image_count} - total: {benign_image_count + malignant_image_count}"
    )
    benign_image_count, malignant_image_count = \
        len(os.listdir(downsample_eval_path / "benign")), len(os.listdir(downsample_eval_path / "malignant"))
    logger.info(
        f"Cross-val images -> benign: {benign_image_count} - malignant: {malignant_image_count} - total: {benign_image_count + malignant_image_count}"
    )
    benign_image_count, malignant_image_count = \
        len(os.listdir(downsample_test_path / "benign")), len(os.listdir(downsample_test_path / "malignant"))
    logger.info(
        f"Test images -> benign: {benign_image_count} - malignant: {malignant_image_count} - total: {benign_image_count + malignant_image_count}"
    )
    # fmt:on

    # NOTE: ResNet50 expects source images to be 224 x 224 size
    train_xforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.ToTensor(),
        ]
    )
    test_xforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    )

    # check bincounts of labels for imbalance - the loss fuction will have
    # to be weighted accordingly
    # np.random.seed(random_state)
    # train_labels_file = download_base_path / "train_labels.csv"
    # labels_df = pd.read_csv(str(train_labels_file))
    # num_benign, num_malignant = np.bincount(labels_df["label"])
    # logger.info(f"Label distribution -> benign: {num_benign} - malignant: {num_malignant}")

    train_dataset = ImageFolder(str(downsample_train_path), transform=train_xforms)
    val_dataset = ImageFolder(str(downsample_eval_path), transform=test_xforms)
    test_dataset = ImageFolder(str(downsample_test_path), transform=test_xforms)

    return (
        benign_image_count,
        malignant_image_count,
        train_dataset,
        val_dataset,
        test_dataset,
    )


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

    plt.style.use("seaborn-v0_8")

    num_rows, num_cols = grid_shape
    assert sample_images.shape[0] == num_rows * num_cols

    # a dict to help encode/decode the labels
    LABELS = {
        0: "Benign",
        1: "Malignant",
    }

    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=1.0)

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


def plot_confusion_matrix(
    conf_matrix, font_size=10, fig_size=(4, 4), cmap=plt.cm.Blues
):
    """graphical plot of confusion matric"""
    fig, ax = plt.subplots(figsize=fig_size)
    ax.matshow(conf_matrix, cmap=cmap, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va="center", ha="center")

    plt.xlabel("Predictions", fontsize=font_size)
    plt.ylabel("Actuals", fontsize=font_size)
    plt.title("Confusion Matrix", fontsize=font_size)
    plt.show()
