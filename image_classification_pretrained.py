#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
image_classification_pretrained.py - use pre-trained Pytorch model to classify images

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings
import logging
import logging.config

warnings.filterwarnings("ignore")
logging.config.fileConfig(fname="logging.config")

import sys
import os
import pathlib
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
plt.style.use("seaborn")
sns.set(style="whitegrid", font_scale=1.1, palette="muted")

# Pytorch imports
import torch
import torch.nn as nn
from torchvision import transforms, models

print("Using Pytorch version: ", torch.__version__)

# My helper functions for training/evaluating etc.
import torch_training_toolkit as t3

SEED = t3.seed_all()

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = pathlib.Path(__file__).parent / "csv_files" / "data_banknote_authentication.txt"
assert os.path.exists(DATA_FILE_PATH), f"FATAL: {DATA_FILE_PATH} - data file does not exist!"

logger.info(f"Training model on {DEVICE}")
logger.info(f"Using data file {DATA_FILE_PATH}")


def get_model():
    # download the pre-trained resnet18 model
    resnet = models.resnet101(pretrained=True)
    return resnet


def preprocess(image):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transform(image)


def imshow(img, title=None, fig_size=None, show_axis=False):
    if fig_size is not None:
        plt.figure(figsize=fig_size)
    if title is not None:
        plt.title(title)
    if not show_axis:
        plt.axis("off")
    plt.imshow(img, cmap="Greys")
    plt.show()


def main():
    # download data
    parser = argparse.ArgumentParser("Pytorch Pre-trained classifiers")
    parser.add_argument(
        "-i",
        "--image",
        required=True,
        help="Full path of image to classify",
    )
    args = parser.parse_args()

    image_path = args.image
    if pathlib.Path(image_path).exists():
        image = Image.open(image_path)
        imshow(image, title=str(image_path))
    else:
        raise ValueError(f"FATAL ERROR: path does not exist {image_path}!")

    # apply transforms
    image_tensor = preprocess(image)
    # make batch
    batch = torch.unsqueeze(image_tensor, 0)

    # load class labels
    classes = {}
    classes_file_path = pathlib.Path("./imagenet_classes.txt")
    if classes_file_path.exists():
        with open(classes_file_path, "r") as f:
            for i, line in enumerate(f.readlines()):
                classes[i] = line.strip()
    else:
        raise ValueError(f"FATAL ERROR: cannot read label files {str(classes_file_path)}")

    # instantiate the model
    resnet = get_model()
    resnet.eval()
    out = resnet(batch)
    print(out)
    # find the index of max value
    value, index = torch.max(out, 1)
    print(f"Max value is {value.item():.2f} at index {index[0]}")
    print(f"Prediction(s): {classes[index[0].item()]}")
    image_title = f"{image_path}\nPredictions: ({classes[index[0].item()]})"
    imshow(image, title=str(image_title))


if __name__ == "__main__":
    main()
