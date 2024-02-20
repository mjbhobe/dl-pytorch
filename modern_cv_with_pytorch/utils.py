#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" utils.py - utility functions """
import sys
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch_training_toolkit as t3

MEANS, STDS = 0.5, 0.5


def get_data(data_dir, debug=False):
    xforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEANS, STDS)]
    )

    # data_dir = os.path.join(os.getcwd(), "data")

    train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=xforms
    )

    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=xforms
    )

    # split the test dataset into test/cross-val sets
    val_dataset, test_dataset = t3.split_dataset(test_dataset, split_perc=0.20)

    if debug:
        print(
            f"train_dataset: {len(train_dataset)} recs - val_dataset: {len(val_dataset)} recs - "
            f"test_dataset: {len(test_dataset)} recs"
        )
        print(f"classes: {train_dataset.classes}")

    return train_dataset, val_dataset, test_dataset, train_dataset.classes


def get_data_cifar10(data_dir, debug=False):
    xforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEANS, STDS)]
    )

    # data_dir = os.path.join(os.getcwd(), "data")

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=xforms
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=xforms
    )

    # split the test dataset into test/cross-val sets
    val_dataset, test_dataset = t3.split_dataset(test_dataset, split_perc=0.20)

    if debug:
        print(
            f"train_dataset: {len(train_dataset)} recs - val_dataset: {len(val_dataset)} recs - "
            f"test_dataset: {len(test_dataset)} recs"
        )
        print(f"classes: {train_dataset.classes}")

    return train_dataset, val_dataset, test_dataset, train_dataset.classes


def display_sample(dataset, class_names, title=None, fig_size=(8, 8)):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    images, labels = next(iter(dataloader))
    images, labels = images.cpu().numpy(), labels.cpu().numpy()
    num_rows, num_cols = 8, 8

    with sns.axes_style("whitegrid"):
        plt.style.use("seaborn-v0_8")
        sns.set(context="notebook", font_scale=0.95)
        sns.set_style(
            {
                "font.sans-serif": [
                    "SF UI Text",
                    "Calibri",
                    "Arial",
                    "DejaVu Sans",
                    "sans",
                ]
            }
        )

        f, ax = plt.subplots(num_rows, num_cols, figsize=fig_size)
        f.tight_layout()
        f.subplots_adjust(top=0.90)

        for r in range(num_rows):
            for c in range(num_cols):
                index = r * num_cols + c
                ax[r, c].axis("off")
                sample_image = images[index]
                sample_image = sample_image.transpose((1, 2, 0))
                sample_image = (sample_image * STDS) + MEANS
                ax[r, c].imshow(sample_image, cmap="Greys", interpolation="nearest")
                ax[r, c].set_title(class_names[labels[index]])

    if title is not None:
        plt.suptitle(title)

    plt.show()


if __name__ == "__main__":
    print(f"Oops! It looks like you are running a utility functions module: {__file__}")
