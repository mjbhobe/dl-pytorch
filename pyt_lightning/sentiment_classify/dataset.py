# -*- coding: utf-8 -*-
"""
dataset.py : utility functions to load the data & define the Dataset

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
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix

# Pytorch imports
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

DATA_PATH = pathlib.Path(__file__).parent / "data/sentiment+labelled+sentences"


def load_data(download_path: pathlib.Path = DATA_PATH):
    datasets = {}

    assert download_path.exists(), f"FATAL: {str(download_path)} does not exist!"
    csvs = list(download_path.glob("**/*.txt"))
    assert len(csvs) > 0, f"FATAL: no data files in {str(download_path)}"

    for f in csvs:
        logger.info(f"Processing: {f}")
        dataset_name = f.name.lower().split("_")[0]

        if dataset_name == "imdb":
            data = pd.read_csv(
                str(f), sep="\t", header=None, names=["review", "sentiment"]
            )
            datasets[dataset_name] = data

    return datasets


def get_dataset(key, vocab_size):

    datasets = load_data()

    if key in datasets.keys():
        df = datasets[key]

        if key == "imdb":

            # instantiate a vectorizer
            vectorizer = CountVectorizer(
                min_df=0.0,
                max_df=1.0,
                max_features=vocab_size,
                lowercase=True,
            )

            X = df["review"].values
            y = df["sentiment"].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42
            )
            X_train = vectorizer.fit_transform(X_train)
            X_test = vectorizer.transform(X_test)

            return (X_train, y_train), (X_test, y_test)
        else:
            logger.warning(f"{key} dataset not implemented yet!")
    else:
        raise ValueError(f"{key} - unrecognized dataset!")


class ImdbDataset(Dataset):
    """custom dataset for IMDB sentiments data"""

    def __init__(self, train_or_test, vocab_size):
        (X_train, y_train), (X_test, y_test) = get_dataset("imdb", vocab_size)

        # NOTE: X_train & X_test are sparse matrices created by CountVectorizer

        if train_or_test.lower() == "train":
            self.X = torch.from_numpy(X_train.todense()).float()
            self.y = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        else:
            self.X = torch.from_numpy(X_test.todense()).float()
            self.y = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        review = self.X[idx]
        label = self.y[idx]
        return (review, label)


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = get_dataset("imdb")
    print(
        f"X_train.shape: {X_train.shape} - y_train.shape: {y_train.shape} - X_test.shape: {X_test.shape} - y_test.shape: {y_test.shape}"
    )

    import pytorch_enlightning as pel

    train_dataset = ImdbDataset("train")
    test_dataset = ImdbDataset("test")
    train_dataset, val_dataset = pel.split_dataset(train_dataset, split_perc=0.2)

    print(
        f"train_dataset: {len(train_dataset)} recs - val_dataset: {len(val_dataset)} recs - test_dataset: {len(test_dataset)} recs"
    )

    # try iterating over train_dataset
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=5)
    for step, (review, sentiment) in enumerate(train_loader):
        print(f"{step+1} - {review} - {sentiment}")
        print(f"review.shape: {review.shape} - sentiment.shape: {sentiment.shape}")
        break
