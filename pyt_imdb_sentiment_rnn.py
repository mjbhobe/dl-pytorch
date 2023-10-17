# -*- coding: utf-8 -*-
"""
pyt_imdb_sentiment_rnn.py: Sentiment analysis of IMDB reviewes using RNNs

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings

warnings.filterwarnings("ignore")

import re
from collections import Counter, OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use("ggplot")
sns.set(style="darkgrid", context="notebook", font_scale=1.20)

# Pytorch imports
import torch

print("Using Pytorch version: ", torch.__version__)
import torch.nn as nn
from torchtext.datasets import IMDB
from torchtext.vocab import vocab
import torchmetrics
import torchsummary

# My helper functions for training/evaluating etc.
import torch_training_toolkit as t3

# to ensure that you get consistent results across runs & machines
# @see: https://discuss.pytorch.org/t/reproducibility-over-different-machines/63047
SEED = t3.seed_all()
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training model on {DEVICE}")


def tokenizer(text):
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text.lower())
    text = re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", "")
    tokenized = text.split()
    return tokenized


def main():
    train_dataset = IMDB(split="train")
    test_dataset = IMDB(split="test")

    from torch.utils.data.dataset import random_split

    train_dataset, valid_dataset = random_split(list(train_dataset), [12_000, 500])
    token_counts = Counter()
    for label, line in train_dataset:
        tokens = tokenizer(line)
        token_counts.update(tokens)
    print(f"Vocab size: {len(token_counts)}")

    # create a vocabulary from ordered dict of tokens
    sorted_by_freq_tuples = sorted(
        token_counts.items(), key=lambda x: x[1], reverse=True
    )
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    vocabulary = vocab(ordered_dict)
    vocabulary.insert_token("<pad>", 0)
    vocabulary.insert_token("<unk>", 1)
    vocabulary.set_default_index(1)
    # display some mappings of words from vocabulary
    print([vocabulary[tok] for tok in ["this", "is", "an", "example", "extemporary"]])


if __name__ == "__main__":
    main()
