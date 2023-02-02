# -*- coding: utf-8 -*-
""" dataset_utils.py - utility functions for Pytorch datasets """
import warnings

warnings.filterwarnings('ignore')

# Pytorch imports
import torch


def split_dataset(dataset: torch.utils.data.Dataset, split_perc: float = 0.20):
    """ randomly splits a dataset into 2 based on split percentage (split_perc)
        @params:
            - dataset (torch.utils.data.Dataset): the dataset to split
            - split_perc (float) : defines ratio (>= 0.0 and <= 1.0) for number
                of records in 2nd split. Default = 0.2
            Example: if dataset has 100 records and split_perc = 0.2, then
            2nd dataset will have 0.2 * 100 = 20 randomly selected records
            and first dataset will have (100 - 20 = 80) records.
        @returns: tuple of datasets (split_1, split_2)
    """
    assert (split_perc >= 0.0) and (split_perc <= 1.0), \
        f"FATAL ERROR: invalid split_perc value {split_perc}." \
        f"Expecting float >= 0.0 and <= 1.0"

    if split_perc > 0.0:
        num_recs = len(dataset)
        train_count = int((1.0 - split_perc) * num_recs)
        test_count = num_recs - train_count
        train_dataset, test_dataset = \
            torch.utils.data.random_split(dataset, [train_count, test_count])
        return train_dataset, test_dataset
    else:
        return dataset, None
