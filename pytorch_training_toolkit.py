""" pytorch_training_toolkit.py - utility functions & classes to help with training Pytorch models"""
import warnings

warnings.filterwarnings('ignore')

import sys

if sys.version_info < (2,):
    raise Exception(
        "torch_training_toolkit does not support Python 1. Please use a Python 3+ interpreter!"
    )

import os
import random
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Pytorch imports
import torch
import torch.nn as nn
import torchmetrics

# print('Using Pytorch version: ', torch.__version__)

# print(f"Using torchmetrics: {torchmetrics.__version__}")

version_info = (1, 0, 0, "dev0")

__version__ = '.'.join(map(str, version_info))
__installer_version__ = __version__
__title__ = "Torch Training Toolkit (t3)"
__author__ = "Manish Bhobé"
__organization__ = "Nämostuté Ltd."
__org_domain__ = "namostute.pytorch.in"
__license__ = __doc__
__project_url__ = "https://github.com/mjbhobe/dl_pytorch"

T3_FAV_SEED = 41


# DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# ----------------------------------------------------------------------------------
# utility functions
# ----------------------------------------------------------------------------------
def seed_all(seed = None):
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
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    return seed


def get_logger(name: str, level: int = logging.WARNING) -> logging.Logger:
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    logger.addHandler(formatter)
    logger.setLevel(level)
    return logger


def plot_confusion_matrix(
    cm, class_names = None, title = "Confusion Matrix",
    cmap = plt.cm.Purples,
    fig_size = (8, 6)
):
    """ graphical plot of the confusion matrix
        @params:
            cm - the confusion matrix (value returned by the sklearn.metrics.confusion_matrix(...) call)
            class_names (list) - names (text) of classes you want to use (list of strings)
            title (string, default='Confusion Matrix') - title of the plot
            cmap (matplotlib supported palette, default=plt.cm.Blues) - color palette you want to use
    """

    class_names = ['0', '1'] if class_names is None else class_names
    df = pd.DataFrame(cm, index = class_names, columns = class_names)

    plt.figure(figsize = fig_size)
    with sns.axes_style("darkgrid"):
        # sns.set_context("notebook")  # , font_scale = 1.1)
        sns.set_style(
            {
                "font.sans-serif": ["Segoe UI", "Calibri", "SF Pro Display", "Arial",
                                    "DejaVu Sans", "Sans"]
            }
        )
        hmap = sns.heatmap(df, annot = True, fmt = "d", cmap = cmap)
        hmap.yaxis.set_ticklabels(
            hmap.yaxis.get_ticklabels(), rotation = 0, ha = 'right'
        )
        hmap.xaxis.set_ticklabels(
            hmap.xaxis.get_ticklabels(), rotation = 30, ha = 'right'
        )

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(title)
    plt.show()
    plt.close()


# ----------------------------------------------------------------------------------
# convenience functions to create layers with weights & biases initialized
# ----------------------------------------------------------------------------------
def Linear(in_features, out_features, bias = True, device = None, dtype = None):
    """
        (convenience function)
        creates a nn.Linear layer, with weights initiated using xavier_uniform initializer
        and bias, if set, initialized using zeros initializer as is the default in Keras.
        @params:
        - in_nodes: # of nodes from pervious layer
        - out_nodes: # of nodes in this layer
        @returns:
        - an instance of nn.Linear class
    """
    layer = nn.Linear(in_features, out_features, bias, device, dtype)
    # @see: https://msdn.microsoft.com/en-us/magazine/mt833293.aspx for example
    torch.nn.init.xavier_uniform_(layer.weight)
    if bias:
        torch.nn.init.zeros_(layer.bias)
    return layer


def Conv2d(
    in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1,
    dilation = 1, groups = 1, bias = True, padding_mode = 'zeros',
    device = None, dtype = None
):
    """
        (convenience function)
        Creates a nn.Conv2d layer, with weights initiated using xavier_uniform initializer
        and bias, if set, initialized using zeros initializer. This is similar to Keras.
        @params:
            - same as nn.Conv2d params
        @returns:
            - instance of nn.Conv2d layer
    """
    layer = nn.Conv2d(
        in_channels, out_channels, kernel_size = kernel_size,
        stride = stride, padding = padding, dilation = dilation,
        groups = groups, bias = bias, padding_mode = padding_mode,
        device = device, dtype = dtype
    )
    # @see: https://msdn.microsoft.com/en-us/magazine/mt833293.aspx for example
    torch.nn.init.xavier_uniform_(layer.weight)
    if bias:
        torch.nn.init.zeros_(layer.bias)
    return layer


# ----------------------------------------------------------------------------------
# dataset related functions
# ----------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------
# MetricsHistory
# ----------------------------------------------------------------------------------
class MetricsHistory:
    """ class to calculate & store metrics across training batches """

    def __init__(self, metrics_map, include_val_metrics = False):
        """ constructor of MetricsHistory class
            @params:
                - metrics_map: map of metric alias and calculation function
                    example:
                        metrics_map = {
                            "acc": torchmetrics.classification.BinaryAccuracy(),
                            "f1" : torchmetrics.classification.BinaryF1Score()
                        }
                        or - for multi-class classification
                        metrics_map = {
                            "acc": torchmetrics.classification.MultiClassAccuracy(4),
                            "f1" : torchmetrics.classification.MultiClassF1Score(4)
                        }
                - include_val_metrics (boolean) - True if validation metrics must also
                    be tracked else False
        """
        self.metrics_map = metrics_map
        self.include_val_metrics = include_val_metrics
        self.metrics_history = self.__createMetricsHistory()

    def tracked_metrics(self) -> list:
        """ returns list of all metrics tracked """
        metric_names = ["loss"]
        if self.metrics_map is not None:
            metric_names.extend([key for key in self.metrics_map.keys()])
        return metric_names

    def __createMetricsHistory(self):
        """ Internal function: creates a map to store metrics history """
        """
        metrics_map = {
            "acc" : acc_calc_fxn,
            "f1"  : f1_calc_fxn,
        }
        metrics_history = { 
            "loss" : { 
                "batch_vals": [],
                "epoch_vals": []
            },
            "acc"  : { 
                "batch_vals": [],
                "epoch_vals": []
            },
            ...
            [optional - present only if validation dataset used]
            "val_loss" : {
                "batch_vals": [],
                "epoch_vals": []
            },
            "val_acc" : {
                "batch_vals": [],
                "epoch_vals": []
            },
            ...
        }               
        """
        metrics_history = {}

        # must add key for loss
        metric_names = ["loss"]
        # rest will depend on metrics defined in metrics_map
        if self.metrics_map is not None:
            metric_names.extend([key for key in self.metrics_map.keys()])

        for metric_name in metric_names:
            metrics_history[metric_name] = {
                "batch_vals": [],
                "epoch_vals": []
            }

            if self.include_val_metrics:
                metrics_history[f"val_{metric_name}"] = {
                    "batch_vals": [],
                    "epoch_vals": []
                }
        return metrics_history

    def get_metric_vals(self, metrics_list, include_val_metrics = False):
        """ gets the last epoch value for each metric tracked """
        metrics_list2 = ["loss"] if metrics_list is None else metrics_list
        metric_vals = {
            metric_name: self.metrics_history[metric_name]["epoch_vals"][-1]
            for metric_name in metrics_list2
        }
        if include_val_metrics:
            for metric_name in metrics_list2:
                metric_vals[f"val_{metric_name}"] = \
                    self.metrics_history[f"val_{metric_name}"]["epoch_vals"][-1]
        return metric_vals

    def calculate_batch_metrics(
        self, preds: torch.tensor, targets: torch.tensor,
        loss_val: float, val_metrics = False
    ):
        if val_metrics:
            self.metrics_history["val_loss"]["batch_vals"].append(loss_val)
        else:
            self.metrics_history["loss"]["batch_vals"].append(loss_val)

        if self.metrics_map is not None:
            for metric_name, calc_fxn in self.metrics_map.items():
                metric_val = calc_fxn(preds, targets).item()
                if val_metrics:
                    self.metrics_history[f"val_{metric_name}"]["batch_vals"].append(
                        metric_val
                    )
                else:
                    self.metrics_history[metric_name]["batch_vals"].append(metric_val)

    def calculate_epoch_metrics(self, val_metrics = False):
        """ calculates average value of the accumulated metrics from last batch & appends
            to epoch metrics list
        """
        metric_names = self.tracked_metrics()

        for metric in metric_names:
            if val_metrics:
                mean_val = np.array(
                    self.metrics_history[f"val_{metric}"]["batch_vals"]
                ).mean()
                self.metrics_history[f"val_{metric}"]["epoch_vals"].append(
                    mean_val
                )
            else:
                mean_val = np.array(
                    self.metrics_history[metric]["batch_vals"]
                ).mean()
                self.metrics_history[metric]["epoch_vals"].append(mean_val)

    def clear_batch_metrics(self):
        """ reset the lists that track batch metrics """
        metric_names = self.tracked_metrics()

        for metric in metric_names:
            self.metrics_history[metric]["batch_vals"].clear()
            if self.include_val_metrics:
                self.metrics_history[f"val_{metric}"]["batch_vals"].clear()

    def get_metrics_str(
        self, batch_metrics = True, include_val_metrics = False
    ):
        # will not include loss
        metric_names = self.tracked_metrics()
        metrics_str = ""
        for metric_name in metric_names:
            # first display all training metrics & then the cross-val metrics
            if batch_metrics:
                # batch metrics (pick last value)
                metric_val = self.metrics_history[metric_name]["batch_vals"][-1]
            else:
                # epoch metrics (pick last value in list)
                metric_val = self.metrics_history[metric_name]["epoch_vals"][-1]
            metrics_str += f"{metric_name}: {metric_val:.4f} - "

        if include_val_metrics:
            # repeat for validation metrics
            for metric_name in metric_names:
                # first display all training metrics & then the cross-val metrics
                if batch_metrics:
                    # batch metrics (pick last value)
                    metric_val = \
                        self.metrics_history[f"val_{metric_name}"][
                            "batch_vals"][-1]
                else:
                    # epoch metrics
                    metric_val = \
                        self.metrics_history[f"val_{metric_name}"][
                            "epoch_vals"][-1]
                metrics_str += f"val_{metric_name}: {metric_val:.4f} - "
        # trim ending " - "
        if metrics_str.endswith(" - "):
            metrics_str = metrics_str[:-3]
        return metrics_str

    def plot_metrics(self, title = None, fig_size = None):
        """ plots epoch metrics values across epochs to show how
            training progresses
        """
        metric_names = self.tracked_metrics()
        metric_vals = {
            metric_name: self.metrics_history[metric_name]["epoch_vals"]
            for metric_name in metric_names
        }
        if self.include_val_metrics:
            # also has validation metrics
            for metric_name in metric_names:
                metric_vals[f"val_{metric_name}"] = \
                    self.metrics_history[f"val_{metric_name}"]["epoch_vals"]

        # we will plot a max of 3 metrics per row
        MAX_COL_COUNT = 3
        col_count = MAX_COL_COUNT if len(metric_names) > MAX_COL_COUNT \
            else len(metric_names)
        row_count = len(metric_names) // MAX_COL_COUNT
        row_count += 1 if len(metric_names) % MAX_COL_COUNT != 0 else 0
        # we'll always have "loss" metric in list, so safest to pick!
        x_vals = np.arange(1, len(metric_vals["loss"]) + 1)

        with sns.axes_style("darkgrid"):
            sns.set_context("notebook")  # , font_scale = 1.2)
            sns.set_style(
                {
                    "font.sans-serif": ["Segoe UI", "Calibri", "SF Pro Display", "Arial",
                                        "DejaVu Sans", "Sans"]
                }
            )
            fig_size = (16, 5) if fig_size is None else fig_size

            f, ax = plt.subplots(row_count, col_count, figsize = fig_size)
            for r in range(row_count):
                for c in range(col_count):
                    index = r * (col_count - 1) + c
                    if index < len(metric_names):
                        metric_name = metric_names[index]
                        if row_count == 1:
                            ax[c].plot(
                                x_vals, metric_vals[metric_name], lw = 2, markersize = 7
                            )
                        else:
                            ax[r, c].plot(
                                x_vals, metric_vals[metric_name], lw = 2, markersize = 7
                            )
                        if self.include_val_metrics:
                            if row_count == 1:
                                ax[c].plot(
                                    x_vals, metric_vals[f"val_{metric_name}"], lw = 2,
                                    markersize = 7
                                )
                            else:
                                ax[r, c].plot(
                                    x_vals, metric_vals[f"val_{metric_name}"], lw = 2,
                                    markersize = 7
                                )
                        legend = ["train", "valid"] if self.include_val_metrics else ["train"]
                        title = f"Training & Cross-validation \'{metric_name}\' vs Epochs" \
                            if len(legend) == 2 else f"Training \'{metric_name}\' vs Epochs"
                        if row_count == 1:
                            ax[c].legend(legend, loc = "best")
                            ax[c].set_title(title)
                        else:
                            ax[r, c].legend(legend, loc = "best")
                            ax[r, c].set_title(title)

        if title is not None:
            plt.suptitle(title)


# ----------------------------------------------------------------------------------
# training related functions & classes
# ----------------------------------------------------------------------------------
# custom data types
from typing import Union, Dict, Tuple
from collections.abc import Callable
import torchmetrics

# LossFxnType = Callable[[torch.tensor, torch.tensor], torch.tensor]
LRSchedulerType = torch.optim.lr_scheduler._LRScheduler
ReduceLROnPlateauType = torch.optim.lr_scheduler.ReduceLROnPlateau
NumpyArrayTuple = Tuple[np.ndarray, np.ndarray]
MetricsMapType = Dict[str, torchmetrics.Metric]


def cross_train_model(
    model: nn.Module,
    dataset: Union[NumpyArrayTuple, torch.utils.data.Dataset],
    loss_fxn,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    validation_split: float = 0.0,
    validation_dataset: Union[NumpyArrayTuple, torch.utils.data.Dataset] = None,
    metrics_map: MetricsMapType = None,
    epochs: int = 5,
    batch_size: int = 64,
    reporting_interval: int = 1,
    lr_scheduler: Union[LRSchedulerType, ReduceLROnPlateauType] = None,
    shuffle: bool = True,
    num_workers: int = 0
) -> MetricsHistory:
    """
        Cross-trains model (derived from nn.Module) across epochs using specified loss function,
        optimizer, validation dataset (if any), learning rate scheduler, epochs and batch size etc.
        @params:
            - model: the model being trained (instance of nn.Module)
            - dataset: the training dataset (subclass of torch.data.utils.Dataset)
            - loss_fxn: loss function used to calculate loss for each batch of data
                from the 'dataset' (instance of one of the loss functions available in Pytorch)
            - optimizer: optimizer used to optimize the weights of the model.
                One of the optimizers available in torch.nn.optim package
            - validation_

    """
    # validate parameters passed into function
    # assert isinstance(model, nn.Module), \
    #     "cross_train_model: 'model' parameter must be an instance of nn.Module!"
    # assert isinstance(dataset, torch.utils.data.Dataset), \
    #     "cross_train_model: 'dataset' must be a subclass of torch.utils.data.Dataset"
    assert (0.0 <= validation_split < 1.0), \
        "cross_train_model: 'validation_split' must be a float between (0.0, 1.0]"
    # if validation_dataset is not None:
    #     assert isinstance(validation_dataset, torch.utils.data.Dataset), \
    #         "cross_train_model: 'validation_dataset' must be a subclass of torch.utils.data.Dataset"
    if loss_fxn is None:
        raise ValueError("cross_train_model: 'loss_fxn' cannot be None")
    if optimizer is None:
        raise ValueError("cross_train_model: 'optimizer' cannot be None")
    # if lr_scheduler is not None:
    #     # NOTE:  ReduceLROnPlateau is NOT derived from _LRScheduler, but from object, which
    #     # is odd as all other schedulers derive from _LRScheduler
    #     assert (isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler) \
    #             or isinstance(
    #             lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
    #         )), \
    #         "lr_scheduler: incorrect type. Expecting class derived from torch.optim._LRScheduler or " \
    #         "ReduceLROnPlateau"

    reporting_interval = 1 if reporting_interval < 1 else reporting_interval
    reporting_interval = 1 if reporting_interval >= epochs else reporting_interval

    train_dataset, val_dataset = dataset, validation_dataset

    if isinstance(train_dataset, tuple):
        # train dataset was a tuple of np.ndarrays - convert to Dataset
        torch_X_train = torch.from_numpy(train_dataset[0]).type(torch.FloatTensor)
        torch_y_train = torch.from_numpy(train_dataset[1]).type(torch.FloatTensor)
        train_dataset = torch.utils.data.TensorDataset(torch_X_train, torch_y_train)

    if (val_dataset is not None) and isinstance(val_dataset, tuple):
        # cross-val dataset was a tuple of np.ndarrays - convert to Dataset
        torch_X_val = torch.from_numpy(val_dataset[0]).type(torch.FloatTensor)
        torch_y_val = torch.from_numpy(val_dataset[1]).type(torch.FloatTensor)
        val_dataset = torch.utils.data.TensorDataset(torch_X_val, torch_y_val)

    # split the dataset if validation_split > 0.0
    if (validation_split > 0.0) and (validation_dataset is None):
        # NOTE: validation_dataset supersedes validation_split, use
        # validation_split only if validation_dataset is None
        train_dataset, val_dataset = \
            split_dataset(train_dataset, validation_split)

    if val_dataset is not None:
        print(
            f"Cross training on \'{device}\' with {len(train_dataset)} training and " +
            f"{len(val_dataset)} cross-validation records...", flush = True
        )
    else:
        print(
            f"Training on \'{device}\' with {len(train_dataset)} records...",
            flush = True
        )

    if reporting_interval != 1:
        print(
            f"NOTE: progress will be reported every {reporting_interval} epoch!"
        )

    history = None

    try:
        model = model.to(device)
        tot_samples = len(train_dataset)
        len_num_epochs, len_tot_samples = len(str(epochs)), len(str(tot_samples))
        # create metrics history
        history = MetricsHistory(metrics_map, (val_dataset is not None))
        train_batch_size = batch_size if batch_size != -1 else len(train_dataset)

        for epoch in range(epochs):
            model.train()
            # reset metrics
            history.clear_batch_metrics()
            # loop over records in training dataset (use DataLoader)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size = train_batch_size,
                shuffle = shuffle,
                num_workers = num_workers
            )
            num_batches, samples = 0, 0

            for batch_no, (X, y) in enumerate(train_dataloader):
                X = X.to(device)
                y = y.to(device)
                # clear accumulated gradients
                optimizer.zero_grad()
                # make forward pass
                preds = model(X)
                # calculate loss
                loss_tensor = loss_fxn(preds, y)
                # compute gradients
                loss_tensor.backward()
                # update weights
                optimizer.step()

                # compute batch metric(s)
                preds = preds.to(device)
                history.calculate_batch_metrics(
                    preds.to("cpu"), y.to("cpu"), loss_tensor.item(),
                    val_metrics = False
                )

                num_batches += 1
                samples += len(X)

                if reporting_interval == 1:
                    # display progress with batch metrics - will display line like this:
                    # Epoch (  3/100): (  45/1024) -> loss: 3.456 - acc: 0.275
                    metricsStr = history.get_metrics_str(
                        batch_metrics = True,
                        include_val_metrics = False
                    )
                    print(
                        "\rEpoch (%*d/%*d): (%*d/%*d) -> %s" %
                        (len_num_epochs, epoch + 1, len_num_epochs, epochs,
                         len_tot_samples, samples, len_tot_samples, tot_samples,
                         metricsStr), end = '', flush = True
                    )
            else:
                # all train batches are over - display average train metrics
                history.calculate_epoch_metrics(val_metrics = False)
                if val_dataset is None:
                    if (epoch == 0) or ((epoch + 1) % reporting_interval == 0) \
                        or ((epoch + 1) == epochs):
                        metricsStr = history.get_metrics_str(
                            batch_metrics = False,
                            include_val_metrics = False
                        )
                        print(
                            "\rEpoch (%*d/%*d): (%*d/%*d) -> %s" %
                            (len_num_epochs, epoch + 1, len_num_epochs, epochs,
                             len_tot_samples, samples, len_tot_samples,
                             tot_samples,
                             metricsStr), flush = True
                        )
                        # training ends here as there is no cross-validation dataset
                else:
                    # we have a validation dataset
                    # same print as above except for trailing ... and end=''
                    if (epoch == 0) or ((epoch + 1) % reporting_interval == 0) \
                        or ((epoch + 1) == epochs):
                        metricsStr = history.get_metrics_str(
                            batch_metrics = False,
                            include_val_metrics = False
                        )
                        print(
                            "\rEpoch (%*d/%*d): (%*d/%*d) -> %s..." %
                            (len_num_epochs, epoch + 1, len_num_epochs, epochs,
                             len_tot_samples, samples, len_tot_samples,
                             tot_samples,
                             metricsStr),
                            end = '', flush = True
                        )

                    val_batch_size = batch_size if batch_size != -1 else len(val_dataset)
                    model.eval()
                    with torch.no_grad():
                        # val_dataloader = None if val_dataset is None else \
                        val_dataloader = torch.utils.data.DataLoader(
                            val_dataset,
                            batch_size = val_batch_size,
                            shuffle = shuffle,
                            num_workers = num_workers
                        )
                        num_val_batches = 0

                        for val_X, val_y in val_dataloader:
                            val_X = val_X.to(device)
                            val_y = val_y.to(device)
                            val_preds = model(val_X)
                            val_batch_loss = loss_fxn(val_preds, val_y).item()
                            history.calculate_batch_metrics(
                                val_preds.to("cpu"), val_y.to("cpu"), val_batch_loss,
                                val_metrics = True
                            )
                            num_val_batches += 1
                        else:
                            # loop over val_dataset completed - compute val average metrics
                            history.calculate_epoch_metrics(val_metrics = True)
                            # display final metrics
                            if (epoch == 0) or ((epoch + 1) % reporting_interval == 0) \
                                or ((epoch + 1) == epochs):
                                metricsStr = history.get_metrics_str(
                                    batch_metrics = False,
                                    include_val_metrics = True
                                )
                                print(
                                    "\rEpoch (%*d/%*d): (%*d/%*d) -> %s" %
                                    (len_num_epochs, epoch + 1, len_num_epochs,
                                     epochs,
                                     len_tot_samples, samples, len_tot_samples,
                                     tot_samples,
                                     metricsStr), flush = True
                                )

            # step the learning rate scheduler at end of epoch
            if (lr_scheduler is not None) and (epoch < epochs - 1):
                # have to go to these hoops as ReduceLROnPlateau requires a metric for step()
                if isinstance(
                    lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    # lr_metric = cum_metrics['val_loss'] if validation_dataset is not None \
                    #     else cum_metrics['loss']
                    lr_metric = history.metrics_history["loss"]["epoch_vals"][
                        -1] \
                        if val_dataset is not None \
                        else history.metrics_history["loss"]["epoch_vals"][-1]
                    lr_scheduler.step(lr_metric)
                else:
                    lr_scheduler.step()
        return history
    finally:
        model = model.to('cpu')


def evaluate_model(
    model: nn.Module,
    dataset: Union[NumpyArrayTuple, torch.utils.data.Dataset],
    loss_fn,
    device: torch.device,
    metrics_map: MetricsMapType = None,
    batch_size: int = 64
):
    try:
        model = model.to(device)

        # if dataset is a tuple of np.ndarrays, convert to torch Dataset
        if isinstance(dataset, tuple):
            X = torch.from_numpy(dataset[0]).type(torch.FloatTensor)
            y = torch.from_numpy(dataset[1]).type(torch.FloatTensor)
            dataset = torch.utils.data.TensorDataset(X, y)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size = batch_size,
            shuffle = False
        )

        tot_samples, samples, num_batches = len(dataset), 0, 0
        len_tot_samples = len(str(tot_samples))

        # create metrics history
        history = MetricsHistory(metrics_map)

        with torch.no_grad():
            model.eval()
            for X, y in loader:
                X = X.to(device)
                y = y.to(device)

                # forward pass
                preds = model(X)
                # compute batch loss
                batch_loss = loss_fn(preds, y).item()
                history.calculate_batch_metrics(
                    preds.to("cpu"), y.to("cpu"), batch_loss,
                    val_metrics = False
                )
                samples += len(y)
                num_batches += 1
                metricsStr = history.get_metrics_str(
                    batch_metrics = True,
                    include_val_metrics = False
                )
                print(
                    "\rEvaluating (%*d/%*d) -> %s" %
                    (len_tot_samples, samples, len_tot_samples, tot_samples,
                     metricsStr), end = '', flush = True
                )
            else:
                # iteration over batch completed
                # calculate average metrics across all batches
                history.calculate_epoch_metrics(val_metrics = False)
                metricsStr = history.get_metrics_str(
                    batch_metrics = False,
                    include_val_metrics = False
                )
                print(
                    "\rEvaluating (%*d/%*d) -> %s" %
                    (len_tot_samples, samples, len_tot_samples, tot_samples,
                     metricsStr), flush = True
                )
        return history.get_metric_vals(history.tracked_metrics())
    finally:
        model = model.to('cpu')


def predict_dataset(
    model: nn.Module,
    dataset: Union[NumpyArrayTuple, torch.utils.data.Dataset],
    device: torch.device,
    batch_size: int = 64
) -> NumpyArrayTuple:
    try:
        model = model.to(device)

        # if dataset is a tuple of np.ndarrays, convert to torch Dataset
        if isinstance(dataset, tuple):
            X = torch.from_numpy(dataset[0]).type(torch.FloatTensor)
            y = torch.from_numpy(dataset[1]).type(torch.FloatTensor)
            dataset = torch.utils.data.TensorDataset(X, y)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size = batch_size,
            shuffle = False
        )
        preds, actuals = [], []

        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            with torch.no_grad():
                model.eval()
                batch_preds = list(model(X).to("cpu").numpy())
                batch_actuals = list(y.to("cpu").numpy())
                preds.extend(batch_preds)
                actuals.extend(batch_actuals)
        return (np.array(preds), np.array(actuals))
    finally:
        model = model.to('cpu')


def predict(
    model: nn.Module,
    data: np.ndarray,
    device: torch.device,
    batch_size: int = 64
) -> np.ndarray:
    """
        runs predictions on Numpy Array (use for classification ONLY)
        @params:
            - model: instance of model derived from nn.Module (or instance of pyt.PytModel or pyt.PytSequential)
            - data: Numpy array of values on which predictions should be run
        @returns:
            - Numpy array of class predictions (probabilities)
            NOTE: to convert to classes use np.max(...,axis=1) after this call.
    """
    try:
        assert isinstance(model, nn.Module), \
            "predict() works with instances of nn.Module only!"
        assert ((isinstance(data, np.ndarray)) or (isinstance(data, torch.Tensor))), \
            "data must be an instance of Numpy ndarray or torch.tensor"
        # train on GPU if you can
        model = model.to(device)

        # run prediction
        with torch.no_grad():
            model.eval()
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype = torch.float32)
            data = data.to(device)
            # forward pass
            logits = model(data)
            preds = np.array(logits.cpu().numpy())
        return preds
    finally:
        model = model.cpu()


def save_model(model: nn.Module, model_save_path: str, verbose: bool = True):
    """ saves Pytorch state (state_dict) to disk
        @params:
            - model: instance of model derived from nn.Module (or instance of pytk.PytModel or pytk.PytSequential)
            - model_save_path: absolute or relative path where model's state-dict should be saved
              (NOTE:
                 - the model_save_path file is overwritten at destination without warning
                 - if `model_save_path` is just a file name, then model saved to current dir
                 - if `model_save_path` contains directory that does not exist, the function attempts to create
                   the directories
              )
    """
    save_dir, _ = os.path.split(model_save_path)

    if not os.path.exists(save_dir):
        # create directory from file_path, if it does not exist
        # e.g. if model_save_path = '/home/user_name/pytorch/model_states/model.pyt' and the
        # directory '/home/user_name/pytorch/model_states' does not exist, it is created
        try:
            os.mkdir(save_dir)
        except OSError as err:
            print(
                f"Unable to create folder/directory {save_dir} to save model!"
            )
            raise err

    # now save the model to file_path
    torch.save(model.state_dict(), model_save_path)
    if verbose:
        print(f"Pytorch model saved to {model_save_path}")


def load_model(model: nn.Module, model_state_dict_path: str, verbose: bool = True):
    """ loads model's state dict from file on disk
        @params:
            - model: instance of model derived from nn.Module (or instance of pytk.PytModel or pytk.PytSequential)
            - model_state_dict_path: complete/relative path from where model's state dict should be loaded. \
                This should be a valid path (i.e. should exist), else an IOError is raised.
    """

    # convert model_state_dict_path to absolute path
    model_save_path = pathlib.Path(model_state_dict_path).absolute()
    if not os.path.exists(model_save_path):
        raise IOError(
            f"ERROR: can't load model from {model_state_dict_path} - file does not exist!"
        )

    # load state dict from path
    state_dict = torch.load(model_save_path)
    model.load_state_dict(state_dict)
    if verbose:
        print(f"Pytorch model loaded from {model_state_dict_path}")
    model.eval()
    return model


class Trainer:
    def __init__(
        self,
        loss_fn,
        device: torch.device,
        metrics_map: MetricsMapType = None,
        epochs: int = 5, batch_size: int = 64, reporting_interval: int = 1,
        shuffle: bool = True,
        num_workers: int = 0
    ):
        if loss_fn is None:
            raise ValueError("FATAL ERROR: Trainer() -> 'loss_fn' cannot be None")
        if device is None:
            raise ValueError("FATAL ERROR: Trainer() -> 'device' cannot be None")
        if epochs < 1:
            raise ValueError("FATAL ERROR: Trainer() -> 'epochs' >= 1")
        # batch_size can be -ve
        batch_size = -1 if batch_size < 0 else batch_size
        reporting_interval = 1 if reporting_interval < 1 else reporting_interval
        assert num_workers >= 0, \
            "FATAL ERROR: Trainer() -> 'num_workers' must be >= 0"

        self.loss_fn = loss_fn
        self.device = device
        self.metrics_map = metrics_map
        self.epochs = epochs
        self.batch_size = batch_size
        self.reporting_interval = reporting_interval
        self.shuffle = shuffle
        self.num_workers = num_workers

    def fit(
        self, model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataset: Union[NumpyArrayTuple, torch.utils.data.Dataset],
        validation_dataset: Union[NumpyArrayTuple, torch.utils.data.Dataset] = None,
        validation_split: float = 0.0,
        lr_scheduler: Union[LRSchedulerType, ReduceLROnPlateauType] = None
    ) -> MetricsHistory:
        assert model is not None, \
            "FATAL ERROR: Trainer.fit() -> 'model' cannot be None"
        assert optimizer is not None, \
            "FATAL ERROR: Trainer.fit() -> 'optimizer' cannot be None"
        assert train_dataset is not None, \
            "FATAL ERROR: Trainer.fit() -> 'train_dataset' cannot be None"
        if lr_scheduler is not None:
            # NOTE:  ReduceLROnPlateau is NOT derived from _LRScheduler, but from object, which
            # is odd as all other schedulers derive from _LRScheduler
            assert (isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler) or \
                    isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)), \
                "lr_scheduler: incorrect type. Expecting class derived from " \
                "torch.optim._LRScheduler or ReduceLROnPlateau"

        history = cross_train_model(
            model, train_dataset, self.loss_fn, optimizer, device = self.device,
            validation_split = validation_split, validation_dataset = validation_dataset,
            metrics_map = self.metrics_map, epochs = self.epochs, batch_size = self.batch_size,
            reporting_interval = self.reporting_interval, lr_scheduler = lr_scheduler,
            shuffle = self.shuffle, num_workers = self.num_workers
        )
        return history

    def evaluate(
        self,
        model: nn.Module,
        dataset: Union[NumpyArrayTuple, torch.utils.data.Dataset]
    ) -> dict:
        return evaluate_model(
            model, dataset, self.loss_fn, device = self.device, metrics_map = self.metrics_map,
            batch_size = self.batch_size
        )

    def predict_dataset(self, model: nn.Module, dataset: torch.utils.data.Dataset) -> tuple:
        return predict_dataset(model, dataset, self.device, self.batch_size)

    def predict(self, model: nn.Module, data: np.ndarray) -> np.ndarray:
        return predict(model, data, self.device, self.batch_size)
