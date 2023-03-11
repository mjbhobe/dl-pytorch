# -*- coding: utf-8 -*-
"""
    pytorch_toolkit.py: 
        Functions, Classes and Metrics to ease the process of training, evaluating and testing Pytorch models.
        This module provides a Keras-like API to preclude the need to write boiler-plate code when
        training your Pytorch models. Convenience functions and classes that wrap those functions
        have been provided to ease the process of training, evaluating & testing your models.
        NOTE: these are utility classes/functions to ease the process to training/evaluating & testing ONLY!

    @author: Manish Bhobe
    This code is shared with MIT license as-is. Feel free to use it, extend it, but please give me 
    some credit as the original author of this code :) & don't hold me responsible if your project blows up!! ;)

    Usage:
    - Copy this file into a directory in sys.path
    - import the file into your code - I use this syntax
        import pytorch_toolkit as pytk
"""
import warnings

warnings.filterwarnings('ignore')

import sys

if sys.version_info < (3,):
    raise Exception(
        "pytorch_toolkit does not support Python 2. Please use a Python 3+ interpreter!"
    )

import os
import random
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import roc_auc_score

# torch imports
import torch
import torch.nn as nn
import torchmetrics
from torchsummary import summary
from torch.utils.data.dataset import Dataset

PYTK_FAV_SEED = 41

__v0rsion__ = "1.5.0"
__author__ = "Manish Bhobe"


def seed_all(seed = None):
    # to ensure that you get consistent results across runs & machines
    # @see: https://discuss.pytorch.org/t/reproducibility-over-different-machines/63047
    """seed all random number generators to get consistent results
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


# -----------------------------------------------------------------------------
# helper function to create various layers of model
# -----------------------------------------------------------------------------


def Conv2d(
    in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1,
    dilation = 1, groups = 1, bias = True, padding_mode = 'zeros'
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
        groups = groups, bias = bias, padding_mode = padding_mode
    )
    # @see: https://msdn.microsoft.com/en-us/magazine/mt833293.aspx for example
    torch.nn.init.xavier_uniform_(layer.weight)
    if bias:
        torch.nn.init.zeros_(layer.bias)
    return layer


def Linear(in_nodes, out_nodes, bias = True):
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
    layer = nn.Linear(in_nodes, out_nodes, bias)
    # @see: https://msdn.microsoft.com/en-us/magazine/mt833293.aspx for example
    torch.nn.init.xavier_uniform_(layer.weight)
    if bias:
        torch.nn.init.zeros_(layer.bias)
    return layer


def Dense(in_nodes, out_nodes, bias = True):
    """
        another shortcut for Linear(in_nodes, out_nodes)
    """
    return Linear(in_nodes, out_nodes, bias)


def Flatten(x):
    """
        (convenience function)
        Flattens out the previous layer. Normally used between Conv2D/MaxPooling2D or LSTM layers
        and Linear/Dense layers
    """
    return x.view(x.shape[0], -1)


def getConv2dFlattenShape(image_height, image_width, conv2d_layer, pool = 2):
    kernel_size = conv2d_layer.kernel_size
    padding = conv2d_layer.padding
    stride = conv2d_layer.stride
    dilation = conv2d_layer.dilation

    # calculate the output shape for Flatten layer
    # Andrew Ng's formula without dilation -> out = (f + 2p - (k-1))/s) + 1
    # with dilation, out = ((f + 2p - d * ((k-1) - 1)) / s)  + 1
    out_height = np.floor(
        (image_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[
            0] + 1
    )
    out_width = np.floor(
        (image_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[
            1] + 1
    )
    if pool > 0:
        out_height /= pool
        out_width /= pool
    return int(out_height), int(out_width)


# --------------------------------------------------------------------------------------
# Metrics used during training of model
# In this section, I provide code for typical metrics used during training
# NOTE: In this version, there is no provision to add your own metric!
# --------------------------------------------------------------------------------------


def epsilon():
    return torch.tensor(1e-7)


def accuracy(logits, labels):
    """
        computes accuracy given logits (computed probabilities) & labels (actual values)
        @params:
            - logits: predictions computed from call to model.forward(...) (Tensor)
            - labels: actual values (labels) (Tensor)
        @returns:
            computed accuracy value
            accuracy = (correct_predictions) / len(labels)
    """
    if logits.size()[1] == 1:
        # binary classification case (just 2 classes)
        # predict any value > 0.5 -> 1 else 0

        # predicted = torch.round(logits.data).reshape(-1)
        predicted = logits.ge(0.5).view(-1)
    else:
        # for multi-class classification, get index of max value
        vals, predicted = torch.max(logits.data, 1)

    total_count = labels.size(0)

    y_pred = predicted.long()
    if len(labels.shape) > 1:
        y_true = labels.reshape(-1).long()  # flatten
    else:
        y_true = labels.long()

    correct_predictions = (y_pred == y_true).sum().item()
    # accuracy is the fraction of correct predictions to total_count
    acc = (correct_predictions / total_count)
    return acc


def accuracy_new(logits, labels):
    acc = torchmetrics.functional.accuracy(logits, labels).item()
    return acc


def precision(logits, labels):
    """
        computes precision given logits (computed probabilities) & labels (actual values)
            - logits: predictions computed from call to model.forward(...) (Tensor)
            - labels: actual values (labels) (Tensor)
        @returns:   
            precision = true_positives / (predicted_positives + epsilon)
            where predicted positives = true positive + false positives
        A value close to 1 indicates that there are low incidences of false positives.
        ML algorithm should aim at getting values of precision closer to 1
    """
    if logits.size()[1] == 1:
        # binary classification case (just 2 classes)
        # predict any value > 0.5 as 1 else 0

        # predicted = torch.round(logits.data).reshape(-1)
        predicted = logits.ge(0.5).view(-1)
    else:
        vals, predicted = torch.max(logits.data, 1)

    y_pred = predicted.long()
    if len(labels.shape) > 1:
        y_true = labels.reshape(-1).long()  # flatten
    else:
        y_true = labels.long()

    # y_true * y_pred - element-wise multiplication, and will result in 1 only
    # when y_true == y_pred == 1 - all other cases will give 0
    true_positives = torch.sum(torch.clamp(y_true * y_pred, 0, 1))
    predicted_positives = torch.sum(torch.clamp(y_pred, 0, 1))

    prec = true_positives / (predicted_positives + epsilon())
    return prec.detach().numpy()


def precision_new(logits, labels):
    precision_score = torchmetrics.functional.precision(logits, labels).item()
    return precision_score


def recall(logits, labels):
    """
        computes recall (a.k.a. sensitivity) given logits (computed probabilities) & labels (actual values)
            - logits: predictions computed from call to model.forward(...) (Tensor)
            - labels: actual values (labels) (Tensor)
        @returns:
            recall = true_positives / (all_positives + epsilon)
            where all positives = true positives + false negatives
        A value close to 1 indicates that there are low incidences of false negatives.
        ML algorithm should aim at getting values closer to 1
    """
    if logits.size()[1] == 1:
        # binary classification case (just 2 classes)
        # predicted = torch.round(logits.data).reshape(-1)
        # any value > 0.5 -> 1 else 0
        predicted = logits.ge(0.5).view(-1)
    else:
        vals, predicted = torch.max(logits.data, 1)

    y_pred = predicted.long()
    if len(labels.shape) > 1:
        y_true = labels.reshape(-1).long()  # flatten
    else:
        y_true = labels.long()

    true_positives = torch.sum(torch.clamp(y_true * y_pred, 0, 1))
    all_positives = torch.sum(torch.clamp(y_true, 0, 1))

    rec = true_positives / (all_positives + epsilon())
    return rec.detach().numpy()


def recall_new(logits, labels):
    recall_score = torchmetrics.functional.recall(logits, labels).item()
    return recall_score


def f1_score2(logits, labels):
    """
        computes F1 score (for binary classification)
            - logits: predictions computed from call to model.forward(...) (Tensor)
            - labels: actual values (labels) (Tensor)
        @returns:
            f1 = 2*((precision*recall)/(precision+recall))
            F1 score is harmonic mean of precision & recall
    """

    prec = precision(logits, labels)
    rec = recall(logits, labels)
    f1 = 2 * ((prec * rec) / (prec + rec + epsilon()))
    return f1.detach().numpy()


def f1_score2_new(logits, labels):
    prec = precision_new(logits, labels)
    rec = recall_new(logits, labels)
    f1_score_val = 2 * ((prec * rec) / (prec + rec + epsilon()))
    return f1_score_val


def roc_auc(logits, labels):
    """
        computes roc_auc score (for BINARY classification)
            - logits: predictions computed from call to model.forward(...) (Tensor)
            - labels: actual values (labels) (Tensor)
        @returns:
            computed roc_auc score
    """
    _, predicted = torch.max(logits.data, 1)
    y_true = labels.detach().numpy()
    y_pred = predicted.detach().numpy()
    rcac = roc_auc_score(y_true, y_pred)
    return rcac  # .detach().numpy()


def roc_auc_new(logits, labels):
    roc_auc_score = torchmetrics.functional.auroc(logits, labels).item()
    return roc_auc_score


def mse(predictions, actuals):
    """
        computes mean-squared-error
            - predictions: predictions computed from call to model.forward(...) (Tensor)
            - actuals: actual values (labels) (Tensor)
        @returns:
            computed mse = sum((actuals - predictions)**2) / actuals.size(0)
    """
    diff = actuals - predictions
    mse_err = torch.sum(diff * diff) / (diff.numel() + epsilon())
    return mse_err.detach().numpy()


def mse_new(predictions, actuals):
    mse_score = torchmetrics.functional.mean_squared_error(predictions, actuals).item()
    return mse_score


def rmse(predictions, actuals):
    """
        computes root-mean-squared-error
            - predictions: predictions computed from call to model.forward(...) (Tensor)
            - actuals: actual values (labels) (Tensor)
        @returns:
            computed rmse = sqrt(mse(predictions, actuals))
    """
    rmse_err = torch.sqrt(torch.tensor(mse(predictions, actuals), dtype = torch.float32))
    return rmse_err.detach().numpy()


def rmse_new(predictions, actuals):
    rmse_score = torch.sqrt(
        torchmetrics.functional.mean_squared_error(predictions, actuals)
    ).item()
    return rmse_score


def mae(predictions, actuals):
    """
        computes mean absolute error
            - predictions: predictions computed from call to model.forward(...) (Tensor)
            - actuals: actual values (labels) (Tensor)
        @returns:
            computed mae = sum(abs(predictions - actuals)) / actuals.size(0)
    """
    diff = actuals - predictions
    mae_err = torch.mean(torch.abs(diff))
    return mae_err.detach().numpy()


def mae_new(predictions, actuals):
    mae_score = torchmetrics.functional.mean_absolute_error(predictions, actuals).item()
    return mae_score


def r2_score(predictions, actuals):
    """
        computes the r2_score
        @returns:
            computed r2_score
    """
    SS_res = torch.sum(torch.pow(actuals - predictions, 2))
    SS_tot = torch.sum(torch.pow(actuals - torch.mean(actuals), 2))
    return (1 - SS_res / (SS_tot + epsilon())).detach().numpy()


def r2_score_new(predictions, actuals):
    r2_score_val = torchmetrics.functional.r2_score(predictions, actuals).item()
    return r2_score_val


METRICS_MAP_OLD = {
    'acc': accuracy,
    'accuracy': accuracy,
    'prec': precision,
    'precision': precision,
    'rec': recall,
    'recall': recall,
    'sens': recall,
    'sensitivity': recall,
    'f1': f1_score2,  # f1_score2 to avoid conflict with scklern.metrics.f1_score
    'f1_score': f1_score2,
    'roc_auc': roc_auc,
    # regression metrics
    'mse': mse,
    'rmse': rmse,
    'mae': mae,
    'r2_score': r2_score
}

METRICS_MAP_NEW = {
    'acc': accuracy_new,
    'accuracy': accuracy_new,
    'prec': precision_new,
    'precision': precision_new,
    'rec': recall_new,
    'recall': recall_new,
    'sens': recall_new,
    'sensitivity': recall_new,
    'f1': f1_score2_new,  # f1_score2 to avoid conflict with scklern.metrics.f1_score
    'f1_score': f1_score2_new,
    'roc_auc': roc_auc_new,
    # regression metrics
    'mse': mse_new,
    'rmse': rmse_new,
    'mae': mae_new,
    'r2_score': r2_score_new
}

USE_OLD_METRICS_MAP = False
METRICS_MAP = METRICS_MAP_OLD if USE_OLD_METRICS_MAP else METRICS_MAP_NEW


# -------------------------------------------------------------------------------------
# helper class to implement early stopping (# based on Bjarten's implementation)
# (@see: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py)
# -------------------------------------------------------------------------------------

class EarlyStopping:
    """
        Early stops the training if monitored metric (usually validation loss) doesn't improve 
       after a given patience (or no of epochs).
    """

    def __init__(
        self,
        monitor = 'val_loss',
        min_delta = 0,
        patience = 5,
        mode = 'min',
        verbose = False,
        save_best_weights = True, checkpoint_file_path = None
    ):
        """
            Args:
                monitor (str): which metric should be monitored (default: 'val_loss')
                min_delta (float): Minimum change in the monitored quantity to qualify as an improvement. (default: 0)
                patience (int): How many epochs to wait until after last validation loss improvement. (default: 5)
                mode (str): one of {'min','max'} (default='min') In 'min' mode, training will stop when the quantity 
                    monitored has stopped decreasing; in 'max' mode it will stop when the quantity monitored has 
                    stopped increasing;
                verbose (bool): If True, prints a message for each validation loss improvement. (default: False)
                save_best_weights (bool): Save state with best weights so far (default: False)
                checkpoint_file_path (string, optional): directory to which the checkpoint file must be saved
                (optional, defaults to current directory)
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        if mode not in ['min', 'max']:
            warnings.warn(
                f'EarlyStopping - \'mode\' {mode} is unknown. Using \'min\' instead!'
            )
            self.mode = 'min'
        self.verbose = verbose
        self.save_best_weights = save_best_weights

        self.monitor_op = np.less if self.mode == 'min' else np.greater
        self.min_delta *= -1 if self.monitor_op == np.less else 1
        self.best_score = np.Inf if self.monitor_op == np.less else -np.Inf
        self.counter = 0
        self.best_epoch = 0

        if (checkpoint_file_path is None) or (not os.path.exists(checkpoint_file_path)):
            # create our own directory for logging -> ./chkplogs%Y%m%d%H%M%S
            now = datetime.now()
            dir_name = os.path.join(
                os.getcwd(),
                f"checkpoint_{now.strftime('%Y%m%d%H%M%S')}"
            )
            self.checkpoint_file_path = dir_name
        os.mkdir(self.checkpoint_file_path)
        assert os.path.exists(self.checkpoint_file_path), \
            "FATAL: could not create dir {self.checkpoint_file_path} for checkpoints"
        # self.checkpoint_file_path = None # os.path.join(checkpoint_file_path, 'checkpoint.pt')
        self.best_model_path = None
        self.metrics_log = []
        self.early_stop = False

    def __call__(self, model, curr_metric_val, epoch):
        if not (isinstance(model, PytkModule) or isinstance(model, PytkModuleWrapper)):
            raise TypeError("model should be derived from PytModule or PytkModuleWrapper")

        # self.is_wrapped = isinstance(model, PytkModuleWrapper)
        if self.monitor_op(curr_metric_val - self.min_delta, self.best_score):
            if self.save_best_weights:
                # save model state for restore later
                self.save_checkpoint(model, self.monitor, curr_metric_val)
            self.best_score = curr_metric_val
            self.counter = 0
            self.metrics_log = []
            self.best_epoch = epoch + 1
            if self.verbose:
                print(
                    f'   EarlyStopping (log): patience counter reset to 0 at epoch {epoch}' +
                    f'where best score of \'{self.monitor}\' is {self.best_score:.3f}'
                )
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f'   EarlyStopping (log): patience counter increased to {self.counter}' +
                    f' - best_score of \'{self.monitor}\' is {self.best_score:.3f} at' +
                    f' epoch {self.best_epoch}'
                )
            if self.counter >= self.patience:
                self.early_stop = True
                print(
                    '   EarlyStopping: Early stopping training at epoch %d. \'%s\' has not improved for past %d '
                    'epochs.' % (epoch, self.monitor, self.patience)
                )
                print(
                    '     - Best score: %.4f at epoch %d. Last %d scores -> %s' % (
                        self.best_score, self.best_epoch, len(self.metrics_log),
                        self.metrics_log)
                )
            else:
                self.metrics_log.append(curr_metric_val)

    def save_checkpoint(self, model, metric_name, curr_metric_val):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                '   EarlyStopping (log): \'%s\' metric has \'improved\' - from %.4f to %.4f. Saving checkpoint...' % (
                    metric_name, self.best_score, curr_metric_val)
            )
        best_model_file_path = os.path.join(
            self.checkpoint_file_path,
            f"checkpoint{datetime.now().strftime('%Y%m%d%H%M%S')}.pt"
        )
        model.save(best_model_file_path, verbose = 0)
        self.best_model_path = best_model_file_path
        # mod = model
        # if isinstance(model, PytkModuleWrapper):
        #     mod = model.model
        # # torch.save(mod.state_dict(), self.checkpoint_file_path)
        # mod.save(self.checkpoint_file_path)


class EarlyStopping_New:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience = 7, verbose = False, delta = 0, path = 'checkpoint.pt', trace_func = print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = pathlib.Path(os.getcwd()) / path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def checkpoint_path(self) -> str:
        return self.path

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# -------------------------------------------------------------------------------------
# helper functions for training model, evaluating performance & making predictions
# -------------------------------------------------------------------------------------


def check_attribs__(model, loss_fn, optimizer = None, check_only_loss = False):
    """ internal helper function - checks various attributes of "model" """
    if loss_fn is None:
        # model instance must have self.loss_fn attribute defined
        try:
            l = model.loss_fn
            if l is None:
                # defined in model, but set to None
                raise ValueError(
                    'FATAL ERROR: it appears that you have not set a value for loss_fn ' +
                    'Detected None value for both the loss_fn parameter and module.loss_fn attribute!'
                )
        except AttributeError as e:
            print(
                "FATAL ERROR: when loss_fn parameter is None, the model's instance is expected " +
                "to have the loss function defined with attribute self.loss_fn!\n" +
                "This model's instance does not have a self.loss_fn attribute defined."
            )
            raise e

    if not check_only_loss:
        if optimizer is None:
            # model instance must have self.optimizer attribute defined
            try:
                o = model.optimizer
                if o is None:
                    raise ValueError(
                        'FATAL ERROR: it appears that you have not set a value ' +
                        'for optimizer. Detected None value for both the optimizer parameter and ' +
                        'module.optimizer attribute!'
                    )
            except AttributeError as e:
                print(
                    "FATAL ERROR: when optimizer parameter is None, the model's instance is expected " +
                    "to have the optimizer function defined with attribute self.optimizer!\n" +
                    "This model's instance does not have a self.optimizer attribute defined."
                )
                raise e


# def compute_metrics__(logits, labels, metrics, batch_metrics, validation_dataset=False):
#     """ internal helper functions - computes metrics in an epoch loop """
#     for metric_name in metrics:
#         metric_value = METRICS_MAP[metric_name](logits, labels)
#         if validation_dataset:
#             # .append(metric_value)
#             batch_metrics['val_%s' % metric_name] = metric_value
#         else:
#             batch_metrics[metric_name] = metric_value


# def accumulate_metrics__(metrics, cum_metrics, batch_metrics, validation_dataset=False):
#     """ internal helper function - "sums" metrics across batches """
#     if metrics is not None:
#         for metric in metrics:
#             if validation_dataset:
#                 cum_metrics['val_%s' % metric] += \
#                     batch_metrics['val_%s' % metric]
#             else:
#                 cum_metrics[metric] += batch_metrics[metric]

#     # check for loss separately
#     if 'loss' not in metrics:
#         if validation_dataset:
#             cum_metrics['val_loss'] += batch_metrics['val_loss']
#         else:
#             cum_metrics['loss'] += batch_metrics['loss']
#     return cum_metrics


# def get_metrics_str__(metrics_list, batch_or_cum_metrics, validation_dataset=False):
#     """ internal helper functions: formats metrics for printing to console """
#     metrics_str = ''

#     for i, metric in enumerate(metrics_list):
#         if i > 0:
#             metrics_str += ' - %s: %.4f' % (
#                 metrics_list[i], batch_or_cum_metrics[metric])
#         else:
#             metrics_str += '%s: %.4f' % (metrics_list[i],
#                                          batch_or_cum_metrics[metric])

#     # append validation metrics too
#     if validation_dataset:
#         for i, metric in enumerate(metrics_list):
#             metrics_str += ' - val_%s: %.4f' % (
#                 metrics_list[i], batch_or_cum_metrics['val_%s' % metric])

#     return metrics_str


def get_lrates__(optimizer, format_str = '%.8f'):
    """given the optimizer, returns the current learning rates as a string
       (to be used to report metrics per epoch only!) """
    lr_rates_o = []
    for g in optimizer.param_groups:
        lr_rates_o.append(g['lr'])
    lr_rates_o = [format_str % lrr for lrr in lr_rates_o]
    return ' - lr: %s' % lr_rates_o


class MetricsHistory:
    def __init__(self, metrics_list = None, include_val_metrics = False):
        # epoch metrics
        self.history = {'loss': []}
        # batch metrics
        self.batch_metrics = {'loss': []}
        # list of metric names, excluding loss
        self.metrics_list = metrics_list

        if include_val_metrics:
            self.history['val_loss'] = []
            self.batch_metrics['val_loss'] = []

        if (metrics_list is not None) and (len(metrics_list) > 0):
            for metric_name in metrics_list:
                # NOTE: we do not have support for custom metrics yet!!
                if metric_name not in METRICS_MAP.keys():
                    raise ValueError(f"{metric_name} - unrecognized metric!")
                else:
                    self.history[metric_name] = []
                    self.batch_metrics[metric_name] = []

                    if include_val_metrics:
                        self.history[f'val_{metric_name}'] = []
                        self.batch_metrics[f'val_{metric_name}'] = []

    def has_metric(self, metric_name):
        """given name of metric (e.g. 'loss' or 'val_acc'), 
           checks if this metric is in the metrics list
        """
        metric_name = metric_name.lower()
        return metric_name in self.history.keys()

    def clear_batch_metrics(self):
        for metric_name in self.batch_metrics.keys():
            self.batch_metrics[metric_name].clear()

    def compute_batch_metrics(
        self, logits, labels, batch_loss,
        update_validation_metrics = False
    ):
        for metric_name in self.batch_metrics.keys():
            if update_validation_metrics:
                if (not metric_name.startswith('val_')):
                    continue  # ignore metrics not like val_XXX
                # NOTE: drop the 'val_' from metric name when call formula
                # to calculate value of val_XXX metric - 'val_XXX'[4:] does the trick!
                metric_val = batch_loss if metric_name.endswith('loss') \
                    else METRICS_MAP[metric_name[4:]](logits, labels)
                self.batch_metrics[metric_name].append(metric_val)
            else:
                # for training metrics
                if (metric_name.startswith('val_')):
                    continue  # ignore metrics like val_XXX
                metric_val = batch_loss if metric_name.endswith(
                    'loss'
                ) else METRICS_MAP[metric_name](logits, labels)
                self.batch_metrics[metric_name].append(metric_val)

    def accumulate(self, accum_validation_metrics = False):
        for metric_name in self.batch_metrics.keys():
            if accum_validation_metrics:
                if (not metric_name.startswith('val_')):
                    continue  # skip metrics not like val_XXX
                metric_mean = np.array(self.batch_metrics[metric_name]).mean()
            else:
                if metric_name.startswith('val_'):
                    continue  # skip metrics like val_XXX
                metric_mean = np.array(self.batch_metrics[metric_name]).mean()
            self.history[metric_name].append(metric_mean)
            self.batch_metrics[metric_name].clear()

    def get_batch_metrics_str(self, include_validation_metrics = False):
        metrics_str = ''

        metrics_str = f"loss: {np.array(self.batch_metrics['loss']).mean():.4f}"
        for metric_name in self.metrics_list:
            metrics_str += f" - {metric_name}: {np.array(self.batch_metrics[metric_name]).mean():.4f}"

        if include_validation_metrics:
            assert self.has_metric('val_loss'), \
                f"Error: 'val_loss' is not tracked in training loop"
            metrics_str += f" - val_loss: {np.array(self.batch_metrics['val_loss']).mean():.4f}"
            for metric_name in self.metrics_list:
                mname = f"val_{metric_name}"
                metrics_str += f" - {mname}: {np.array(self.batch_metrics[mname]).mean():.4f}"

        return metrics_str

    def get_metrics_str(self, include_validation_metrics = False):
        metrics_str = ''

        metrics_str = f"loss: {np.array(self.history['loss']).mean():.4f}"
        for metric_name in self.metrics_list:
            metrics_str += f" - {metric_name}: {np.array(self.history[metric_name]).mean():.4f}"

        if include_validation_metrics:
            assert self.has_metric('val_loss'), \
                f"Error: 'val_loss' is not tracked in training loop"
            metrics_str += f" - val_loss: {np.array(self.history['val_loss']).mean():.4f}"
            for metric_name in self.metrics_list:
                mname = f"val_{metric_name}"
                metrics_str += f" - {mname}: {np.array(self.history[mname]).mean():.4f}"

        return metrics_str


# def create_hist_and_metrics_ds__(metrics, include_val_metrics=True):
#     """ internal helper functions - create data structures to log epoch metrics,
#         batch metrics & cumulative betch metrics """
#     history = {'loss': []}
#     batch_metrics = {'loss': 0.0}
#     cum_metrics = {'loss': 0.0}

#     if include_val_metrics:
#         history['val_loss'] = []
#         batch_metrics['val_loss'] = 0.0
#         cum_metrics['val_loss'] = 0.0

#     if metrics is not None and len(metrics) > 0:
#         # walk list of metric names & create one entry per metric
#         for metric_name in metrics:
#             if metric_name not in METRICS_MAP.keys():
#                 raise ValueError('%s - unrecognized metric!' % metric_name)
#             else:
#                 history[metric_name] = []
#                 batch_metrics[metric_name] = 0.0
#                 cum_metrics[metric_name] = 0.0

#                 if include_val_metrics:
#                     history['val_%s' % metric_name] = []
#                     batch_metrics['val_%s' % metric_name] = 0.0
#                     cum_metrics['val_%s' % metric_name] = 0.0

#     return history, batch_metrics, cum_metrics


def train_model(
    model, train_dataset, collate_fn = None, loss_fn = None, optimizer = None,
    validation_split = 0.0,
    validation_dataset = None, lr_scheduler = None, epochs = 25,
    batch_size = 64,
    metrics = None, shuffle = True, num_workers = 0, early_stopping = None,
    verbose = 2, report_interval = 1
):
    """
        Trains model (derived from nn.Module) across epochs using specified loss function,
        optimizer, validation dataset (if any), learning rate scheduler, epochs and batch size etc.
        @parms:
            - model: instance of a model derived from torch.nn.Module class
            - train_dataset: training dataset derived from torchvision.dataset
            - loss_fn: loss function to use when training (optional, default=None)
                if loss_fn == None, then the model class must define an attribute self.loss_fn
                defined as an instance of a loss function
            - optimizer: Pytorch optimizer to use during training (instance of any optimizer
            from the torch.optim package)
                if optimizer == None, then model must define an attribute self.optimizer, which is 
                an instance of any optimizer defined in the torch.optim package
            - validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. 
            The model will set apart this fraction of the training data, will not train on it, and will 
            evaluate the loss and any model metrics on this data at the end of each epoch.
            - validation_dataset: cross-validation dataset to use during training (optional, default=None)
                if validation_dataset is not None, then model is cross trained on test dataset & 
                cross-validation dataset (good practice to always use a cross-validation dataset!)
                (NOTE: validation_dataset will supercede validation_split if both specified)
            - lr_scheduler: learning rate scheduler to use during training (optional, default=None)
                if specified, should be one of the learning rate schedulers (e.g. StepLR) defined
                in the torch.optim.lr_scheduler package
            - epochs (int): number of epochs for which the model should be trained (optional, default=25)
            - batch_size (int): batch size to split the training & cross-validation datasets during 
            training (optional, default=64). If batch_size = -1, entire dataset size is used as batch size.
            - metrics (list of strings): metrics to compute (optional, default=None)
                pass a list of strings from one or more of the following ['acc','prec','rec','f1']
                when metrics = None, only loss is computed for training set (and validation set, if any)
                when metrics not None, in addition to loss all specified metrics are computed for training
                set (and validation set, if specified)
            - shuffle (boolean, default = True): determines if the training dataset & validation dataset, if provided
                should be shuffled between epochs or not. It's a good practice to shuffle, unless absolutely not required
            - num_workers (int, default=0): number of worker threads to use when loading datasets internally using 
                DataLoader objects.
            - early_stopping(EarlyStopping, default=None): instance of EarlyStopping class to be passed in if training
                has to be early-stopped based on parameters used to construct instance of EarlyStopping
            - verbose (0, 1 or 2, default=0): sets the verbosity level for reporting progress during training
                verbose=2 - very verbose output, displays batch wise metrics
                verbose=1 - medium output, displays metrics at end of epoch, but shows incrimenting counter of batches
                verbose=0 - least verbose output, does NOT display any output until the training dataset (and validation
                dataset, if any) completes
            - report_interval (value >= 1 & < num_epochs): interval at which training progress gets updated (e.g. if
            report_interval=100,
                training progress is printed every 100th epoch.) Default = 1, meaning status reported at end of each epoch.
        @returns:
            - history: dictionary of the loss & accuracy metrics across epochs
                Metrics are saved by key name (see METRICS_MAP) 
                Metrics (across epochs) are accessed by key (e.g. hist['loss'] accesses training loss
                and hist['val_acc'] accesses validation accuracy
                Validation metrics are available ONLY if validation set is provided during training
    """
    try:
        # checks for parameters
        assert isinstance(model, nn.Module), \
            "train_model() works with instances of nn.Module only!"
        assert isinstance(train_dataset, torch.utils.data.Dataset), \
            "train_dataset must be a subclass of torch.utils.data.Dataset"
        assert (0.0 <= validation_split < 1.0), \
            "validation_split must be a float between (0.0, 1.0]"
        if validation_dataset is not None:
            assert isinstance(validation_dataset, torch.utils.data.Dataset), \
                "validation_dataset must be a subclass of torch.utils.data.Dataset"
        check_attribs__(model, loss_fn, optimizer)
        if loss_fn is None:
            loss_fn = model.loss_fn
        if loss_fn is None:
            # still not assigned??
            raise ValueError(
                "Loss function is not defined. Must be passed as a parameter or defined in class"
            )
        if optimizer is None:
            optimizer = model.optimizer
        if optimizer is None:
            # still not assigned??
            raise ValueError(
                "Optimizer is not defined. Must be passed as a parameter or defined in class"
            )
        if lr_scheduler is not None:
            # NOTE:  ReduceLROnPlateau is NOT derived from _LRScheduler, but from object, which
            # is odd as all other schedulers derive from _LRScheduler
            assert (isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler) or
                    isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)), \
                "lr_scheduler: incorrect type. Expecting class derived from torch.optim._LRScheduler or " \
                "ReduceLROnPlateau"
        early_stopping_metric = None
        if early_stopping is not None:
            assert isinstance(early_stopping, EarlyStopping), \
                "early_stopping: incorrect type. Expecting instance of EarlyStopping class"
            early_stopping_metric = early_stopping.monitor

        if verbose not in [0, 1, 2]:
            verbose = 2  # most verbose

        report_interval = 1 if report_interval < 1 else report_interval
        report_interval = 1 if report_interval >= epochs else report_interval

        # if validation_split is provided by user, then split train_dataset
        if (validation_split > 0.0) and (validation_dataset is None):
            # NOTE: validation_dataset supersedes validation_split, use validation_split only
            # if validation_dataset is None
            num_recs = len(train_dataset)
            train_count = int((1.0 - validation_split) * num_recs)
            val_count = num_recs - train_count
            train_dataset, validation_dataset = \
                torch.utils.data.random_split(train_dataset, [train_count, val_count])
            assert (train_dataset is not None) and (len(train_dataset) == train_count), \
                "Something is wrong with validation_split - getting incorrect train_dataset counts!!"
            assert (validation_dataset is not None) and (
                len(validation_dataset) == val_count), \
                "Something is wrong with validation_split - getting incorrect validation_dataset counts!!"

        # train on GPU if available
        gpu_available = torch.cuda.is_available()

        if verbose != 0:
            print('Training on %s...' % ('GPU' if gpu_available else 'CPU'))
        model = model.cuda() if gpu_available else model.cpu()

        if verbose != 0:
            if validation_dataset is not None:
                print(
                    'Training on %d samples, cross-validating on %d samples' %
                    (len(train_dataset), len(validation_dataset))
                )
            else:
                print('Training on %d samples' % len(train_dataset))

        if (verbose != 0) and (report_interval != 1):
            print(
                f"NOTE: training progress will be reported after every {report_interval} epochs"
            )

        tot_samples = len(train_dataset)
        len_tot_samples = len(str(tot_samples))

        # create data structures to hold batch metrics, epoch metrics etc.
        # history, batch_metrics, cum_metrics = \
        #     create_hist_and_metrics_ds__(metrics, validation_dataset is not None)

        # metrics_list = ['loss']
        # if metrics is not None:
        #     metrics_list = metrics_list + metrics
        #     if early_stopping_metric is not None:
        #         metrics_list_check = []
        #         for metric in metrics_list:
        #             metrics_list_check.append(metric)
        #             metrics_list_check.append('val_%s' % metric)
        #         assert early_stopping_metric in metrics_list_check, \
        #             "early stopping metric (%s) is not logged during training!" % early_stopping_metric

        # create data structures to hold batch metrics, epoch metrics etc.
        metrics_history = MetricsHistory(metrics, validation_dataset is not None)

        len_num_epochs = len(str(epochs))

        curr_lr = None

        # if batch_size == -1, then use entire training dataset as batch (not recommended
        # unless len(training_dataset) is reasonably small)
        train_batch_size = batch_size if batch_size != -1 else len(train_dataset)

        for epoch in range(epochs):
            model.train()  # 'flag model as training', so batch normalization & dropouts can be applied
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size = train_batch_size,
                collate_fn = collate_fn,
                shuffle = shuffle,
                num_workers = num_workers
            )
            num_batches = 0
            samples = 0

            # zero out batch & cum metrics for next epoch
            # for metric_name in metrics_list:
            #     batch_metrics[metric_name] = 0.0
            #     cum_metrics[metric_name] = 0.0
            #     if validation_dataset is not None:
            #         batch_metrics['val_%s' % metric_name] = 0.
            #         cum_metrics['val_%s' % metric_name] = 0.0

            # learning rates used by the optimizer (as a string)
            learning_rates = ''

            # iterate over the training dataset
            for batch_no, (data, labels) in enumerate(train_loader):
                # move to GPU if available
                data = data.cuda() if gpu_available else data.cpu()
                labels = labels.cuda() if gpu_available else labels.cpu()

                # clear accumulated gradients
                optimizer.zero_grad()
                # make a forward pass
                logits = model(data)
                # apply loss function
                loss_tensor = loss_fn(logits, labels)
                # compute gradients
                loss_tensor.backward()
                # update weights
                optimizer.step()
                batch_loss = loss_tensor.item()

                # compute metrics for batch + accumulate metrics across batches
                # batch_metrics['loss'] = batch_loss
                # if metrics is not None:
                #     # compute metrics for training dataset only!
                #     compute_metrics__(logits, labels, metrics,
                #                       batch_metrics, validation_dataset=False)
                # # same as cum_metrics[metric_name] += batch_metric[metric_name] across all metrics
                # cum_metrics = accumulate_metrics__(
                #     metrics_list, cum_metrics, batch_metrics, validation_dataset=False)

                # compute training metrics for this batch
                metrics_history.compute_batch_metrics(
                    logits, labels, batch_loss,
                    update_validation_metrics = False
                )

                samples += len(labels)
                num_batches += 1

                # display incremental progress
                learning_rates = get_lrates__(optimizer)

                # if verbose == 2:
                #     if (epoch == 0) or ((epoch + 1) % report_interval == 0):
                #         # verbose == 2 -> display progress counter + metrics after each batch
                #         # e.g: Epoch (2/20): (1024/5000) -> loss: 28.45 - acc: 0.4567
                #         # metrics_str = get_metrics_str__(
                #         #     metrics_list, batch_metrics, validation_dataset=False)
                #         metrics_str = metrics_history.get_batch_metrics_str(include_validation_metrics=False)
                #         metrics_str += learning_rates
                #         print('\rEpoch (%*d/%*d): (%*d/%*d) -> %s' %
                #               (len_num_epochs, epoch + 1, len_num_epochs, epochs,
                #                len_tot_samples, samples, len_tot_samples, tot_samples,
                #                metrics_str),
                #               end='', flush=True)
                # elif verbose == 1:
                #     if (epoch == 0) or ((epoch + 1) % report_interval == 0):
                #         # verbose == 1 -> display progress counter only, no metrics
                #         # e.g: Epoch (2/20): (1024/5000) -> ...
                #         print('\rEpoch (%*d/%*d): (%*d/%*d) -> ...' %
                #               (len_num_epochs, epoch + 1, len_num_epochs, epochs,
                #                len_tot_samples, samples, len_tot_samples, tot_samples),
                #               end='', flush=True)

                if (verbose in [1, 2] and (
                    (epoch == 0) or ((epoch + 1) % report_interval == 0))):
                    # report progress on epoch 0 and every report_interval thereafter if verbose in [1,2]
                    # fetch metrics only if verbose == 2, else set metrics = '...'
                    # if verbose == 0, there is no output whatsoever!
                    metrics_str = "..." if verbose == 1 else \
                        metrics_history.get_batch_metrics_str(
                            include_validation_metrics = False
                        ) + learning_rates
                    # if verbose == 1 -> display progress counter only, no metrics
                    #    e.g.: Epoch (2/20): (1024/5000) -> ...
                    # if verbose == 2 -> display progress counter + metrics after each batch
                    #    e.g.: Epoch (2/20): (1024/5000) -> loss: 28.45 - acc: 0.4567
                    print(
                        '\rEpoch (%*d/%*d): (%*d/%*d) -> %s' %
                        (len_num_epochs, epoch + 1, len_num_epochs, epochs,
                         len_tot_samples, samples, len_tot_samples, tot_samples,
                         metrics_str),
                        end = '', flush = True
                    )
            else:
                # all batches in train_loader dataset are complete...

                # compute average metrics across all batches of train_loader
                # for metric_name in metrics_list:
                #     cum_metrics[metric_name] = cum_metrics[metric_name] / num_batches
                #     history[metric_name].append(cum_metrics[metric_name])

                # compute average of training metrics across all batches of train_loader
                metrics_history.accumulate(accum_validation_metrics = False)

                # display average training metrics for this epoch if verbose = 1 or 2
                # learning_rates = get_lrates__(optimizer)
                if ((verbose in [1, 2]) or (validation_dataset is None)):
                    if (epoch == 0) or ((epoch + 1) % report_interval == 0):
                        # metrics_str = get_metrics_str__(
                        #     metrics_list, cum_metrics, validation_dataset=False)

                        metrics_str = metrics_history.get_metrics_str(
                            include_validation_metrics = False
                        )

                        metrics_str += learning_rates
                        print(
                            '\rEpoch (%*d/%*d): (%*d/%*d) -> %s ...' %
                            (len_num_epochs, epoch + 1, len_num_epochs, epochs,
                             len_tot_samples, samples, len_tot_samples, tot_samples,
                             metrics_str),
                            end = '' if validation_dataset is not None else '\n',
                            flush = True
                        )

                if validation_dataset is not None:
                    # perform validation over validation dataset
                    val_batch_size = batch_size if batch_size != -1 \
                        else len(validation_dataset)
                    model.eval()  # mark model as evaluating - don't apply dropouts or batch norms
                    with torch.no_grad():
                        # run through the validation dataset
                        val_loader = torch.utils.data.DataLoader(
                            validation_dataset,
                            batch_size = val_batch_size,
                            collate_fn = collate_fn,
                            shuffle = shuffle,
                            num_workers = num_workers
                        )
                        num_val_batches = 0

                        for val_data, val_labels in val_loader:
                            # val_data, val_labels = val_data.to(device), val_labels.to(device)
                            val_data = val_data.cuda() if gpu_available else val_data.cpu()
                            val_labels = val_labels.cuda() if gpu_available else val_labels.cpu()

                            # forward pass
                            val_logits = model(val_data)
                            # apply loss function
                            loss_tensor = loss_fn(val_logits, val_labels)
                            batch_loss = loss_tensor.item()

                            # calculate all metrics for validation dataset batch
                            # batch_metrics['val_loss'] = batch_loss
                            # if metrics is not None:
                            #     compute_metrics__(val_logits, val_labels, metrics,
                            #                       batch_metrics, validation_dataset=True)
                            # # same as cum_metrics[val_metric_name] += batch_metrics[val_metric_name] for all metrics
                            # cum_metrics = accumulate_metrics__(metrics_list, cum_metrics,
                            #                                    batch_metrics, validation_dataset=True)

                            # calculate all metrics for validation dataset batch
                            metrics_history.compute_batch_metrics(
                                val_logits, val_labels, batch_loss,
                                update_validation_metrics = True
                            )

                            num_val_batches += 1
                        else:
                            # validation loop completed for this epoch
                            # average metrics across all val-dataset batches
                            # for metric_name in metrics_list:
                            #     cum_metrics['val_%s' % metric_name] = \
                            #         cum_metrics['val_%s' % metric_name] / num_val_batches
                            #     history['val_%s' % metric_name].append(cum_metrics['val_%s' % metric_name])

                            # average metrics across all val-dataset batches
                            metrics_history.accumulate(accum_validation_metrics = True)

                            if (verbose in [1, 2]) and (
                                (epoch == 0) or ((epoch + 1) % report_interval == 0)):
                                # display train + val set metrics only if verbose =1 or 2 and at
                                # reporting interval epoch
                                # metrics_str = get_metrics_str__(
                                #     metrics_list, cum_metrics, validation_dataset=True)

                                metrics_str = metrics_history.get_metrics_str(
                                    include_validation_metrics = True
                                )

                                # learning_rates = get_lrates__(optimizer)
                                metrics_str += learning_rates
                                print(
                                    '\rEpoch (%*d/%*d): (%*d/%*d) -> %s' %
                                    (len_num_epochs, epoch + 1, len_num_epochs, epochs,
                                     len_tot_samples, samples, len_tot_samples,
                                     tot_samples,
                                     metrics_str),
                                    flush = True
                                )

            # check for early stopping
            if early_stopping is not None:
                assert early_stopping_metric in metrics_history.history.keys(), \
                    f"Early stopping metric {early_stopping_metric} not tracked during training!"
                curr_metric_val = metrics_history.history[early_stopping_metric][-1]
                early_stopping(model, curr_metric_val, epoch)
                if early_stopping.early_stop:
                    print(f"Early stopping training at epoch {epoch}")
                    if early_stopping.save_best_weights:
                        mod = model
                        if isinstance(model, PytkModuleWrapper):
                            mod = model.model
                        mod.load_state_dict(
                            torch.load(
                                early_stopping.checkpoint_file_path
                            )
                        )
                    return metrics_history.history

            # step the learning rate scheduler at end of epoch
            if (lr_scheduler is not None) and (epoch < epochs - 1):
                # have to go to these hoops as ReduceLROnPlateau requires a metric for step()
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # lr_metric = cum_metrics['val_loss'] if validation_dataset is not None \
                    #     else cum_metrics['loss']
                    lr_metric = metrics_history.history['val_loss'][
                        -1] if validation_dataset is not None \
                        else metrics_history.history['loss'][-1]
                    lr_scheduler.step(lr_metric)
                else:
                    lr_scheduler.step()

        return metrics_history.history
    finally:
        model = model.cpu()


def evaluate_model(
    model, dataset, collate_fn = None, loss_fn = None, batch_size = 64,
    metrics = None, num_workers = 0
):
    """ evaluate's model performance against dataset provided
        @params:
            - model: instance of model derived from nn.Model (or instance of pyt.PytModel or pyt.PytSequential)
            - dataset: instance of dataset to evaluate against (derived from torch.utils.data.Dataset)
            - loss_fn (optional, default=None): loss function to use during evaluation
                If not provided the model's loss function (i.e. model.loss_fn) is used
                asserts error if both are not provided!
            - batch_size (optional, default=64): batch size to use during evaluation
            - metrics (optional, default=None): list of metrics to evaluate 
                (e.g.: metrics=['acc','f1']) evaluates accuracy & f1-score
                Loss is ALWAYS evaluated, even when metrics=None
        @returns:
            - value of loss across dataset, if metrics=None (single value)
            - value of loss + list of metrics (in order provided), if metrics list is provided
            (e.g. if metrics=['acc', 'f1'], then a list of 3 values will be returned loss, accuracy & f1-score)
    """
    try:
        assert isinstance(model, nn.Module), \
            "evaluate_module() works with instances of nn.Module only!"
        assert isinstance(dataset, torch.utils.data.Dataset), \
            "dataset must be a subclass of torch.utils.data.Dataset"
        check_attribs__(model, loss_fn, check_only_loss = True)
        if loss_fn is None:
            loss_fn = model.loss_fn

        # evaluate on GPU if available
        gpu_available = torch.cuda.is_available()
        model = model.cuda() if gpu_available else model.cpu()

        samples, num_batches = 0, 0
        loader = torch.utils.data.DataLoader(
            dataset, batch_size = batch_size,
            collate_fn = collate_fn, shuffle = False,
            num_workers = num_workers
        )
        tot_samples = len(dataset)
        len_tot_samples = len(str(tot_samples))

        # history, batch_metrics, cum_metrics = \
        #     create_hist_and_metrics_ds__(metrics, dataset is not None)

        # metrics_list = ['loss']
        # if metrics is not None:
        #     metrics_list = metrics_list + metrics

        # NOTE: we are not tracking 'validation' metrics
        metrics_history = MetricsHistory(metrics)  # ataset is not None)

        with torch.no_grad():
            model.eval()
            for data, labels in loader:
                data = data.cuda() if gpu_available else data.cpu()
                labels = labels.cuda() if gpu_available else labels.cpu()

                # forward pass
                logits = model(data)
                # compute batch loss
                loss_tensor = loss_fn(logits, labels)
                batch_loss = loss_tensor.item()

                # compute all metrics for this batch
                # compute_metrics__(logits, labels, metrics,
                #                   batch_metrics, validation_dataset=False)
                # batch_metrics['loss'] = batch_loss
                # # same as cum_metrics[metric_name] += batch_metrics[metric_name] for all metrics
                # cum_metrics = accumulate_metrics__(
                #     metrics_list, cum_metrics, batch_metrics, validation_dataset=False)

                # compute all metrics for this batch
                metrics_history.compute_batch_metrics(
                    logits, labels, batch_loss,
                    update_validation_metrics = False
                )

                samples += len(labels)
                num_batches += 1

                # display progress for this batch
                # metrics_str = get_metrics_str__(
                #     metrics_list, batch_metrics, validation_dataset=False)

                # display progress for this batch
                metrics_str = metrics_history.get_batch_metrics_str(
                    include_validation_metrics = False
                )

                print(
                    '\rEvaluating (%*d/%*d) -> %s' %
                    (len_tot_samples, samples, len_tot_samples, tot_samples,
                     metrics_str),
                    end = '', flush = True
                )
            else:
                # compute average of all metrics provided in metrics list
                # for metric_name in metrics_list:
                #     cum_metrics[metric_name] = cum_metrics[metric_name] / num_batches

                # compute average of all metrics provided in metrics list
                metrics_history.accumulate(accum_validation_metrics = False)

                # metrics_str = get_metrics_str__(
                #     metrics_list, cum_metrics, validation_dataset=False)

                metrics_str = metrics_history.get_metrics_str(
                    include_validation_metrics = False
                )

                print(
                    '\rEvaluating (%*d/%*d) -> %s' %
                    (len_tot_samples, tot_samples, len_tot_samples, tot_samples,
                     metrics_str),
                    flush = True
                )

        if metrics is None:
            # return cum_metrics['loss']
            return metrics_history.history['loss']
        else:
            ret_metrics = []
            # for metric_name in metrics_list:
            #     ret_metrics.append(cum_metrics[metric_name])
            for metric_name in metrics_history.history.keys():
                ret_metrics.append(metrics_history.history[metric_name][0])
            return ret_metrics
    finally:
        model = model.cpu()


def predict_dataset(model, dataset, batch_size = 64, num_workers = 0):
    """ runs prediction on dataset (use for classification ONLY)
        @params:
            - model: instance of model derived from nn.Model (or instance of pyt.PytModel or pyt.PytSequential)
            - dataset: instance of dataset to evaluate against (derived from torch.utils.data.Dataset)
            - batch_size (optional, default=64): batch size to use during evaluation
        @returns:
            - tuple of Numpy Arrays of class predictions & labels
    """
    try:
        assert isinstance(model, nn.Module), \
            "predict_module() works with instances of nn.Module only!"
        assert isinstance(dataset, torch.utils.data.Dataset), \
            "dataset must be a subclass of torch.utils.data.Dataset"

        # run on GPU, if available
        gpu_available = torch.cuda.is_available()
        model = model.cuda() if gpu_available else model.cpu()

        loader = torch.utils.data.DataLoader(
            dataset, batch_size = batch_size,
            shuffle = False, num_workers = num_workers
        )

        preds, actuals = [], []

        for images, labels in loader:
            images = images.cuda() if gpu_available else images.cpu()
            labels = labels.cuda() if gpu_available else labels.cpu()

            # run prediction
            with torch.no_grad():
                model.eval()
                logits = model(images)
                batch_preds = list(logits.to("cpu").numpy())
                batch_actuals = list(labels.to("cpu").numpy())
                preds.extend(batch_preds)
                actuals.extend(batch_actuals)
        return np.array(preds), np.array(actuals)
    finally:
        model = model.cpu()


def predict(model, data):
    """ 
        runs predictions on Numpy Array (use for classification ONLY)
        @params:
            - model: instance of model derived from nn.Model (or instance of pyt.PytModel or pyt.PytSequential)
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
        gpu_available = torch.cuda.is_available()
        model = model.cuda() if gpu_available else model.cpu()

        # run prediction
        with torch.no_grad():
            model.eval()
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype = torch.float32)
            data = data.cuda() if gpu_available else data.cpu()
            # forward pass
            logits = model(data)
            preds = np.array(logits.cpu().numpy())
        return preds
    finally:
        model = model.cpu()


def save_model(
    model, model_save_name,
    model_save_dir = os.path.join('.', 'model_states'), verbose = 1
):
    """ saves Pytorch model to disk 
        @params:
            - model: instance of model derived from nn.Model (or instance of pytk.PytModel or pytk.PytSequential)
            - model_save_name: name of file or complete path of file to save model to 
            (NOTE: this file is overwritten without warning!)
            - model_save_dir (optional, defaul='./model_states'): folder to save Pytorch model to
            used only if model_save_name is just a name of file
            ignored if model_save_name is complete path to a file
    """

    # some checks
    if not model_save_name.endswith('.pt'):
        model_save_name = model_save_name + '.pt'

    # model_save_name could be just a file name or complete path
    if not (len(os.path.dirname(model_save_name)) == 0):
        # model_save_name is a complete path (e.g. ./model_save_dir/model_name.pt)
        # extract dir & file name
        model_save_dir, model_save_name = os.path.split(model_save_name)

    # create model_save_dir if it does not exist
    if not os.path.exists(model_save_dir):
        try:
            os.mkdir(model_save_dir)
        except OSError as err:
            print(
                f"FATAL: Unable to create folder {model_save_dir} to save Pytorch model!"
            )
            raise err

    model_save_path = os.path.join(model_save_dir, model_save_name)

    torch.save(model, model_save_path)
    if verbose == 1:
        print(f'Pytorch model saved to {model_save_path}')


def save_model_state2(model, model_save_path, verbose = 1):
    """ saves Pytorch state (state_dict) to disk  
        @params:
            - model: instance of model derived from nn.Model (or instance of pytk.PytModel or pytk.PytSequential)
            - model_save_path: complete path where model should be saved 
              (NOTE:
                 - the file is overwritten at destination without warning
                 - if `model_save_path` is just a file name, then model saved to current dir
                 - if `model_save_path` contains directory, it attempts to create directories if possible 
              )
    """

    # do I have an extension? If not append '.pyt' extension
    model_save_path = pathlib.Path(model_save_path)
    if len(model_save_path.suffix) == 0:
        # no extension specified, append a '.pyt' extension
        model_save_path = model_save_path.with_suffix('.pyt')

    # is model_save_path a path of just a file name?
    model_save_path_dir, model_save_path_name = os.path.split(model_save_path)
    if len(model_save_path_dir) == 0:
        # print("model_save_path is just a file-name (no dir). Saving to current dir")
        model_save_path = pathlib.Path.cwd() / model_save_path_name
        # print(f"model_save_path: {model_save_path}")
    else:
        # print("model_save_path contains dir name - will create if required")
        model_save_path_dir = pathlib.Path(model_save_path_dir).absolute()
        model_save_path = model_save_path_dir / model_save_path_name
        if not os.path.exists(model_save_path_dir):
            try:
                # print(f"model dir {model_save_path_dir} does NOT exist. Will try & create")
                os.mkdir(model_save_path_dir)
                # print(f"{model_save_path_dir} created successfully!")
            except OSError as err:
                print(f"FATAL: cannot create dir {model_save_path_dir}! Will abort")
                raise err

    # save model to model_save_path
    torch.save(model.state_dict(), model_save_path)
    if verbose == 1:
        print(f"Pytorch model saved to {model_save_path}")


def save_model_state(model, model_save_path, verbose = 1):
    """ saves Pytorch state (state_dict) to disk  
        @params:
            - model: instance of model derived from nn.Model (or instance of pytk.PytModel or pytk.PytSequential)
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
            print(f"Unable to create folder/directory {save_dir} to save model!")
            raise err

    # now save the model to file_path
    torch.save(model.state_dict(), model_save_path)
    if verbose == 1:
        print(f"Pytorch model saved to {model_save_path}")


def load_model(model_save_name, model_save_dir = './model_states', verbose = 1):
    """ loads model from disk and create a complete instance from saved state
        @params:
            - model_save_name: name of file or complete path of file to save model to 
            - model_save_dir (optional, defaul='./model_states'): folder to load the Pytorch model from
            used only if model_save_name is just a name of file; ignored if model_save_name is complete path to a file
        @returns:
            - 'ready-to-go' instance of model restored from saved state
    """
    if not model_save_name.endswith('.pt'):
        model_save_name = model_save_name + '.pt'

    # base_file_name could be just a file name or complete path
    if (len(os.path.dirname(model_save_name)) == 0):
        # only file name specified e.g. pyt_model.pt
        model_save_path = os.path.join(model_save_dir, model_save_name)
    else:
        # user passed in complete path e.g. './save_states/kr_model.h5'
        model_save_path = model_save_name

    if not os.path.exists(model_save_path):
        raise IOError('Cannot find model state file at %s!' % model_save_path)

    model = torch.load(model_save_path)
    model.eval()
    if verbose == 1:
        print('Pytorch model loaded from %s' % model_save_path)
    return model


def load_model_state2(file_path, verbose = 1):
    """ loads model from disk and create a complete instance from saved state
        @params:
            - model_save_name: name of file or complete path of file to save model to 
            - model_save_dir (optional, defaul='./model_states'): folder to load the Pytorch model from
            used only if model_save_name is just a name of file; ignored if model_save_name is complete path to a file
        @returns:
            - 'ready-to-go' instance of model restored from saved state
    """
    if not os.path.exists(file_path):
        raise IOError(f"Cannot load Keras model {file_path} - invalid path!")

    model = torch.load(file_path)
    model.eval()
    if verbose == 1:
        print(f"Pytorch model loaded from {file_path}")
    return model


def load_model_state(model, model_state_dict_path, verbose = 1):
    """ loads model's state dict from file on disk
        @params:
            - model: instance of model derived from nn.Model (or instance of pytk.PytModel or pytk.PytSequential)
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
    if verbose == 1:
        print(f"Pytorch model loaded from {model_state_dict_path}")
    model.eval()
    return model


def show_plots(history, metric = None, plot_title = None, fig_size = None):
    """ Useful function to view plot of loss values & 'metric' across the various epochs
        Works with the history object returned by the fit() or fit_generator() call """
    assert type(history) is dict

    """
        history is a dict object that saves epoch-wise loss (mandatory) and any other metrics you
        specify during model.compile(...) call. If a validation dataset is provided during training,
        then it will also save val_loss and corresponding validation metric(s).
        Suppose your specified metric=['acc'] in your model.compile(...) call, AND you also provided
        a validation dataset to your model.fit(...) call, then at the end of training your history dict 
        will look something like this:
            history {
                # NOTE: length of each list below = number of epochs used for training
                'loss': [ 0.045, 0.032 ...]         # list of epoch wise losses on train dataset (floats)
                'val_loss': [ 0.56, 0.044, ...]     # list of epoch wise losses on validation dataset (floats)]
                'acc': [ 0.022, 0.034, ...]         # list of epoch wise accuracies on train dataset (float)
                'val_acc': [ 0.034, 0.38, ...]      # list of epoch wise accuracies on validation dataset (float)
            }

            - history['loss']  - always present!
            - history['val_loss'] - always present if you specify a validation dataset during training
            - each metric your specify in your model.compile(...) call, will have a history['metric'] entry, and
            a history['val_metric'] if you provided a validation dataset during training.
            Example: suppose you have model.compile(...., metrics=['acc', 'r2_score']) and you also provide
            a validation dataset to your model.fit(....) call, which you run for 50 epochs, then your history object
            will look like:

            history {
                'loss' : [...]      # 50 epoch-wise losses on train dataset (corresponding to 50 epochs)
                'val_loss' : [...]  # 50 epoch-wise losses on cross-validation dataset
                'acc' : [...]       # 50 epoch-wise accuracy values on train dataset 
                'val_acc' : [...]   # 50 epoch-wise accuracy values on cross-validation dataset 
                'r2_score' : [...]  # 50 epoch-wise r2_score values on train dataset 
                'val_r2_score' : [...]   # 50 epoch-wise r2_score values on cross-validation dataset 
            }

    """

    # we must have at least loss in the history object
    assert 'loss' in history.keys(), \
        f"ERROR: expecting \'loss\' as one of the metrics in history object"
    if metric is not None:
        assert isinstance(metric, str), \
            "ERROR: expecting a string value for the \'metric\' parameter"
        assert metric in history.keys(), \
            f"{metric} is not tracked in training history!"

    loss_metrics = ['loss']
    if 'val_loss' in history.keys():
        loss_metrics.append('val_loss')
    # after above lines, loss_metrics = ['loss'] OR ['loss', 'val_loss']

    other_metrics = []
    if metric is not None:
        assert metric in METRICS_MAP.keys(), \
            f"ERROR: {metric} is not a metric being tracked. Please check compile() function!"
        other_metrics.append(metric)
        if f"val_{metric}" in history.keys():
            other_metrics.append(f"val_{metric}")
    # After above lines, other_metrics = [] OR
    #   if metric is not None (e.g. if metrics = 'accuracy'),
    #       then other_metrics = ['accuracy'] OR ['accuracy', 'val_accuracy']

    # display the plots
    col_count = 1 if len(other_metrics) == 0 else 2
    df = pd.DataFrame(history)

    with sns.axes_style("darkgrid"):
        sns.set_context("notebook", font_scale = 1.2)
        sns.set_style(
            {"font.sans-serif": ["SF Pro Display", "Arial", "Calibri", "DejaVu Sans"]}
        )

        f, ax = plt.subplots(
            nrows = 1, ncols = col_count,
            figsize = ((16, 5) if fig_size is None else fig_size)
        )
        axs = ax[0] if col_count == 2 else ax

        # plot the losses
        losses_df = df.loc[:, loss_metrics]
        losses_df.plot(ax = axs)
        # ax[0].set_ylim(0.0, 1.0)
        axs.grid(True)
        losses_title = 'Training \'loss\' vs Epochs' if len(
            loss_metrics
        ) == 1 else 'Training & Validation \'loss\' vs Epochs'
        axs.title.set_text(losses_title)

        # plot the metric, if specified
        if metric is not None:
            metrics_df = df.loc[:, other_metrics]
            metrics_df.plot(ax = ax[1])
            # ax[1].set_ylim(0.0, 1.0)
            ax[1].grid(True)
            metrics_title = f'Training \'{other_metrics[0]}\' vs Epochs' if len(
                other_metrics
            ) == 1 \
                else f'Training & Validation \'{other_metrics[0]}\' vs Epochs'
            ax[1].title.set_text(metrics_title)

        if plot_title is not None:
            plt.suptitle(plot_title)

        plt.show()
        plt.close()


def plot_confusion_matrix(
    cm, class_names = None, title = "Confusion Matrix",
    cmap = plt.cm.Blues
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

    with sns.axes_style("darkgrid"):
        sns.set_context("notebook", font_scale = 1.1)
        sns.set_style(
            {"font.sans-serif": ["SF Pro Display", "Arial", "Calibri", "DejaVu Sans"]}
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


# --------------------------------------------------------------------------------------------
# Utility classes
# --------------------------------------------------------------------------------------------

class PytkModule(nn.Module):
    """
        A class that you can inherit from to define your project's model
        Inheriting from this class provides a Keras-like interface for training model, evaluating model's 
        performance and for generating predictions.
            - As usual, you must override the constructor and the forward() method in your derived class.
            - You may provide a compile() function to set loss, optimizer and metrics at one location, else
            you will have to provide these as parameters to the fit(), evaluate() calls
        This class provides the following convenience methods: 
        - compile(loss, optimizer, metrics=None) - Keras-like compile() function. Sets the loss function,
                optimizer and metrics (if any) to use during testing. 
        - fit() - trains the model on numpy arrays (X = data & y = labels). 
        - fit_dataset() - trains model on torch.utils.data.Dataset instance. 
        - evaluate() - evaluate on numpy arrays (X & y)
        - evaluate_dataset() - evaluate on torch.utils.data.Dataset
        - predict() - generates class predictions
        - predict_module() - returns labels & predictions from dataset
        - save() - saves model's state to disk.
        - load() - loads the model's state from file on disk.
        - summary() - provides a Keras like summary of model
    """

    def __init__(self):
        super(PytkModule, self).__init__()
        self.loss_fn = None
        self.optimizer = None
        self.metrics_list = None

    def compile(self, loss, optimizer, metrics = None):
        """
            this function sets loss, optimizer and metrics attributes of the module
            @params:
                - loss: instance of loss function from torch.nn package (e.g. torch.nn.CrossEntroptLoss())
                - optimizer: instance of an optimizer from torch.optim package (e.g. torch.optim.Adam(...))
                - metrics (optional): list of metrics that should be tracked during training. Metrics are
                    specified in a list e.g. ['acc', 'f1-score']. The 'loss' metric is always tracked,
                    and you don't have to explicitly specify this in the list of metrics to track.
        """
        assert loss is not None, "ERROR: loss function must be a valid loss function!"
        assert optimizer is not None, "ERROR: optimizer must be a valid optimizer function"
        self.loss_fn = loss
        self.optimizer = optimizer
        self.metrics_list = metrics

    def forward(self, input):
        raise NotImplementedError(
            "You have landed up calling PytModule.forward(). " +
            "You must re-implement this method in your derived class!"
        )

    def fit_dataset(
        self, train_dataset, collate_fn = None, loss_fn = None,
        optimizer = None, validation_split = 0.0,
        validation_dataset = None, lr_scheduler = None, epochs = 25,
        batch_size = 64, metrics = None,
        shuffle = True, num_workers = 0, early_stopping = None, verbose = 2,
        report_interval = 1
    ):
        """ 
            train model on instance of torch.utils.data.Dataset
            @params:
                - train_dataset: instance of torch.utils.data.Dataset on which the model trains
                - loss_fn (optional, default=None): instance of one of the loss functions defined in Pytorch
                You could pass loss functions as a parameter to this function or pre-set it using the compile function.
                Value passed into this parameter takes precedence over value set in compile(...) call
                - optimizer (optional, default=None): instance of any optimizer defined by Pytorch
                You could pass optimizer as a parameter to this function or pre-set it using the compile function.
                Value passed into this parameter takes precedence over value set in compile(...) call
                - validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. 
                The model will set apart this fraction of the training data, will not train on it, and will 
                evaluate the loss and any model metrics on this data at the end of each epoch.
                - validation_dataset (optional, default=None) - instance of torch.utils.data.Dataset used for
                cross-validation
                If you pass a valid instance, then model is cross-trained on train_dataset and validation_dataset,
                else model
                is trained on just train_dataset. As a best practice, it is advisible to use cross training.
                - lr_scheduler (optional, default=NOne) - learning rate scheduler, used to step the learning rate across
                epochs
                as model trains. Instance of any scheduler defined in torch.optim.lr_scheduler package
                - epochs (optional, default=25): no of epochs for which model is trained
                - batch_size (optional, default=64): batch size to use
                - metrics (optional, default=None): list of metrics to measure across epochs as model trains. 
                Following metrics are supported (each identified by a key)
                    'acc' or 'accuracy' - accuracy
                    'prec' or 'precision' - precision
                    'rec' or 'recall' - recall
                    'f1' or 'f1_score' - f1_score
                    'roc_auc' - roc_auc_score
                    'mse' - mean squared error
                    'rmse' - root mean squared error
                metrics are provided as a list (e.g. ['acc','f1'])
                Loss is ALWAYS measures, even if you don't provide a list of metrics
                NOTE: if validation_dataset is provided, each metric is also measured for the validaton dataset
                - num_workers: no of worker threads to use to load datasets
                - early_stopping: instance of EarlyStopping class if early stopping is to be used (default: None)
                - verbose (integer = 0,1 or 2, default=0): sets verbosity level of progress reported during training
                    verbose=2: max verbose output, displays metrics batchwise
                    verbose=1: medium verbosity, displays batch progress, but metrics only at end of epoch
                    verbose=0: least verbose, no output is displayed until epoch completes.
                - report_interval (default=1): interval at which training progress is reported.
            @returns:
           - history object (which is a map of metrics measured across epochs).
             Each metric list is accessed as hist[metric_name] (e.g. hist['loss'] or hist['acc'])
             If validation_dataset is also provided, it will return corresponding metrics for validation dataset too
             (e.g. hist['val_acc'], hist['val_loss'])
        """
        p_loss_fn = self.loss_fn if loss_fn is None else loss_fn
        p_optimizer = self.optimizer if optimizer is None else optimizer
        p_metrics_list = self.metrics_list if metrics is None else metrics
        return train_model(
            self, train_dataset, collate_fn = collate_fn,
            loss_fn = p_loss_fn, optimizer = p_optimizer,
            validation_split = validation_split,
            validation_dataset = validation_dataset,
            lr_scheduler = lr_scheduler, epochs = epochs,
            batch_size = batch_size,
            metrics = p_metrics_list, shuffle = shuffle,
            num_workers = num_workers,
            early_stopping = early_stopping, verbose = verbose,
            report_interval = report_interval
        )

    def fit(
        self, X_train, y_train, collate_fn = None, loss_fn = None, optimizer = None,
        validation_split = 0.0,
        validation_data = None, lr_scheduler = None, epochs = 25, batch_size = 64,
        metrics = None, shuffle = True,
        num_workers = 0, early_stopping = None, verbose = 2, report_interval = 1
    ):
        """ 
            train model on Numpy arrays (X_train, y_train)
            @params:
                - X_train: Numpy array of features from training set
                - y_train: Numpy array of labels 
                - loss_fn (optional, default=None): instance of one of the loss functions defined in Pytorch
                You could pass loss functions as a parameter to this function or pre-set it using the compile function.
                Value passed into this parameter takes precedence over value set in compile(...) call
                - optimizer (optional, default=None): instance of any optimizer defined by Pytorch
                You could pass optimizer as a parameter to this function or pre-set it using the compile function.
                Value passed into this parameter takes precedence over value set in compile(...) call
                - validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. 
                The model will set apart this fraction of the training data, will not train on it, and will 
                evaluate the loss and any model metrics on this data at the end of each epoch.
                - validation_data (optional, default=None) - a tuple of validation data comprising Numpy arrays
                of features & labels (e.g. (X_val, y_val)) - columns of X_val & y_val should match those of X_train, y_train
                If you do not include validation_data, the model is trained on just the training data (X_train, y_train)
                - lr_scheduler (optional, default=NOne) - learning rate scheduler, used to step the learning rate across
                epochs as model trains. Instance of any scheduler defined in torch.optim.lr_scheduler package
                - epochs (optional, default=25): no of epochs for which model is trained
                - batch_size (optional, default=64): batch size to use
                - metrics (optional, default=None): list of metrics to measure across epochs as model trains. 
                Following metrics are supported (each identified by a key)
                    'acc' or 'accuracy' - accuracy
                    'prec' or 'precision' - precision
                    'rec' or 'recall' - recall
                    'f1' or 'f1_score' - f1_score
                    'roc_auc' - roc_auc_score
                    'mse' - mean squared error
                    'rmse' - root mean squared error
                metrics are provided as a list (e.g. ['acc','f1'])
                Loss is ALWAYS measures, even if you don't provide a list of metrics
                NOTE: if validation_data is provided, each metric is also measured for the validaton data
                - num_workers: no of worker threads to use to load datasets
                - early_stopping: instance of EarlyStopping class if early stopping is to be used (default: None)
                - verbose (integer = 0,1 or 2, default=0): sets verbosity level of progress reported during training
                    verbose=2: max verbose output, displays metrics batchwise
                    verbose=1: medium verbosity, displays batch progress, but metrics only at end of epoch
                    verbose=0: least verbose, no output is displayed until epoch completes.
                - report_interval (default=1): interval at which training progress is reported.
            @returns:
           - history object (which is a map of metrics measured across epochs).
             Each metric list is accessed as hist[metric_name] (e.g. hist['loss'] or hist['acc'])
             If validation_dataset is also provided, it will return corresponding metrics for validation dataset too
             (e.g. hist['val_acc'], hist['val_loss'])
        """

        assert ((X_train is not None) and (isinstance(X_train, np.ndarray))), \
            "Parameter error: X_train is None or is NOT an instance of np.ndarray"
        assert ((y_train is not None) and (isinstance(y_train, np.ndarray))), \
            "Parameter error: y_train is None or is NOT an instance of np.ndarray"
        if (y_train.dtype == np.int) or (y_train.dtype == np.long):
            y_dtype = np.long
        else:
            y_dtype = np.float32

        torch_X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
        torch_y_train = torch.from_numpy(y_train).type(
            torch.LongTensor if y_dtype == np.long else torch.FloatTensor
        )
        train_dataset = torch.utils.data.TensorDataset(torch_X_train, torch_y_train)

        validation_dataset = None
        if validation_data is not None:
            assert isinstance(validation_data, tuple)
            assert isinstance(
                validation_data[0],
                np.ndarray
            ), "Expecting validation_dataset[0] to be a Numpy array"
            assert isinstance(
                validation_data[1],
                np.ndarray
            ), "Expecting validation_dataset[1] to be a Numpy array"
            if (validation_data[1].dtype == np.int) or (
                validation_data[1].dtype == np.long):
                y_val_dtype = np.long
            else:
                y_val_dtype = np.float32

            torch_X_val = torch.from_numpy(validation_data[0]).type(torch.FloatTensor)
            torch_y_val = torch.from_numpy(validation_data[1]).type(
                torch.LongTensor if y_val_dtype == np.long else torch.FloatTensor
            )
            validation_dataset = torch.utils.data.TensorDataset(torch_X_val, torch_y_val)

        p_loss_fn = self.loss_fn if loss_fn is None else loss_fn
        p_optimizer = self.optimizer if optimizer is None else optimizer
        p_metrics_list = self.metrics_list if metrics is None else metrics
        return self.fit_dataset(
            train_dataset, collate_fn = collate_fn,
            loss_fn = p_loss_fn, optimizer = p_optimizer,
            validation_split = validation_split,
            validation_dataset = validation_dataset,
            lr_scheduler = lr_scheduler,
            epochs = epochs, batch_size = batch_size,
            metrics = p_metrics_list,
            shuffle = shuffle, num_workers = num_workers,
            early_stopping = early_stopping,
            verbose = verbose, report_interval = report_interval
        )

    def evaluate_dataset(
        self, dataset, collate_fn = None, loss_fn = None,
        batch_size = 64, metrics = None,
        num_workers = 0
    ):
        p_loss_fn = self.loss_fn if loss_fn is None else loss_fn
        p_metrics_list = self.metrics_list if metrics is None else metrics
        return evaluate_model(
            self, dataset, collate_fn = collate_fn, loss_fn = p_loss_fn,
            batch_size = batch_size,
            metrics = p_metrics_list,
            num_workers = num_workers
        )

    def evaluate(
        self, X, y, collate_fn = None, loss_fn = None, batch_size = 64,
        metrics = None, num_workers = 0
    ):
        assert ((X is not None) and (isinstance(X, np.ndarray))), \
            "Parameter error: X is None or is NOT an instance of np.ndarray"
        assert ((y is not None) and (isinstance(y, np.ndarray))), \
            "Parameter error: y is None or is NOT an instance of np.ndarray"

        if (y.dtype == np.int) or (y.dtype == np.long):
            y_dtype = np.long
        else:
            y_dtype = np.float32

        torch_X = torch.from_numpy(X).type(torch.FloatTensor)
        torch_y = torch.from_numpy(y).type(
            torch.LongTensor if y_dtype == np.long else torch.FloatTensor
        )
        p_dataset = torch.utils.data.TensorDataset(torch_X, torch_y)
        return self.evaluate_dataset(
            p_dataset, collate_fn = collate_fn,
            loss_fn = loss_fn, batch_size = batch_size,
            metrics = metrics, num_workers = num_workers
        )

    def predict_dataset(self, dataset, batch_size = 32, num_workers = 0):
        assert dataset is not None
        assert isinstance(dataset, torch.utils.data.Dataset)
        return predict_dataset(self, dataset, batch_size, num_workers = num_workers)

    def predict(self, data):
        assert data is not None
        assert ((isinstance(data, np.ndarray)) or (isinstance(data, torch.Tensor))), \
            "data must be an instance of Numpy ndarray or torch.tensor"
        return predict(self, data)

    def save__(self, model_save_name, model_save_dir = './model_states'):
        save_model(self, model_save_name, model_save_dir)

    def save(self, model_save_path):
        save_model_state(self, model_save_path)

    def load(self, model_save_path):
        load_model_state(self, model_save_path)

    def summary(self, input_shape):
        if torch.cuda.is_available():
            summary(self.cuda(), input_shape)
        else:
            summary(self.cpu(), input_shape)


class PytkModuleWrapper():
    """
    Utility class that wraps an instance of nn.Module or nn.Sequential or a pre-trained Pytorch module
    and provides a Keras-like interface to train, evaluate & predict results from model.
    """

    def __init__(self, model):
        super(PytkModuleWrapper, self).__init__()
        assert (model is not None) and isinstance(model, nn.Module), \
            "model parameter is None or not of type nn.Module"
        self.model = model
        self.loss_fn = None
        self.optimizer = None
        self.metrics_list = None

    def compile(self, loss, optimizer, metrics = None):
        assert loss is not None, "ERROR: loss function must be a valid loss function!"
        assert optimizer is not None, "ERROR: optimizer must be a valid optimizer function"
        self.loss_fn = loss
        self.optimizer = optimizer
        self.metrics_list = metrics

    def forward(self, inp):
        return self.model.forward(inp)

    def parameters(self, recurse = True):
        return self.model.parameters(recurse)

    def fit_dataset(
        self,
        train_dataset,
        collate_fn = None,
        loss_fn = None,
        optimizer = None,
        validation_split = 0.0,
        validation_dataset = None,
        lr_scheduler = None,
        epochs = 25,
        batch_size = 64,
        metrics = None,
        shuffle = True,
        num_workers = 0,
        early_stopping = None,
        verbose = 2,
        report_interval = 1
    ):

        p_loss_fn = self.loss_fn if loss_fn is None else loss_fn
        p_optimizer = self.optimizer if optimizer is None else optimizer
        p_metrics_list = self.metrics_list if metrics is None else metrics

        return train_model(
            self.model,
            train_dataset,
            collate_fn = collate_fn,
            loss_fn = p_loss_fn,
            optimizer = p_optimizer,
            validation_split = validation_split,
            validation_dataset = validation_dataset,
            lr_scheduler = lr_scheduler,
            epochs = epochs,
            batch_size = batch_size,
            metrics = p_metrics_list,
            shuffle = shuffle,
            num_workers = num_workers,
            early_stopping = early_stopping,
            verbose = verbose,
            report_interval = report_interval
        )

    def fit(
        self, X_train, y_train, collate_fn = None, loss_fn = None, optimizer = None,
        validation_split = 0.0,
        validation_data = None,
        lr_scheduler = None, epochs = 25, batch_size = 64, metrics = None,
        shuffle = True, num_workers = 0,
        early_stopping = None, verbose = 2, report_interval = 1
    ):

        assert ((X_train is not None) and (isinstance(X_train, np.ndarray))), \
            "Parameter error: X_train is None or is NOT an instance of np.ndarray"
        assert ((y_train is not None) and (isinstance(y_train, np.ndarray))), \
            "Parameter error: y_train is None or is NOT an instance of np.ndarray"
        if (y_train.dtype == np.int) or (y_train.dtype == np.long):
            y_dtype = np.long
        else:
            y_dtype = np.float32

        # train_dataset = XyDataset(X_train, y_train, y_dtype)
        torch_X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
        torch_y_train = torch.from_numpy(y_train).type(
            torch.LongTensor if y_dtype == np.long else torch.FloatTensor
        )
        train_dataset = torch.utils.data.TensorDataset(
            torch_X_train, torch_y_train
        )

        validation_dataset = None
        if validation_data is not None:
            assert isinstance(validation_data, tuple)
            assert isinstance(
                validation_data[0],
                np.ndarray
            ), "Expecting validation_dataset[0] to be a Numpy array"
            assert isinstance(
                validation_data[1],
                np.ndarray
            ), "Expecting validation_dataset[1] to be a Numpy array"
            if (validation_data[1].dtype == np.int) or (
                validation_data[1].dtype == np.long):
                y_val_dtype = np.long
            else:
                y_val_dtype = np.float32
            # validation_dataset = XyDataset(validation_data[0], validation_data[1], y_val_dtype)
            torch_X_val = torch.from_numpy(
                validation_data[0]
            ).type(torch.FloatTensor)
            torch_y_val = torch.from_numpy(validation_data[1]).type(
                torch.LongTensor if y_val_dtype == np.long else torch.FloatTensor
            )
            validation_dataset = torch.utils.data.TensorDataset(
                torch_X_val, torch_y_val
            )

        p_loss_fn = self.loss_fn if loss_fn is None else loss_fn
        p_optimizer = self.optimizer if optimizer is None else optimizer
        p_metrics_list = self.metrics_list if metrics is None else metrics
        return self.fit_dataset(
            train_dataset,
            collate_fn = collate_fn,
            loss_fn = p_loss_fn,
            optimizer = p_optimizer,
            validation_split = validation_split,
            validation_dataset = validation_dataset,
            lr_scheduler = lr_scheduler,
            epochs = epochs,
            batch_size = batch_size,
            metrics = p_metrics_list,
            shuffle = shuffle,
            num_workers = num_workers,
            early_stopping = early_stopping,
            verbose = verbose,
            report_interval = report_interval
        )

    def evaluate_dataset(
        self, dataset, collate_fn = None, loss_fn = None,
        batch_size = 64, metrics = None,
        num_workers = 0
    ):
        p_loss_fn = self.loss_fn if loss_fn is None else loss_fn
        p_metrics_list = self.metrics_list if metrics is None else metrics
        return evaluate_model(
            self.model, dataset, collate_fn = collate_fn,
            loss_fn = p_loss_fn,
            batch_size = batch_size,
            metrics = p_metrics_list, num_workers = num_workers
        )

    def evaluate(
        self, X, y, collate_fn = None, loss_fn = None, batch_size = 64,
        metrics = None, num_workers = 0
    ):
        assert ((X is not None) and (isinstance(X, np.ndarray))), \
            "Parameter error: X is None or is NOT an instance of np.ndarray"
        assert ((y is not None) and (isinstance(y, np.ndarray))), \
            "Parameter error: y is None or is NOT an instance of np.ndarray"
        if (y.dtype == np.int) or (y.dtype == np.long):
            y_dtype = np.long
        else:
            y_dtype = np.float32

        torch_X = torch.from_numpy(X).type(torch.FloatTensor)
        torch_y = torch.from_numpy(y).type(
            torch.LongTensor if y_dtype == np.long else torch.FloatTensor
        )
        p_dataset = torch.utils.data.TensorDataset(torch_X, torch_y)
        return self.evaluate_dataset(
            p_dataset, collate_fn = collate_fn,
            loss_fn = loss_fn, batch_size = batch_size,
            metrics = metrics, num_workers = num_workers
        )

    def predict_dataset(self, dataset, batch_size = 32, num_workers = 0):
        assert dataset is not None
        assert isinstance(dataset, torch.utils.data.Dataset)
        return predict_dataset(self.model, dataset, batch_size, num_workers = num_workers)

    def predict(self, data):
        assert data is not None
        assert ((isinstance(data, np.ndarray)) or (isinstance(data, torch.Tensor))), \
            "data must be an instance of Numpy ndarray or torch.tensor"
        return predict(self.model, data)

    def save__(self, model_save_name, model_save_dir = './model_states'):
        save_model(self.model, model_save_name, model_save_dir)

    def save(self, model_save_path):
        save_model_state(self.model, model_save_path)

    def load(self, model_save_path):
        load_model_state(self.model, model_save_path)

    def summary(self, input_shape):
        if torch.cuda.is_available():
            summary(self.model.cuda(), input_shape)
        else:
            summary(self.model.cpu(), input_shape)
