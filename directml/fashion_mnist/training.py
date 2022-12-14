""" training.py - helper functions to train/test model & evaluate performance """

import warnings

warnings.filterwarnings('ignore')

import sys

if sys.version_info < (3,):
    raise Exception("pytorch_toolkit does not support Python 2. Please use a Python 3+ interpreter!")

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
import torchsummary as ts
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


def accuracy(logits, labels):
    acc = torchmetrics.functional.accuracy(logits, labels)
    return acc


def train_model(model, device, train_dataset, loss_fn, optimizer, val_dataset=None, epochs=5, batch_size=64):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_train_recs = len(train_dataset)
    num_batches = len(train_dataloader)
    num_val_recs = 0 if val_dataset is None else len(val_dataset)
    len_num_epochs, len_num_train_recs = len(str(epochs)), len(str(num_train_recs))

    # epoch-wise metrics history
    history = {
        'loss': [],
        'acc': []
    }

    if val_dataset is not None:
        # also add validation metrics
        history['val_loss'] = []
        history['val_acc'] = []

    try:
        print(f"Training model on {device} for {epochs} epochs with batch size of {batch_size}...")
        model = model.to(device)

        for epoch in range(epochs):
            epoch_train_loss, epoch_train_acc = 0.0, 0.0
            train_recs_so_far = 0

            for batch, (X, y) in enumerate(train_dataloader):
                X, y = X.to(device), y.to(device)
                pred = model(X)
                batch_loss = loss_fn(pred, y)
                batch_acc = accuracy(pred, y)
                epoch_train_loss += batch_loss
                epoch_train_acc += batch_acc

                # back prop
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # display progress
                train_recs_so_far += len(X)
                print("\rEpoch (%*d/%*d): (%*d/%*d) -> loss: %.4f - acc: %.4f" %
                      (len_num_epochs, epoch + 1, len_num_epochs, epochs,
                       len_num_train_recs, train_recs_so_far, len_num_train_recs, num_train_recs,
                       batch_loss, batch_acc), end="", flush=True)
            else:
                # all batches of train_dataset done, compute & display epoch train loss & acc
                epoch_train_loss /= num_batches
                epoch_train_acc /= num_batches

                # append epoch training metrics to history
                history['loss'].append(epoch_train_loss.item())
                history['acc'].append(epoch_train_acc.item())

                train_batch_end = "\n" if val_dataset is None else ""
                train_batch_continue = "" if val_dataset is None else "..."
                print("\rEpoch (%*d/%*d): (%*d/%*d) -> loss: %.4f - acc: %.4f%s" %
                      (len_num_epochs, epoch + 1, len_num_epochs, epochs,
                       len_num_train_recs, train_recs_so_far, len_num_train_recs, num_train_recs,
                       epoch_train_loss, epoch_train_acc, train_batch_continue), end=train_batch_end, flush=True)

                if val_dataset is not None:
                    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
                    val_loss, val_acc = 0.0, 0.0
                    num_val_batches = len(val_dataloader)

                    with torch.no_grad():
                        for batch, (X, y) in enumerate(val_dataloader):
                            X, y = X.to(device), y.to(device)
                            pred = model(X)
                            batch_loss = loss_fn(pred, y)
                            batch_acc = accuracy(pred, y)
                            val_loss += batch_loss
                            val_acc += batch_acc

                    val_loss /= num_val_batches
                    val_acc /= num_val_batches

                    # append to history
                    history['val_loss'].append(val_loss.item())
                    history['val_acc'].append(val_acc.item())

                    print("\rEpoch (%*d/%*d): (%*d/%*d) -> loss: %.4f - acc: %.4f - val_loss: %.4f - val_acc: %.4f" %
                          (len_num_epochs, epoch + 1, len_num_epochs, epochs,
                           len_num_train_recs, train_recs_so_far, len_num_train_recs, num_train_recs,
                           epoch_train_loss, epoch_train_acc, val_loss, val_acc), flush=True)
        return history
    finally:
        model = model.to("cpu")


def evaluate_model(model, device, dataset, loss_fn, batch_size=64):
    try:
        model = model.to(device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        loss, acc = 0.0, 0.0
        num_batches = len(dataloader)
        num_records, len_num_recs = len(dataset), len(str(len(dataset)))
        recs_so_far = 0

        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                pred = model(X)
                batch_loss = loss_fn(pred, y)
                batch_acc = accuracy(pred, y)
                loss += batch_loss
                acc += batch_acc
                recs_so_far += len(X)
                print("\rEvaluating (%*d/%*d) -> loss: %.4f - acc: %.4f" %
                      (len_num_recs, recs_so_far, len_num_recs, num_records,
                       batch_loss, batch_acc), end='', flush=True)
            else:
                loss /= num_batches
                acc /= num_batches

                print("\rEvaluating (%*d/%*d) -> loss: %.4f - acc: %.4f" %
                      (len_num_recs, recs_so_far, len_num_recs, num_records,
                       loss, acc), flush=True)

        return loss.item(), acc.item()
    finally:
        model = model.to("cpu")


def predict_model(model, device, dataset, batch_size=64):
    try:
        model = model.to(device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds, actuals = [], []

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # run prediction
            with torch.no_grad():
                model.eval()
                logits = model(X)
                batch_preds = list(logits.to("cpu").numpy())
                batch_actuals = list(y.to("cpu").numpy())
                preds.extend(batch_preds)
                actuals.extend(batch_actuals)
        return np.array(preds), np.array(actuals)
    finally:
        model = model.to("cpu")


def save_model(model, model_save_path, verbose=1):
    save_dir, _ = os.path.split(model_save_path)
    if not os.path.exists(save_dir):
        try:
            os.mkdir(save_dir)
        except OSError as err:
            print(f"Unable to create folder/directory {save_dir} to save model!")
            raise err

    torch.save(model.state_dict(), model_save_path)
    if verbose == 1:
        print(f"Pytorch model saved to {model_save_path}")


def load_model(model, model_state_dict_path, verbose=1):
    model_save_path = pathlib.Path(model_state_dict_path).absolute()
    if not os.path.exists(model_save_path):
        raise IOError(f"ERROR: can't load model from {model_state_dict_path} - file does not exist!")

    state_dict = torch.load(model_state_dict_path)
    model.load_state_dict(state_dict)
    if verbose == 1:
        print(f"Pytorch model loaded from {model_state_dict_path}")
    model.eval()


def show_plots(history, metric='acc', plot_title=None, fig_size=None):
    assert type(history) is dict

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

    other_metrics = []
    if metric is not None:
        assert metric in history.keys(), \
            f"ERROR: {metric} is not a metric being tracked!"
        other_metrics.append(metric)
        if f"val_{metric}" in history.keys():
            other_metrics.append(f"val_{metric}")

    col_count = 1 if len(other_metrics) == 0 else 2
    df = pd.DataFrame(history)

    with sns.axes_style("darkgrid"):
        sns.set_context("notebook", font_scale=1.2)
        sns.set_style(
            {"font.sans-serif": ["SF Pro Display", "Arial", "Calibri", "DejaVu Sans"]})

        f, ax = plt.subplots(nrows=1, ncols=col_count,
                             figsize=((16, 5) if fig_size is None else fig_size))
        axs = ax[0] if col_count == 2 else ax

        # plot the losses
        losses_df = df.loc[:, loss_metrics]
        losses_df.plot(ax=axs)
        # ax[0].set_ylim(0.0, 1.0)
        axs.grid(True)
        losses_title = 'Training \'loss\' vs Epochs' if len(
            loss_metrics) == 1 else 'Training & Validation \'loss\' vs Epochs'
        axs.title.set_text(losses_title)

        # plot the metric, if specified
        if metric is not None:
            metrics_df = df.loc[:, other_metrics]
            metrics_df.plot(ax=ax[1])
            # ax[1].set_ylim(0.0, 1.0)
            ax[1].grid(True)
            metrics_title = f'Training \'{other_metrics[0]}\' vs Epochs' if len(other_metrics) == 1 \
                else f'Training & Validation \'{other_metrics[0]}\' vs Epochs'
            ax[1].title.set_text(metrics_title)

        if plot_title is not None:
            plt.suptitle(plot_title)

        plt.show()
        plt.close()
