# -*- coding: utf-8 -*-
""" metrics_history.py - custom class to track training metrics """
import warnings

warnings.filterwarnings("ignore")

import sys

if sys.version_info < (2,):
    raise Exception("torch_training_toolkit does not support Python 1. Please use a Python 3+ interpreter!")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pytorch imports
import torch


class MetricsHistory:
    """class to calculate & store metrics across training batches"""

    def __init__(self, metrics_map, include_val_metrics=False):
        """constructor of MetricsHistory class
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
        """returns list of all metrics tracked"""
        metric_names = ["loss"]
        if self.metrics_map is not None:
            metric_names.extend([key for key in self.metrics_map.keys()])
        return list(set(metric_names))  # eliminate any duplicates

    def __createMetricsHistory(self):
        """Internal function: creates a map to store metrics history"""
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

        # # must add key for loss
        # metric_names = ["loss"]
        # # rest will depend on metrics defined in metrics_map
        # if self.metrics_map is not None:
        #     metric_names.extend([key for key in self.metrics_map.keys()])

        metric_names = self.tracked_metrics()

        for metric_name in metric_names:
            metrics_history[metric_name] = {"batch_vals": [], "epoch_vals": []}

            if self.include_val_metrics:
                metrics_history[f"val_{metric_name}"] = {"batch_vals": [], "epoch_vals": []}
        return metrics_history

    def get_metric_vals(self, metrics_list, include_val_metrics=False):
        """gets the last epoch value for each metric tracked"""
        metrics_list2 = ["loss"] if metrics_list is None else metrics_list
        metric_vals = {
            metric_name: self.metrics_history[metric_name]["epoch_vals"][-1] for metric_name in metrics_list2
        }
        if include_val_metrics:
            for metric_name in metrics_list2:
                metric_vals[f"val_{metric_name}"] = self.metrics_history[f"val_{metric_name}"]["epoch_vals"][-1]
        return metric_vals

    def calculate_batch_metrics(self, preds: torch.tensor, targets: torch.tensor, loss_val: float, val_metrics=False):
        if val_metrics:
            self.metrics_history["val_loss"]["batch_vals"].append(loss_val)
        else:
            self.metrics_history["loss"]["batch_vals"].append(loss_val)

        if self.metrics_map is not None:
            for metric_name, calc_fxn in self.metrics_map.items():
                metric_val = calc_fxn(preds, targets).item()
                if val_metrics:
                    self.metrics_history[f"val_{metric_name}"]["batch_vals"].append(metric_val)
                else:
                    self.metrics_history[metric_name]["batch_vals"].append(metric_val)

    def calculate_epoch_metrics(self, val_metrics=False):
        """calculates average value of the accumulated metrics from last batch & appends
        to epoch metrics list
        """
        metric_names = self.tracked_metrics()

        for metric in metric_names:
            if val_metrics:
                mean_val = np.array(self.metrics_history[f"val_{metric}"]["batch_vals"]).mean()
                self.metrics_history[f"val_{metric}"]["epoch_vals"].append(mean_val)
            else:
                mean_val = np.array(self.metrics_history[metric]["batch_vals"]).mean()
                self.metrics_history[metric]["epoch_vals"].append(mean_val)

    def clear_batch_metrics(self):
        """reset the lists that track batch metrics"""
        metric_names = self.tracked_metrics()

        for metric in metric_names:
            self.metrics_history[metric]["batch_vals"].clear()
            if self.include_val_metrics:
                self.metrics_history[f"val_{metric}"]["batch_vals"].clear()

    def get_metrics_str(self, batch_metrics=True, include_val_metrics=False):
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
                    metric_val = self.metrics_history[f"val_{metric_name}"]["batch_vals"][-1]
                else:
                    # epoch metrics
                    metric_val = self.metrics_history[f"val_{metric_name}"]["epoch_vals"][-1]
                metrics_str += f"val_{metric_name}: {metric_val:.4f} - "
        # trim ending " - "
        if metrics_str.endswith(" - "):
            metrics_str = metrics_str[:-3]
        return metrics_str

    def plot_metrics(self, title=None, fig_size=None, col_count=3):
        """plots epoch metrics values across epochs to show how
        training progresses
        """
        metric_names = self.tracked_metrics()
        metric_vals = {metric_name: self.metrics_history[metric_name]["epoch_vals"] for metric_name in metric_names}
        if self.include_val_metrics:
            # also has validation metrics
            for metric_name in metric_names:
                metric_vals[f"val_{metric_name}"] = self.metrics_history[f"val_{metric_name}"]["epoch_vals"]

        # we will plot a max of 3 metrics per row
        MAX_COL_COUNT = col_count
        col_count = MAX_COL_COUNT if len(metric_names) > MAX_COL_COUNT else len(metric_names)
        row_count = len(metric_names) // MAX_COL_COUNT
        row_count += 1 if len(metric_names) % MAX_COL_COUNT != 0 else 0
        # we'll always have "loss" metric in list, so safest to pick!
        x_vals = np.arange(1, len(metric_vals["loss"]) + 1)

        with sns.axes_style("darkgrid"):
            sns.set_context("notebook")  # , font_scale = 1.2)
            sns.set_style(
                {"font.sans-serif": ["Segoe UI", "Calibri", "SF Pro Display", "Arial", "DejaVu Sans", "Sans"]}
            )
            fig_size = (16, 5) if fig_size is None else fig_size

            if len(metric_names) == 1:
                # only loss
                plt.figure(figsize=fig_size)
                plt.plot(
                    x_vals,
                    metric_vals["loss"],
                    lw=2,
                    markersize=5,
                    color="steelblue",
                    marker="o",
                )
                if self.include_val_metrics:
                    plt.plot(
                        x_vals,
                        metric_vals["val_loss"],
                        lw=2,
                        markersize=5,
                        color="firebrick",
                        marker="o",
                    )
                legend = ["train", "valid"] if self.include_val_metrics else ["train"]
                plt_title = (
                    f"Training & Cross-validation  Loss vs Epochs" if len(legend) == 2 else f"Training Loss vs Epochs"
                )
                plt.title(plt_title)
                plt.legend(legend, loc="best")
            else:
                f, ax = plt.subplots(row_count, col_count, figsize=fig_size)
                for r in range(row_count):
                    for c in range(col_count):
                        index = r * (col_count - 1) + c
                        if index < len(metric_names):
                            metric_name = metric_names[index]
                            if row_count == 1:
                                ax[c].plot(
                                    x_vals,
                                    metric_vals[metric_name],
                                    lw=2,
                                    markersize=5,
                                    marker="o",
                                    color="steelblue",
                                )
                            else:
                                ax[r, c].plot(
                                    x_vals,
                                    metric_vals[metric_name],
                                    lw=2,
                                    markersize=5,
                                    marker="o",
                                    color="steelblue",
                                )
                            if self.include_val_metrics:
                                if row_count == 1:
                                    ax[c].plot(
                                        x_vals,
                                        metric_vals[f"val_{metric_name}"],
                                        lw=2,
                                        markersize=5,
                                        marker="o",
                                        color="firebrick",
                                    )
                                else:
                                    ax[r, c].plot(
                                        x_vals,
                                        metric_vals[f"val_{metric_name}"],
                                        lw=2,
                                        markersize=5,
                                        marker="o",
                                        color="firebrick",
                                    )
                            legend = ["train", "valid"] if self.include_val_metrics else ["train"]
                            ax_title = (
                                f"Training & Cross-validation '{metric_name}' vs Epochs"
                                if len(legend) == 2
                                else f"Training '{metric_name}' vs Epochs"
                            )
                            if row_count == 1:
                                ax[c].legend(legend, loc="best")
                                ax[c].set_title(ax_title)
                            else:
                                ax[r, c].legend(legend, loc="best")
                                ax[r, c].set_title(ax_title)

        if title is not None:
            plt.suptitle(title)

        plt.show()
