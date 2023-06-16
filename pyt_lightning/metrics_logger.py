"""
metrics_logger.py - implements a custom logger that logs metrics across epochs,
  which we can then used to plot metrics. This is useful when you don't want to
  use Tensorboard for viewing epoch-wise progress of training

Thanks due to Marine Galantin 
(@see: https://stackoverflow.com/questions/69276961/how-to-extract-loss-and-accuracy-from-logger-by-each-epoch-in-pytorch-lightning)
"""
import collections

from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.logger import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsLogger(Logger):
    def __init__(self):
        super().__init__()

        self.history = collections.defaultdict(list)  # copy not necessary here
        # The defaultdict in contrast will simply create any items that you try to access

    @property
    def name(self):
        return "Metrics History Logger"

    @property
    def version(self):
        return "1.0"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        for metric_name, metric_value in metrics.items():
            if metric_name != "epoch":
                self.history[metric_name].append(metric_value)
            else:  # case epoch. We want to avoid adding multiple times the same. It happens for multiple losses.
                if (
                    not len(self.history["epoch"]) or not self.history["epoch"][-1] == metric_value  # len == 0:
                ):  # the last values of epochs is not the one we are currently trying to add.
                    self.history["epoch"].append(metric_value)
                else:
                    pass
        return

    def log_hyperparams(self, params):
        pass

    def plot_metrics(self, title=None, fig_size=None, col_count=3):
        # let's extract the relevant metrics data from self.history

        # We are interested only in epoch-level metrics
        useful_keys = [k for k in self.history.keys() if not k.endswith("_step")]
        # build our dict of useful keys & values
        data = dict((k, self.history[k]) for k in useful_keys)
        epoch_keys = [k for k in data.keys() if k.endswith("_epoch")]
        for key in epoch_keys:
            k2 = key[: -len("_epoch")]
            data[k2] = data[key]
            del data[key]
        data_df = pd.DataFrame(data)
        data_df = data_df.set_index("epoch")
        # data_cols = ['val_loss', 'val_metric1',...'val_metricN', 'train_loss', 'train_metric1',...'train_metricN']
        # of which the 'val_xxx' entries are all optional!
        data_cols = list(data_df.columns)
        # metrics_tracked = ['loss', 'metric1', 'metric2', ..., 'metricN']
        metrics_tracked = [m[m.find("_") + 1 :] for m in data_cols if m.startswith("train")]

        # now plot the metrics, col_count pairs per row
        with sns.axes_style("darkgrid"):
            sns.set(context="notebook")
            sns.set_style(
                {
                    "font.sans-serif": [
                        "Segoe UI",
                        "Calibri",
                        "SF Pro Display",
                        "Arial",
                        "DejaVu Sans",
                        "Sans",
                    ],
                }
            )
            fig_size = (16, 5) if fig_size is None else fig_size

            if len(metrics_tracked) == 1:
                # only loss is being tracked - that is mandatory!
                plt.figure(figsize=fig_size)
                plt.plot(
                    data_df.index,
                    data_df["train_loss"],
                    lw=2,
                    markersize=7,
                    color="steelblue",
                    marker="o",
                    label="train_loss",
                )
                ax_title = "Training loss vs epochs"
                if "val_loss" in data_cols:
                    plt.plot(
                        data_df.index,
                        data_df["val_loss"],
                        lw=2,
                        markersize=7,
                        color="firebrick",
                        marker="o",
                        label="val_loss",
                    )
                    ax_title = "Training & cross-val loss vs epochs"
                plt.title(ax_title)
                plt.legend(loc="best")
            else:
                # loss & additional metrics tracked
                # fmt:off
                MAX_COL_COUNT = col_count
                col_count = MAX_COL_COUNT if len(metrics_tracked) > MAX_COL_COUNT \
                    else len(metrics_tracked)
                row_count = len(metrics_tracked) // MAX_COL_COUNT
                row_count += 1 if len(metrics_tracked) % MAX_COL_COUNT != 0 else 0
                # fmt:on
                f, ax = plt.subplots(row_count, col_count, figsize=fig_size)
                for r in range(row_count):
                    for c in range(col_count):
                        index = r * (col_count - 1) + c
                        if index < len(metrics_tracked):
                            metric_name = metrics_tracked[index]
                            if row_count == 1:
                                ax[c].plot(
                                    data_df.index,
                                    data_df[f"train_{metric_name}"],
                                    lw=2,
                                    markersize=7,
                                    color="steelblue",
                                    marker="o",
                                    label=f"train_{metric_name}",
                                )
                                ax_title = f"Training {metric_name} vs epochs"
                                if f"val_{metric_name}" in data_cols:
                                    ax[c].plot(
                                        data_df.index,
                                        data_df[f"val_{metric_name}"],
                                        lw=2,
                                        markersize=7,
                                        color="firebrick",
                                        marker="o",
                                        label=f"val_{metric_name}",
                                    )
                                    ax_title = f"Training & cross-val {metric_name} vs epochs"
                                ax[c].set_title(ax_title)
                                ax[c].legend(loc="best")
                            else:
                                # more than 1 row
                                ax[r, c].plot(
                                    data_df.index,
                                    data_df[f"train_{metric_name}"],
                                    lw=2,
                                    markersize=7,
                                    color="steelblue",
                                    marker="o",
                                    label=f"train_{metric_name}",
                                )
                                ax_title = f"Training {metric_name} vs epochs"
                                if f"val_{metric_name}" in data_cols:
                                    ax[r, c].plot(
                                        data_df.index,
                                        data_df[f"val_{metric_name}"],
                                        lw=2,
                                        markersize=7,
                                        color="firebrick",
                                        marker="o",
                                        label=f"val_{metric_name}",
                                    )
                                    ax_title = f"Training & cross-val {metric_name} vs epochs"
                                ax[r, c].set_title(ax_title)
                                ax[r, c].legend(loc="best")

        if title is not None:
            plt.suptitle(title)
        plt.show()
        plt.close()


if __name__ == "__main__":
    raise RuntimeError("FATAL ERROR: this is a re-useable functions module. Cannot run it independently.")
