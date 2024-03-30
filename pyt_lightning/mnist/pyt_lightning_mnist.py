#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pyt_lightning_mnist.py: multiclass classification of MNIST dataset using an ANN
with Pytorch Lightning

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for
commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import sys, os
import warnings

# need Python >= 3.2 for pathlib
# fmt: off
if sys.version_info < (3, 2,):
    import platform

    raise ValueError(
        f"{__file__} required Python version >= 3.2. You are using Python "
        f"{platform.python_version}")

# NOTE: @override decorator available from Python 3.12 onwards
# Using override package which provides similar functionality in previous versions
if sys.version_info < (3, 12,):
    from overrides import override
else:
    from typing import override
# fmt: on


import pathlib
import logging.config


BASE_PATH = pathlib.Path(__file__).parent.parent
sys.path.append(str(BASE_PATH))

warnings.filterwarnings("ignore")
logging.config.fileConfig(fname=BASE_PATH / "logging.config")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
plt.style.use("seaborn-v0_8")
sns.set(style="whitegrid", font_scale=1.1, palette="muted")

# Pytorch imports
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
import torchmetrics
import torchsummary

print("Using Pytorch version: ", torch.__version__)
print("Using Pytorch Lightning version: ", pl.__version__)

# fmt: off
# my utility functions to use with lightning
import pytorch_enlightning as pel
print(f"Pytorch En(hanced)Lightning: {pel.__version__}")
# fmt: on

SEED = pl.seed_everything()

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = pathlib.Path(__file__).parent / "data"
# assert os.path.exists(DATA_FILE_PATH), f"FATAL: {DATA_FILE_PATH} - data file does not exist!"
MODEL_STATE_NAME = "pyt_mnist_dnn.pth"
MODEL_STATE_PATH = pathlib.Path(__file__).parent / "model_state" / MODEL_STATE_NAME
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 28, 28, 1, 10

logger.info(f"Training model on {DEVICE}")
logger.info(f"Using data file {DATA_FILE_PATH}")


def get_datasets():
    xforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.MNIST(
        root=DATA_FILE_PATH,
        train=True,
        download=True,
        transform=xforms,
    )

    test_dataset = datasets.MNIST(
        root=DATA_FILE_PATH,
        train=False,
        download=True,
        transform=xforms,
    )
    val_dataset, test_dataset = torch.utils.data.random_split(
        test_dataset,
        [8000, 2000],
    )
    logger.info(
        f"get_datasets() -> train_dataset: {len(train_dataset)} recs - val_dataset: {len(val_dataset)} recs - test_dataset: {len(test_dataset)} recs"
    )
    return train_dataset, val_dataset, test_dataset


def display_sample(
    sample_images,
    sample_labels,
    grid_shape=(10, 10),
    plot_title=None,
    sample_predictions=None,
):
    # just in case these are not imported!
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("seaborn-v0_8")

    num_rows, num_cols = grid_shape
    assert sample_images.shape[0] == num_rows * num_cols

    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=0.90)
        sns.set_style(
            {
                "font.sans-serif": [
                    "SF Pro Rounded",
                    "SF Pro Display",
                    "SF UI Text",
                    "Verdana",
                    "Arial",
                    "DejaVu Sans",
                    "sans",
                ]
            }
        )

        f, ax = plt.subplots(
            num_rows,
            num_cols,
            figsize=(14, 10),
            gridspec_kw={"wspace": 0.02, "hspace": 0.25},
            squeeze=True,
        )
        # fig = ax[0].get_figure()
        f.tight_layout()
        f.subplots_adjust(top=0.90)

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")
                # show selected image
                ax[r, c].imshow(sample_images[image_index].squeeze(), cmap="Greys")

                if sample_predictions is None:
                    # but show the prediction in the title
                    title = ax[r, c].set_title("No: %d" % sample_labels[image_index])
                else:
                    pred_matches_actual = (
                        sample_labels[image_index] == sample_predictions[image_index]
                    )
                    if pred_matches_actual:
                        # show title from prediction or actual in green font
                        title = "%s" % sample_predictions[image_index]
                        title_color = "g"
                    else:
                        # show title as actual/prediction in red font
                        title = "%s/%s" % (
                            [sample_labels[image_index]],
                            [sample_predictions[image_index]],
                        )
                        title_color = "r"

                    # but show the prediction in the title
                    title = ax[r, c].set_title(title)
                    # if prediction is incorrect title color is red, else green
                    plt.setp(title, color=title_color)

        if plot_title is not None:
            plt.suptitle(plot_title)
        plt.show()
        plt.close()


class MNISTModel(pel.EnLitModule):
    def __init__(self, num_classes, lr):
        super(MNISTModel, self).__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = torchmetrics.classification.MulticlassAccuracy(
            num_classes=self.num_classes
        )
        self.f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=self.num_classes
        )

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.num_classes),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def process_batch(self, batch, batch_idx, dataset_name):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, labels)
        acc = self.acc(logits, labels)
        f1 = self.f1(logits, labels)

        metrics_dict = {
            f"{dataset_name}_loss": loss,
            f"{dataset_name}_acc": acc,
            f"{dataset_name}_f1": f1,
        }

        if dataset_name in ["train", "val"]:
            # self.log_dict({'val_loss': loss, 'val_acc': val_acc})
            self.log_dict(metrics_dict, on_step=True, on_epoch=True, prog_bar=True)
            # self.log(
            #     f"{dataset_name}_loss", loss, on_step=True, on_epoch=True, prog_bar=True
            # )
            # self.log(
            #     f"{dataset_name}_acc", acc, on_step=True, on_epoch=True, prog_bar=True
            # )
            # self.log(
            #     f"{dataset_name}_f1", f1, on_step=True, on_epoch=True, prog_bar=True
            # )
        else:
            self.log_dict(metrics_dict, prog_bar=True)
            # self.log(f"{dataset_name}_loss", loss, prog_bar=True)
            # self.log(f"{dataset_name}_acc", acc, prog_bar=True)
            # self.log(f"{dataset_name}_f1", f1, prog_bar=True)
        return {"loss": loss, "acc": acc, "f1": f1}


def main():
    parser = pel.TrainingArgsParser()
    args = parser.parse_args()

    train_dataset, val_dataset, test_dataset = get_datasets()

    # NOTE: Pytorch on Windows - DataLoader with num_workers > 0 is very slow
    # looks like a known issue
    # @see: https://github.com/pytorch/pytorch/issues/12831
    # This is a hack for Windows
    NUM_WORKERS = 0 if os.name == "nt" else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    if args.show_sample:
        print("Displaying random sample from test dataset")
        data_iter = iter(test_loader)
        # fetch first batch of 64 images & labels
        images, labels = next(data_iter)
        display_sample(
            images.cpu().numpy(),
            labels.cpu().numpy(),
            grid_shape=(8, 8),
            plot_title="Sample Images from Test Dataset",
        )

    if args.train:
        model = MNISTModel(NUM_CLASSES, args.lr).to(DEVICE)
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

        metrics_history = pel.MetricsLogger()
        progbar = pel.EnLitProgressBar()
        trainer = pl.Trainer(
            max_epochs=args.epochs, logger=metrics_history, callbacks=[progbar]
        )
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        metrics_history.plot_metrics("Model Performance")
        pel.save_model(model, MODEL_STATE_PATH)
        del model
        del metrics_history
        del progbar

    if args.eval:
        model = MNISTModel(NUM_CLASSES, args.lr).to(DEVICE)
        # print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))
        model = pel.load_model(model, MODEL_STATE_PATH)
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

        # run a validation on Model
        progbar = pel.EnLitProgressBar()
        trainer = pl.Trainer(callbacks=[progbar])
        print(f"Validating on train-dataset...")
        trainer.validate(model, dataloaders=train_loader)
        print(f"Validating on val-dataset...")
        trainer.validate(model, dataloaders=val_loader)
        print(f"Validating on test-dataset...")
        trainer.validate(model, dataloaders=test_loader)
        del model
        del progbar

    if args.pred:
        model = MNISTModel(NUM_CLASSES, args.lr).to(DEVICE)
        # print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))
        model = pel.load_model(model, MODEL_STATE_PATH)
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

        # predict from test_dataset
        preds, actuals = pel.predict_module(model, test_loader, DEVICE)
        preds = np.argmax(preds, axis=1)
        print("Sample labels (50): ", actuals[:50])
        print("Sample predictions: ", preds[:50])
        print("We got %d/%d incorrect!" % ((preds != actuals).sum(), len(actuals)))

        if args.show_sample:
            # display sample predictions
            data_iter = iter(test_loader)
            images, labels = next(data_iter)
            preds, actuals = pel.predict_module(
                model,
                (images.cpu().numpy(), labels.cpu().numpy()),
                DEVICE,
            )
            preds = np.argmax(preds, axis=1)
            accu = (preds == actuals).sum() / len(actuals)
            display_sample(
                images,
                actuals[: args.batch_size],
                sample_predictions=preds[: args.batch_size],
                grid_shape=(8, 8),
                plot_title=f"Sample Predictions ({accu*100:.2f}% accuracy for sample)",
            )
        del model


if __name__ == "__main__":
    main()
