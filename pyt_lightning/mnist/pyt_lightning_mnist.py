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
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
plt.style.use("seaborn-v0_8")
sns.set(style="whitegrid", font_scale=1.1, palette="muted")

# Pytorch imports
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchmetrics
import torchsummary

from cl_options import parse_command_line
from utils import save_model, load_model, predict_module, predict_array

# print("Using Pytorch version: ", torch.__version__)
# print("Using Pytorch Lightning version: ", pl.__version__)

SEED = pl.seed_everything()

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = pathlib.Path(__file__).parent / "data"
# assert os.path.exists(DATA_FILE_PATH), f"FATAL: {DATA_FILE_PATH} - data file does not exist!"

# logger.info(f"Training model on {DEVICE}")
# logger.info(f"Using data file {DATA_FILE_PATH}")


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


IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 28, 28, 1, 10


# our model
class MNISTModelBase(pl.LightningModule):
    def __init__(self, num_classes, lr):
        super(MNISTModelBase, self).__init__()

        self.num_classes = num_classes
        self.lr = lr

        self.net = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = torchmetrics.classification.MulticlassAccuracy(
            num_classes=self.num_classes
        )

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def process_batch(self, batch, batch_idx, dataset_name):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, labels)
        acc = self.acc(logits, labels)
        self.log(f"{dataset_name}_loss", loss, prog_bar=True)
        self.log(f"{dataset_name}_acc", acc, prog_bar=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        """training step"""
        metrics = self.process_batch(batch, batch_idx, "train")
        return metrics[0]

    def validation_step(self, batch, batch_idx):
        """cross-validation step"""
        metrics = self.process_batch(batch, batch_idx, "val")
        return metrics[0]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """run predictions on a batch"""
        return self.forward(batch)


class MNISTModelANN(MNISTModelBase):
    def __init__(self, num_classes, lr):
        super(MNISTModelANN, self).__init__(num_classes, lr)

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, NUM_CLASSES),
        )


def main():
    args = parse_command_line()
    MODEL_STATE_NAME = "pyt_mnist_dnn.pth"
    MODEL_STATE_PATH = pathlib.Path(__file__).parent / "model_state" / MODEL_STATE_NAME

    train_dataset, val_dataset, test_dataset = get_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
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
        model = MNISTModelANN(NUM_CLASSES, args.lr)
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

        trainer = pl.Trainer(
            max_epochs=args.epochs,
        )
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        save_model(model, MODEL_STATE_PATH)
        del model

    if args.pred:
        model = MNISTModelANN(NUM_CLASSES, args.lr)
        # print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))
        model = load_model(model, MODEL_STATE_PATH)
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

        # run a validation on Model
        trainer = pl.Trainer()
        print(f"Validating on train-dateset...")
        trainer.validate(model, dataloaders=train_loader)
        print(f"Validating on val-dateset...")
        trainer.validate(model, dataloaders=val_loader)
        print(f"Validating on test-dateset...")
        trainer.validate(model, dataloaders=test_loader)

        # predict from test_dataset
        preds, actuals = predict_module(model, test_loader, DEVICE)
        preds = np.argmax(preds, axis=1)
        print("Sample labels (50): ", actuals[:50])
        print("Sample predictions: ", preds[:50])
        print("We got %d/%d incorrect!" % ((preds != actuals).sum(), len(actuals)))

        if args.show_sample:
            # display sample predictions
            data_iter = iter(test_loader)
            images, labels = next(data_iter)
            preds, actuals = predict_module(
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


if __name__ == "__main__":
    main()
