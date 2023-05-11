# -*- coding: utf-8 -*-
"""
pyt_mnist_dnn.py: multiclass classification of MNIST digits dataset using a Pytorch ANN
@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings
import sys

warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use("seaborn")
sns.set(style="darkgrid", context="notebook", font_scale=1.10)

# Pytorch imports
import torch

print("Using Pytorch version: ", torch.__version__)
import torch.nn as nn
from torchvision import datasets, transforms
import torchsummary
import torchmetrics

# import torch_directml

print(f"Using torchmetrics: {torchmetrics.__version__}")
import torchmetrics.classification

# My helper functions for training/evaluating etc.
import torch_training_toolkit as t3

SEED = t3.seed_all()
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    # else (
    #     torch_directml.device(torch_directml.default_device())
    #     if torch_directml.is_available()
    #     else "cpu"
    # )
    else "cpu"
)
print(f"Training model on {DEVICE}")


def load_data():
    """
    load the data using datasets API. We also split the test_dataset into
    cross-val/test datasets using 80:20 ration
    """
    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transformations
    )

    print("No of training records: %d" % len(train_dataset))

    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transformations)
    print("No of test records: %d" % len(test_dataset))

    # lets split the test dataset into val_dataset & test_dataset -> 8000:2000 records
    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [8000, 2000])
    print("No of cross-val records: %d" % len(val_dataset))
    print("No of test records: %d" % len(test_dataset))

    return train_dataset, val_dataset, test_dataset


def display_sample(
    sample_images, sample_labels, grid_shape=(10, 10), plot_title=None, sample_predictions=None
):
    # just in case these are not imported!
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("seaborn")

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


# some globals
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 28, 28, 1, 10


# define our network using Linear layers only
class MNISTNet2(nn.Module):
    def __init__(self):
        super(MNISTNet2, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            t3.Linear(IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            t3.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            # NOTE: we'll be using nn.CrossEntropyLoss(), which includes a
            # logsoftmax call that applies a softmax function to outputs.
            # So, don't apply one yourself!
            t3.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x


# network using Convnets
class MNISTConvNet2(nn.Module):
    def __init__(self):
        super(MNISTConvNet2, self).__init__()
        self.convNet = nn.Sequential(
            t3.Conv2d(1, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.20),
            t3.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.10),
            nn.Flatten(),
            t3.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Dropout(p=0.20),
            t3.Linear(512, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.convNet(x)
        return x


DO_TRAINING = True
DO_PREDICTION = True
SHOW_SAMPLE = False
USE_CNN = True  # if False, will use an MLP

MODEL_SAVE_NAME = "pyt_mnist_cnn.pyt" if USE_CNN else "pyt_mnist_dnn.pyt"
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "model_states", MODEL_SAVE_NAME)
NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, L2_REG = (15 if USE_CNN else 20), 128, 0.001, 0.0005


def main():
    print("Loading datasets...")

    train_dataset, val_dataset, test_dataset = load_data()

    loss_fn = nn.CrossEntropyLoss()
    metrics_map = {"acc": torchmetrics.classification.MulticlassAccuracy(num_classes=NUM_CLASSES)}
    trainer = t3.Trainer(
        loss_fn=loss_fn,
        device=DEVICE,
        metrics_map=metrics_map,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
    )

    if SHOW_SAMPLE:
        # display sample from test dataset
        print("Displaying sample from train dataset...")
        trainloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
        data_iter = iter(trainloader)
        # fetch first batch of 64 images & labels
        images, labels = next(data_iter)
        display_sample(
            images.cpu().numpy(),
            labels.cpu().numpy(),
            grid_shape=(8, 8),
            plot_title="Sample Images",
        )

    if DO_TRAINING:
        print(f'Using {"CNN" if USE_CNN else "ANN"} model...')
        model = MNISTConvNet2() if USE_CNN else MNISTNet2()
        model = model.to(DEVICE)
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG
        )
        # display model structure a-la-Keras
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

        # train model
        print(f'Training {"CNN" if USE_CNN else "ANN"} model')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 20, 30, 40], gamma=0.15
        )

        hist = trainer.fit(
            model, optimizer, train_dataset, validation_dataset=val_dataset, lr_scheduler=scheduler
        )
        # save hist to pickle file
        hist.plot_metrics(title="Model Training Metrics", fig_size=(8, 6))

        # evaluate model performance on train/eval & test datasets
        print("Evaluating model performance...")
        metrics = trainer.evaluate(model, train_dataset)
        print(f"Training metrics -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}")
        metrics = trainer.evaluate(model, val_dataset)
        print(f"Cross-val metrics -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}")
        metrics = trainer.evaluate(model, test_dataset)
        print(f"Testing metrics   -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}")

        # save model state
        t3.save_model(model, MODEL_SAVE_PATH)
        del model

    if DO_PREDICTION:
        print("Running predictions...")
        model = MNISTConvNet2() if USE_CNN else MNISTNet2()
        model = model.to(DEVICE)
        # load model state from .pt file
        t3.load_model(model, MODEL_SAVE_PATH)
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

        # evaluate model performance on train/eval & test datasets
        print("Evaluating model performance...")
        metrics = trainer.evaluate(model, train_dataset)
        print(f"Training metrics -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}")
        metrics = trainer.evaluate(model, val_dataset)
        print(f"Cross-val metrics -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}")
        metrics = trainer.evaluate(model, test_dataset)
        print(f"Testing metrics   -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}")

        y_pred, y_true = trainer.predict(model, test_dataset)
        y_pred = np.argmax(y_pred, axis=1)
        print("Sample labels (50): ", y_true[:50])
        print("Sample predictions: ", y_true[:50])
        print("We got %d/%d incorrect!" % ((y_pred != y_true).sum(), len(y_true)))

        # display sample from test dataset
        print("Displaying sample predictions...")
        trainloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
        data_iter = iter(trainloader)
        images, labels = next(data_iter)  # fetch a batch of 64 random images
        preds = trainer.predict(model, images.cpu().numpy())
        preds = np.argmax(preds, axis=1)
        display_sample(
            images.cpu().numpy(),
            labels.cpu().numpy(),
            sample_predictions=preds,
            grid_shape=(8, 8),
            plot_title="Sample Predictions",
        )


if __name__ == "__main__":
    main()

# --------------------------------------------------
# Results:
#   MLP with epochs=50, batch-size=32, LR=0.001
#       Training  -> acc: 99.77%
#       Cross-val -> acc: 98.65%
#       Testing   -> acc: 97.66%
#   CNN with epochs=25, batch-size=32, LR=0.001
#       Training  -> acc: 99.89%
#       Cross-val -> acc: 99.50%
#       Testing   -> acc: 99.17%
# Clearly the CNN performs better than the MLP
# --------------------------------------------------
