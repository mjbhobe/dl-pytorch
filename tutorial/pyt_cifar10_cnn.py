# -*- coding: utf-8 -*-
"""
pyt_cifar10_cnn.py - multi-class classification of CIFAR10 dataset with a Pytorch CNN

NOTE: This is a tutorial that illustrates how to use torch training toolkit. So we do
not focus on how to optimize model performance or loading data - topics such as regularization,
dropout etc. have been dropped. The intention is to understand how to use torch training toolkit
to ease the training process & not on how to optimize model performance.

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings
import logging
import logging.config

warnings.filterwarnings("ignore")

import sys
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print

# tweaks for libraries
np.set_printoptions(precision=4, linewidth=1024, suppress=True)
plt.style.use("seaborn-v0_8")
sns.set(style="darkgrid", context="notebook", font_scale=1.20)

# Pytorch imports
import torch

print("Using Pytorch version: ", torch.__version__)
import torch.nn as nn
import torchmetrics
import torchsummary
from torchvision import datasets, transforms

# import the Pytorch training toolkit (t3)
import torch_training_toolkit as t3

# to ensure that you get consistent results across runs & machines
seed = t3.seed_all(41)

logger = t3.get_logger(pathlib.Path(__file__), level=logging.INFO)

# define a device to train on
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = pathlib.Path(__file__).parent.parent / "data"

logger.info(f"Will train model on {DEVICE}")
logger.info(f"DATA_PATH = {DATA_PATH}")

MEANS, STDS = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def get_mean_and_std():
    # download as Numpy arrays
    train_dataset = datasets.CIFAR10(
        root=DATA_PATH,
        train=True,
        download=True,
    )
    data = train_dataset.data / 255.0
    means = data.mean(axis=(0, 1, 2))
    stds = data.std(axis=(0, 1, 2))
    print(f"get_mean_and_std(): means: {means} - stds: {stds}", flush=True)
    return means, stds


def load_data(args):
    """
    load the data using datasets API. We also split the test_dataset into
    cross-val/test datasets using 80:20 ratio
    """

    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEANS, STDS),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=transformations,
    )
    print(f"Size of downloaded training dataset: {len(train_dataset)}")

    test_dataset = datasets.CIFAR10(
        DATA_PATH,
        train=False,
        download=True,
        transform=transformations,
    )
    print(f"Size of downloaded test dataset: {len(test_dataset)}")

    # split the train dataset into train/cross-val datasets
    train_dataset, val_dataset = t3.split_dataset(train_dataset, split_perc=0.2)
    print("No of cross-val records: %d" % len(val_dataset))
    print("No of test records: %d" % len(test_dataset))

    return train_dataset, val_dataset, test_dataset


def denormalize(tensor: torch.Tensor, mean: tuple, std: tuple) -> torch.Tensor:
    """
    denormalizes (reverses torchvision.transforms.Normalize()) a batch of torch tensors
    with mean & std (i.e. performs (tensor * std) / mean)

    Args:
        tensor (torch.Tensor): a batch of tensors
        mean (tuple): mean used for normalizaton (e.g. (0.5, 0.5, 0.5))
        std (tuple): standard deviation for normalizaton (e.g. (0.5, 0.5, 0.5))

    Returns:
        denormalized torch.Tensor with same shape as parameter passed
    """

    # create mean & std in the same shape as tensor passed in
    mean_t = torch.Tensor(mean, device=tensor.device).view(1, -1, 1, 1)
    std_t = torch.Tensor(std, device=tensor.device).view(1, -1, 1, 1)

    # denormalize
    return tensor.mul(std_t).add(mean_t).clamp(0, 1)


def display_sample(
    sample_images,
    sample_labels,
    grid_shape=(10, 10),
    plot_title=None,
    sample_predictions=None,
):
    """
    displays grid of images with labels on top of each image
    """
    # just in case these are not imported!
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("seaborn-v0_8")

    num_rows, num_cols = grid_shape
    assert sample_images.shape[0] == num_rows * num_cols

    # a dict to help encode/decode the labels
    FASHION_LABELS = {
        0: "Airplane",
        1: "Auto",
        2: "Bird",
        3: "Cat",
        4: "Deer",
        5: "Dog",
        6: "Frog",
        7: "Horse",
        8: "Ship",
        9: "Truck",
    }

    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=0.98)
        sns.set_style(
            {
                "font.sans-serif": [
                    "SF Pro Display",
                    "Calibri",
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
            gridspec_kw={"wspace": 0.05, "hspace": 0.35},
            squeeze=True,
        )  # 0.03, 0.25
        # fig = ax[0].get_figure()
        f.tight_layout()
        f.subplots_adjust(top=0.90)  # 0.93

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")
                # # de-normalize image
                image = sample_images[image_index].transpose((1, 2, 0))
                # image = (image * STDS) + MEANS

                # show selected image
                # @see: https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa
                # to avoid the "clipping input data..." message\
                # image = image / np.amax(image)
                image = np.clip(image, 0, 1)
                ax[r, c].imshow(
                    image,
                    cmap="Greys",
                    interpolation="nearest",
                )

                if sample_predictions is None:
                    # show the text label as image title
                    title = ax[r, c].set_title(
                        f"{FASHION_LABELS[sample_labels[image_index]]}"
                    )
                else:
                    pred_matches_actual = (
                        sample_labels[image_index] == sample_predictions[image_index]
                    )
                    # show prediction from model as image title
                    title = "%s" % FASHION_LABELS[sample_predictions[image_index]]
                    if pred_matches_actual:
                        # if matches, title color is green
                        title_color = "g"
                    else:
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
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 32, 32, 3, 10
# a dict to help encode/decode the labels
CIFAR10_LABELS = {
    0: "Airplane",
    1: "Auto",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck",
}


# define our network using Conv2d and Linear layers
class Cifar10Net(nn.Module):
    """our base model (using CNNs)"""

    def __init__(self):
        # fmt: off
        super(Cifar10Net, self).__init__()
        self.net = nn.Sequential(
            # NOTE: t3.Conv2d(...) is almost the same as nn.Conv2d
            # the t3 version initialises weights & biases and uses padding=1 by default
            t3.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            t3.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2),

            t3.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            t3.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.3),
            
            t3.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            t3.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.4),
            
            nn.Flatten(),
            t3.Linear(4 * 4 * 128, NUM_CLASSES),
        )
        # fmt: on

    def forward(self, x):
        return self.net(x)


from cmd_opts import TrainingArgsParser  # parse_command_line


def main():
    # setup command line parser
    # args = parse_command_line()
    parser = TrainingArgsParser()
    parser.add_argument(
        "--use_cnn",
        dest="use_cnn",
        action="store_true",
        help="Flag to choose CNN model over ANN",
    )
    parser.set_defaults(use_cnn=False)  # don't use CNN by default
    args = parser.parse_args()

    MODEL_SAVE_NAME = "pyt_cifar10_cnn.pyt"
    MODEL_SAVE_PATH = os.path.join("..", "model_states", MODEL_SAVE_NAME)

    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = load_data(args)

    # declare loss functions
    loss_fn = nn.CrossEntropyLoss()
    # metrics map - metrics to track during training
    # NOTE: loss is always tracked via the loss function
    metrics_map = {
        # accuracy
        "acc": torchmetrics.classification.MulticlassAccuracy(num_classes=NUM_CLASSES),
        # overall f1-score
        # "f1": torchmetrics.classification.MulticlassF1Score(num_classes = NUM_CLASSES)
    }
    # define the trainer
    trainer = t3.Trainer(
        loss_fn=loss_fn,
        device=DEVICE,
        epochs=args.epochs,
        batch_size=args.batch_size,
        metrics_map=metrics_map,
    )

    if args.show_sample:
        # display sample from test dataset
        print("Displaying sample from test dataset...")
        testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=True
        )
        data_iter = iter(testloader)
        images, labels = next(data_iter)  # fetch first batch of 64 images & labels
        # de-normalize batch of images
        images = t3.denormalize_and_permute_images(images, MEANS, STDS)
        t3.display_images_grid(
            images.cpu().numpy(),
            labels.cpu().numpy(),
            grid_shape=(8, 8),
            plot_title="Sample Images",
            labels_dict=CIFAR10_LABELS,
        )

    if args.train:
        model = Cifar10Net()
        model = model.to(DEVICE)
        # display Keras like summary
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))
        # optimizer is required only during training!
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=args.lr, weight_decay=args.l2_reg
        )
        # add L1 regularization & a learning scheduler
        from torch.optim.lr_scheduler import StepLR

        scheduler = StepLR(optimizer, step_size=args.epochs // 5, gamma=0.1)
        # add an EarlyStopping
        early_stopping = t3.EarlyStopping(
            model,
            metrics_map,
            # using_val_dataset=True,
            monitor="val_loss",
            patience=5,
        )
        hist = trainer.fit(
            model,
            optimizer,
            train_dataset,
            validation_dataset=val_dataset,
            lr_scheduler=scheduler,
            # early_stopping=early_stopping,
            logger=logger,
        )
        # display the tracked metrics
        hist.plot_metrics("Model Performance")
        # save model state
        t3.save_model(model, MODEL_SAVE_PATH)
        del model

    if args.eval:
        # evaluate model performance on train/eval & test datasets
        print("Evaluating model performance...")
        logger.debug("Evaluating model performance...")

        # load model state from .pt file
        model = Cifar10Net()
        model = t3.load_model(model, MODEL_SAVE_PATH).to(DEVICE)
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

        metrics = trainer.evaluate(model, train_dataset)
        print(
            f"  Training dataset  -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}"
        )
        logger.debug(
            f"  Training dataset  -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}"
        )
        metrics = trainer.evaluate(model, val_dataset)
        print(
            f"  Cross-val dataset -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}"
        )
        logger.debug(
            f"  Cross-val dataset -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}"
        )
        metrics = trainer.evaluate(model, test_dataset)
        print(
            f"  Testing dataset   -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}"
        )
        logger.debug(
            f"  Testing dataset   -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}"
        )
        del model

    if args.pred:
        # load model state from .pt file
        model = Cifar10Net()
        model = t3.load_model(model, MODEL_SAVE_PATH).to(DEVICE)
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

        y_pred, y_true = trainer.predict(model, test_dataset)
        y_pred = np.argmax(y_pred, axis=1)
        print("Sample labels (50): ", y_true[:50])
        print("Sample predictions: ", y_true[:50])
        print("We got %d/%d incorrect!" % ((y_pred != y_true).sum(), len(y_true)))

        # display sample from test dataset
        print("Displaying sample predictions...")
        testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=True
        )
        data_iter = iter(testloader)
        images, labels = next(data_iter)
        preds = np.argmax(trainer.predict(model, images.cpu().numpy()), axis=1)
        display_sample(
            images.cpu().numpy(),
            labels.cpu().numpy(),
            sample_predictions=preds,
            grid_shape=(8, 8),
            plot_title="Sample Predictions",
        )


if __name__ == "__main__":
    main()

# ---------------------------------------------------------
# Results:
#   CNN with epochs=25, batch-size=128, LR=0.001
#       Training  -> loss: 0.2491 acc: 91.56%
#       Cross-val -> loss: 0.4536 acc: 83.75%
#       Testing   -> loss: 0.4723 acc: 83.44%
# The model is overfitting & under-performing
# --------------------------------------------------
