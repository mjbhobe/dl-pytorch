# -*- coding: utf-8 -*-
"""
pyt_cifar10_resnet18.py - training a resnet18 model on CIFAR10 dataset using Pytorch

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
logging.config.fileConfig(fname="logging.config")

import sys
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
from torchvision import datasets, transforms, models

# import the Pytorch training toolkit (t3)
import torch_training_toolkit as t3

# to ensure that you get consistent results across runs & machines
seed = 123
t3.seed_all(seed)

logger = t3.get_logger(pathlib.Path(__file__), level=logging.INFO)

# define a device to train on
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = pathlib.Path(__file__).parent.parent / "data"

logger.info(f"Will train model on {DEVICE}")
logger.info(f"DATA_PATH = {DATA_PATH}")

# some globals
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 32, 32, 3, 10
MEANS, STDS = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def load_data(args):
    """
    load the data using datasets API. We also split the test_dataset into
    cross-val/test datasets using 80:20 ration
    """
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(MEANS, STDS),
        ]
    )

    valid_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEANS, STDS),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=train_transforms,
    )

    print("No of training records downloaded: %d" % len(train_dataset))

    test_dataset = datasets.CIFAR10(
        DATA_PATH,
        train=False,
        download=True,
        transform=valid_transforms,
    )
    print("No of test records downloaded: %d" % len(test_dataset))

    val_dataset, test_dataset = t3.split_dataset(
        test_dataset, split_perc=args.test_split
    )
    print(f"No of cross-val records: {len(val_dataset)}")
    print(f"No of test records: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def denormalize_and_permute_images(images: torch.Tensor) -> torch.Tensor:
    mu = torch.Tensor(MEANS)
    std = torch.Tensor(STDS)
    images_x = images * std[:, None, None] + mu[:, None, None]
    images_x = images_x.clamp(0, 1)
    images_x = images_x.permute(0, 2, 3, 1)
    return images_x


def display_sample(
    sample_images,  # NOTE: images are de-normalized!!
    sample_labels,
    grid_shape=(10, 10),
    fig_size=(14, 10),
    plot_title=None,
    sample_predictions=None,
):
    # just in case these are not imported!
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("seaborn-v0_8")

    num_rows, num_cols = grid_shape
    assert sample_images.shape[0] == num_rows * num_cols

    # a dict to help encode/decode the labels
    CIFAR10_LABELS = {
        0: "Plane",
        1: "Auto",
        2: "Bird",
        3: "Cat",
        4: "Deer",
        5: "Dog",
        6: "Frog",
        7: "Horse",
        8: "Sheep",
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
            figsize=fig_size,
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
                # de-normalize image - transforms.Normalize() does (img - mean) / std
                # sample_images[image_index] = (sample_images[image_index] * STDS) + MEANS

                # show selected image
                ax[r, c].imshow(
                    sample_images[image_index],
                    cmap="Greys",
                    interpolation="nearest",
                )

                if sample_predictions is None:
                    # show the text label as image title
                    title = ax[r, c].set_title(
                        f"{CIFAR10_LABELS[sample_labels[image_index]]}"
                    )
                else:
                    pred_matches_actual = (
                        sample_labels[image_index] == sample_predictions[image_index]
                    )
                    # show prediction from model as image title
                    title = "%s" % CIFAR10_LABELS[sample_predictions[image_index]]
                    if pred_matches_actual:
                        # if matches, title color is green
                        title_color = "g"
                    else:
                        # else title color is red
                        # title = '%s/%s' % (CIFAR10_LABELS[sample_labels[image_index]],
                        #                    CIFAR10_LABELS[sample_predictions[image_index]])
                        title_color = "r"

                    # but show the prediction in the title
                    title = ax[r, c].set_title(title)
                    # if prediction is incorrect title color is red, else green
                    plt.setp(title, color=title_color)

        if plot_title is not None:
            plt.suptitle(plot_title)
        plt.show()
        plt.close()


def build_model(pretrained=True, fine_tune=False, num_classes=NUM_CLASSES):
    """
    download the resnet18 pre-trained model, if requested, freeze weights &
    replace the linear section of model
    :param pretrained: if True, download pre-trained model
    :param fine_tune: if True, un-freeze all weights so model traines across epochs
       else, just train the final linear layer
    :param num_classes: number of classes for final layer output
    :return: the resnet18 model
    """
    model = models.resnet18(pretrained=pretrained)
    for params in model.parameters():
        params.requires_grad = fine_tune
    # now add the fully-connected layer
    model.fc = nn.Linear(512, num_classes)
    print(model)
    return model


from cmd_opts import TrainingArgsParser  # parse_command_line


def main():
    # setup command line parser
    parser = TrainingArgsParser()
    args = parser.parse_args()

    MODEL_SAVE_NAME = "pyt_resnet18_cifar10.pt"
    MODEL_SAVE_PATH = os.path.join("..", "model_states", MODEL_SAVE_NAME)

    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = load_data(args)

    # loss function for multi-class classification
    loss_fn = nn.CrossEntropyLoss()
    # metrics map - metrics to track during training
    # NOTE: loss is always tracked via the loss function
    metrics_map = {
        # accuracy
        "acc": torchmetrics.classification.MulticlassAccuracy(num_classes=NUM_CLASSES),
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
        print("Displaying sample from train dataset...")
        testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=True
        )
        data_iter = iter(testloader)
        images, labels = next(data_iter)  # fetch first batch of 64 images & labels
        images = denormalize_and_permute_images(images)
        # images = images.permute(0, 2, 3, 1)
        display_sample(
            images.cpu().numpy(),
            labels.cpu().numpy(),
            grid_shape=(8, 8),
            plot_title="Sample Images",
        )

    if args.train:
        model = build_model()
        model = model.to(DEVICE)
        # display Keras like summary
        # print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))
        # optimizer is required only during training!
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.l2_reg
        )
        # # add L1 regularization & a learning scheduler
        # from torch.optim.lr_scheduler import StepLR
        #
        # scheduler = StepLR(optimizer, step_size=args.epochs // 5, gamma=0.1)
        # # add an EarlyStopping
        # early_stopping = t3.EarlyStopping(
        #     model,
        #     metrics_map,
        #     using_val_dataset=True,
        #     monitor="val_loss",
        #     patience=5,
        # )
        hist = trainer.fit(
            model,
            optimizer,
            train_dataset,
            validation_dataset=val_dataset,
            # lr_scheduler=scheduler,
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
        model = build_model()
        model = t3.load_model(model, MODEL_SAVE_PATH)
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
        model = build_model()
        model = t3.load_model(model, MODEL_SAVE_PATH)
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
        images = denormalize_and_permute_images(images)
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
#   ANN with epochs=50, batch-size=32, LR=0.001
#       Training  -> acc: 90.88%
#       Cross-val -> acc: 87.92%
#       Testing   -> acc: 87.50%
#   CNN with epochs=25, batch-size=32, LR=0.001
#       Training  -> acc: 95.34%
#       Cross-val -> acc: 92.01%
#       Testing   -> acc: 92.09%
# Clearly the CNN performs better than the MLP
# However, both models are over-fitting
# --------------------------------------------------
