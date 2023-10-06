# -*- coding: utf-8 -*-
"""
pyt_fashion_mnist_dnn.py: Multiclass classification of Zolando's Fashion MNIST dataset.
Part-1 of the tutorial for Torch Training Toolkit (T3), where we setup the basic training
harness & fit our model, evaluate results and run predictions.

Step01 - setting up the basic harness to train/evaluate/test a Pytorch model

NOTE: This is a tutorial that illustrates how to use Torch Training Toolkit (T3). So we do
not focus on how to optimize model performance or loading data - topics such as regularization,
dropout etc. have been dropped. The intention is to understand how to use Torch Training Toolkit
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
import torchsummary
from torchvision import datasets, transforms

# import the Pytorch training toolkit (t3)
import torch_training_toolkit as t3

# to ensure that you get consistent results across runs & machines
seed = 123
t3.seed_all(seed)

logger = logging.getLogger(__name__)

# define a device to train on
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = pathlib.Path(__file__).parent.parent / "data"
IMAGES_MEAN, IMAGES_STD = 0.5, 0.5

logger.info(f"Will train model on {DEVICE}")
logger.info(f"DATA_PATH = {DATA_PATH}")


def load_data(args):
    """
    load the data using datasets API. We also split the test_dataset into
    cross-val/test datasets using 80:20 ration
    """
    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGES_MEAN, IMAGES_STD),
        ]
    )

    train_dataset = datasets.FashionMNIST(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=transformations,
    )

    print("No of training records: %d" % len(train_dataset))

    test_dataset = datasets.FashionMNIST(
        DATA_PATH,
        train=False,
        download=True,
        transform=transformations,
    )
    print("No of test records: %d" % len(test_dataset))

    # lets split the test dataset into val_dataset & test_dataset -> 8000:2000 records
    # val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [8000, 2000])
    val_dataset, test_dataset = t3.split_dataset(test_dataset, split_perc=args.test_split)
    print("No of cross-val records: %d" % len(val_dataset))
    print("No of test records: %d" % len(test_dataset))

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

    # plt.style.use("seaborn")

    num_rows, num_cols = grid_shape
    assert sample_images.shape[0] == num_rows * num_cols

    # a dict to help encode/decode the labels
    FASHION_LABELS = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
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
                # de-normalize image
                sample_images[image_index] = (sample_images[image_index] * IMAGES_STD) + IMAGES_MEAN

                # show selected image
                ax[r, c].imshow(
                    sample_images[image_index].squeeze(),
                    cmap="Greys",
                    interpolation="nearest",
                )

                if sample_predictions is None:
                    # show the text label as image title
                    title = ax[r, c].set_title(f"{FASHION_LABELS[sample_labels[image_index]]}")
                else:
                    pred_matches_actual = sample_labels[image_index] == sample_predictions[image_index]
                    # show prediction from model as image title
                    title = "%s" % FASHION_LABELS[sample_predictions[image_index]]
                    if pred_matches_actual:
                        # if matches, title color is green
                        title_color = "g"
                    else:
                        # else title color is red
                        # title = '%s/%s' % (FASHION_LABELS[sample_labels[image_index]],
                        #                    FASHION_LABELS[sample_predictions[image_index]])
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
class FMNISTNet(nn.Module):
    def __init__(self):
        super(FMNISTNet, self).__init__()
        # our network - using Sequential API
        self.net = nn.Sequential(
            nn.Flatten(),
            t3.Linear(IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS, 256),
            nn.ReLU(),
            t3.Linear(256, 128),
            nn.ReLU(),
            t3.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


# if you prefer to use Convolutional Neural Network, use the following model definition
class FMNISTConvNet(nn.Module):
    def __init__(self):
        super(FMNISTConvNet, self).__init__()
        self.net = nn.Sequential(
            t3.Conv2d(NUM_CHANNELS, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            t3.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Flatten(),
            t3.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            t3.Linear(512, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def main():
    # setup command line parser
    # args = parse_command_line()
    parser = t3.TrainingArgsParser()
    parser.add_argument(
        "--use_cnn",
        dest="use_cnn",
        action="store_true",
        help="Flag to choose CNN model over ANN",
    )
    parser.set_defaults(use_cnn=False)  # don't use CNN by default
    args = parser.parse_args()

    MODEL_SAVE_NAME = "pyt_mnist_cnn.pyt" if args.use_cnn else "pyt_mnist_dnn.pyt"
    MODEL_SAVE_PATH = os.path.join("..", "model_states", MODEL_SAVE_NAME)

    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = load_data(args)

    # declare loss functions
    loss_fn = nn.CrossEntropyLoss()
    # define the trainer
    trainer = t3.Trainer(
        loss_fn=loss_fn,
        device=DEVICE,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    if args.show_sample:
        # display sample from test dataset
        print("Displaying sample from train dataset...")
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        data_iter = iter(testloader)
        images, labels = next(data_iter)  # fetch first batch of 64 images & labels
        display_sample(
            images.cpu().numpy(),
            labels.cpu().numpy(),
            grid_shape=(8, 8),
            plot_title="Sample Images",
        )

    if args.train:
        print(f'Using {"CNN" if args.use_cnn else "ANN"} model...')
        model = FMNISTConvNet() if args.use_cnn else FMNISTNet()
        # use Pytorch 2 compile() call to speed up model - does not work on Windows :(
        # if (int(torch.__version__.split(".")[0]) >= 2) and (sys.platform != "win32"):
        #     torch.compile(model)
        model = model.to(DEVICE)
        # display Keras like summary
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))
        # optimizer is required only during training!
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # train model -> will return metrics tracked across epochs (default - only loss metrics
        hist = trainer.fit(
            model,
            optimizer,
            train_dataset,
            validation_dataset=val_dataset,
        )
        # display the tracked metrics
        hist.plot_metrics("Model Performance")

        # evaluate model performance on train/eval & test datasets
        print("Evaluating model performance...")
        metrics = trainer.evaluate(model, train_dataset)
        print(f"  Training dataset  -> loss: {metrics['loss']:.4f}")
        metrics = trainer.evaluate(model, val_dataset)
        print(f"  Cross-val dataset  -> loss: {metrics['loss']:.4f}")
        metrics = trainer.evaluate(model, test_dataset)
        print(f"  Test dataset  -> loss: {metrics['loss']:.4f}")

        # save model state
        t3.save_model(model, MODEL_SAVE_PATH)
        del model

    if args.pred:
        # load model state from .pt file
        model = FMNISTConvNet() if args.use_cnn else FMNISTNet()
        model = t3.load_model(model, MODEL_SAVE_PATH)
        if (int(torch.__version__.split(".")[0]) >= 2) and (sys.platform != "win32"):
            torch.compile(model)
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

        y_pred, y_true = trainer.predict(model, test_dataset)
        y_pred = np.argmax(y_pred, axis=1)
        print("Sample labels (50): ", y_true[:50])
        print("Sample predictions: ", y_true[:50])
        print("We got %d/%d incorrect!" % ((y_pred != y_true).sum(), len(y_true)))

        # display sample from test dataset
        print("Displaying sample predictions...")
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
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
