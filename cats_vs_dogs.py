#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cats_vs_dogs.py: binary image classification with Pytorch

@author: Manish Bhobe
My experiments with Python, Data Science, ML & Deep Learning
This code is meant for learning purposes only!!
"""

import warnings

warnings.filterwarnings("ignore")

import os
import sys
import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from PIL import Image

# Pytorch imports
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch import optim
import torchsummary
import torchmetrics

# My helper functions for training/evaluating etc.
# import pytorch_toolkit as pytk
import torch_training_toolkit as t3

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use("seaborn-v0_8")
sns.set(style="darkgrid", context="notebook", font_scale=1.20)

SEED = t3.seed_all()

# hyper parameters
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 224, 224, 3, 2
NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, L2_REG = 5, 64, 1e-3, 0.001
MODEL_SAVE_NAME = "cat_or_dog.pyt"
# MODEL_SAVE_PATH = os.path.join("./model_states", MODEL_SAVE_NAME)
MODEL_SAVE_PATH = pathlib.Path(__file__).parent / "model_states" / MODEL_SAVE_NAME

"""
NOTE: data for this example has been downloaded from Kaggle
    $> mkdir -p ./data/cat_or_dog
    $> kaggle datasets download -d tongpython/cat-and-dog -p ./data/cat_or_dog
    $> unzip -d ./data/cat_or_dog -o ./data/cat-and-dog.zip
"""
IMAGES_BASE_DIR = pathlib.Path(__file__).parent / "data" / "cat_or_dog"
# THIS_DIR = os.path.dirname(__file__)
TRAIN_IMAGES_PATH = IMAGES_BASE_DIR / "training_set"
TEST_IMAGES_PATH = IMAGES_BASE_DIR / "test_set"


def show_sample(images_path=TRAIN_IMAGES_PATH, num_count=10, num_rows=2):
    """shows a random sample of num_count/2 cat & num_count/2 dog images"""
    num_count = num_count if (num_count % 2 == 0) else (num_count + 1)
    num_cols = num_count // num_rows
    num_cols = num_cols if (num_count % num_rows == 0) else num_cols + 1
    image_count = num_count // 2  # half cat & half dog images
    cat_image_paths = glob.glob(images_path + "/cats/*.jpg")
    dog_image_paths = glob.glob(images_path + "/dogs/*.jpg")
    rand_indexes = np.random.choice(np.arange(len(cat_image_paths)), image_count)
    sample_image_cats = [cat_image_paths[i] for i in rand_indexes]
    rand_indexes = np.random.choice(np.arange(len(dog_image_paths)), image_count)
    sample_image_dogs = [dog_image_paths[i] for i in rand_indexes]
    all_images = sample_image_cats + sample_image_dogs
    random.shuffle(all_images)
    # plt.figure(figsize=(15,15))
    fig, ax = plt.subplots(num_rows, num_cols)
    for row in range(num_rows):
        for col in range(num_cols):
            ax[row, col].axis("off")
            index = row * num_cols + col
            if index < num_count:
                image = Image.open(all_images[index])
                ax[row, col].imshow(image)
    plt.suptitle(f"Random sample of {num_count} images")
    plt.show()


MEAN_NUMS = [0.485, 0.456, 0.406]
STD_NUMS = [0.229, 0.224, 0.225]


def display_sample(
    images, labels, predictions=None, fig_size=None, num_cols=10, plot_title=None
):
    num_images = len(images)
    num_rows = num_images // num_cols
    num_rows = num_rows if (num_images % num_cols == 0) else num_rows + 1

    label_lookup = {0: "Cat", 1: "Dog"}

    figsize = (22, 22) if fig_size is None else fig_size

    with sns.axes_style("whitegrid"):
        ctx = sns.plotting_context()
        ctx["axes.labelsize"] = 10.0
        ctx["axes.titlesize"] = 12.0
        ctx["font.size"] = 12.0
        sns.set(context="notebook", font_scale=1.0)
        sns.set_style(
            {
                "font.sans-serif": [
                    "SF Pro Display",
                    "Verdana",
                    "Arial",
                    "Calibri",
                    "DejaVu Sans",
                ]
            }
        )

        # plt.figure(figsize=figsize)
        fig, ax = plt.subplots(
            num_rows,
            num_cols,
            figsize=figsize,
            gridspec_kw={"wspace": 0.02, "hspace": 0.25},
            squeeze=True,
        )
        fig.tight_layout()
        for row in range(num_rows):
            for col in range(num_cols):
                ax[row, col].axis("off")
                index = row * num_cols + col
                if index < num_images:
                    image = images[index]
                    image = image.transpose((1, 2, 0))
                    mean = np.array([MEAN_NUMS])
                    std = np.array([STD_NUMS])
                    # de-normalize image
                    image = std * image + mean
                    image = np.clip(image, 0, 1)
                    ax[row, col].imshow(image)
                    if predictions is not None:
                        title = (
                            label_lookup[labels[index]]
                            if predictions[index] == labels[index]
                            else label_lookup[predictions[index]]
                            + "/"
                            + label_lookup[labels[index]]
                        )
                    else:
                        title = label_lookup[labels[index][0]]
                    ax[row, col].set_title(title)
        if plot_title is not None:
            plt.suptitle(plot_title)

    plt.show()
    plt.close()


xforms = {
    "train": transforms.Compose(
        [
            # transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
            transforms.RandomResizedCrop((IMAGE_WIDTH, IMAGE_HEIGHT)),
            # transforms.RandomAffine(
            #     degrees=(-30, 30), translate=(0.25, 0.35), scale=(0.5, 1.5), shear=0.30
            # ),
            transforms.RandomChoice(
                [
                    transforms.RandomAutocontrast(),
                    transforms.ColorJitter(
                        brightness=0.3, contrast=0.5, saturation=0.1, hue=0.1
                    ),
                    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine(
                        degrees=(-30, 30),
                        translate=(0.25, 0.35),
                        scale=(0.5, 1.5),
                        shear=0.30,
                    ),
                ]
            ),
            transforms.CenterCrop((IMAGE_WIDTH, IMAGE_HEIGHT)),
            transforms.RandomHorizontalFlip(0.30),
            # transforms.RandomVerticalFlip(0.30),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_NUMS, STD_NUMS),
        ]
    ),
    "val_or_test": transforms.Compose(
        [
            transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
            transforms.CenterCrop((IMAGE_WIDTH, IMAGE_HEIGHT)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_NUMS, STD_NUMS),
        ]
    ),
}


class CatOrDogDataset(Dataset):
    def __init__(self, base_path, transforms=None):
        self.cats = glob.glob(base_path + "/cats/*.jpg")
        self.dogs = glob.glob(base_path + "/dogs/*.jpg")
        self.fpaths = self.cats + self.dogs
        # shuffle paths
        random.shuffle(self.fpaths)
        # dog == 1, cat == 0
        self.labels = np.array(
            [
                int(fpath.split(os.path.sep)[-1].startswith("dog"))
                for fpath in self.fpaths
            ],
            dtype="float32",
        ).reshape(-1, 1)
        # print(self.labels[:10], flush=True)
        self.transforms = transforms
        # assert (self.transforms is not None)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.fpaths[idx])
        if self.transforms is not None:
            image = self.transforms(image)
        label = torch.tensor(self.labels[idx])
        # print(f"Index {idx} - image.shape: {image.shape} - label.shape: {label.shape}", flush=True)
        return image, label


def get_datasets(test_perc=0.2):
    assert os.path.exists(
        TRAIN_IMAGES_PATH
    ), f"FATAL: Training images path {TRAIN_IMAGES_PATH} does not exist!"
    assert os.path.exists(
        TEST_IMAGES_PATH
    ), f"FATAL: Test images path {TEST_IMAGES_PATH} does not exist!"
    assert xforms["train"] is not None
    assert xforms["val_or_test"] is not None
    train_dataset = CatOrDogDataset(str(TRAIN_IMAGES_PATH), transforms=xforms["train"])
    test_dataset = CatOrDogDataset(
        str(TEST_IMAGES_PATH), transforms=xforms["val_or_test"]
    )
    num_images = len(test_dataset)
    num_test_images = int(test_perc * num_images)
    num_val_images = num_images - num_test_images
    print(
        f"get_data(): {len(test_dataset)} test images split to {num_val_images} eval & {num_test_images} test images"
    )
    val_dataset, test_dataset = torch.utils.data.random_split(
        test_dataset, [num_val_images, num_test_images]
    )
    return train_dataset, val_dataset, test_dataset


def build_model(lr=LEARNING_RATE):
    def conv_layer(ino, out, kernel_size, stride=1):
        return nn.Sequential(
            nn.Conv2d(ino, out, kernel_size, stride),
            nn.ReLU(),
            nn.BatchNorm2d(out),
            nn.MaxPool2d(2),
        )

    net = nn.Sequential(
        conv_layer(3, 64, 3),
        conv_layer(64, 512, 3),
        # conv_layer(512, 512, 3),
        # conv_layer(512, 512, 3),
        # conv_layer(512, 512, 3),
        # conv_layer(512, 512, 3),
        nn.Flatten(),
        nn.Linear(512 * 54 * 54, 1),
        nn.Sigmoid(),  # binary classification
    )

    # model = pytk.PytkModuleWrapper(net)
    # loss_fn = nn.CrossEntropyLoss()  # nn.BCELoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # model.compile(loss=loss_fn, optimizer=optimizer, metrics=["acc"])
    # return model, optimizer
    return net


# TRAIN_MODEL = True
# PREDICT_MODEL = True
# DISPLAY_SAMPLE = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Model performance:
    Training:
        > Training data   -> loss: 19.1744 - acc: 0.5549
        > Validation data -> loss: 18.1420 - acc: 0.5961
        > Test data       -> loss: 20.0158 - acc: 0.6034
"""


def main():
    parser = t3.TrainingArgsParser()
    # add more command line args here, if needed
    args = parser.parse_args()

    print("Using Pytorch version: ", torch.__version__)

    train_dataset, val_dataset, test_dataset = get_datasets()
    print(
        f"train_dataset: {len(train_dataset)} records - val_dataset: {len(val_dataset)} records - "
        f"test_dataset: {len(test_dataset)} records"
    )

    # declare loss functions
    loss_fn = nn.BCELoss()
    # tracked metrics
    metrics_map = {
        "acc": torchmetrics.classification.BinaryAccuracy(),
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
        # show random sample from test_dataset
        print("Displaying random sample of images from test dataset...")
        testloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True
        )
        data_iter = iter(testloader)
        images, labels = next(data_iter)  # fetch a batch of 64 random images
        display_sample(
            images.cpu().numpy(),
            labels.cpu().numpy(),
            num_cols=8,
            plot_title="Sample Images from test dataset",
        )

    if args.train:
        # build the model
        model = build_model()
        model = model.to(DEVICE)
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))
        # optimizer is required only during training!
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # train model -> will return metrics tracked across epochs (default - only loss metrics
        hist = trainer.fit(
            model,
            optimizer,
            train_dataset,
            validation_dataset=val_dataset,
            seed=SEED,
            num_workers=2,
            # logger=logger,
        )
        # display the tracked metrics
        hist.plot_metrics("Model Performance")
        # save model state
        t3.save_model(model, MODEL_SAVE_PATH)

        # # train the model
        # print("Training model...")
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        # hist = model.fit_dataset(
        #     train_dataset,
        #     validation_dataset=val_dataset,
        #     epochs=NUM_EPOCHS,
        #     batch_size=BATCH_SIZE,
        #     lr_scheduler=scheduler,
        #     num_workers=5,
        # )
        # pytk.show_plots(hist, metric="acc")

        # # evaluate performance
        # print("Evaluating performance...")
        # loss, acc = model.evaluate_dataset(train_dataset, batch_size=BATCH_SIZE)
        # print(f"Training data   -> loss: {loss:.4f} - acc: {acc:.4f}")
        # loss, acc = model.evaluate_dataset(val_dataset, batch_size=BATCH_SIZE)
        # print(f"Validation data -> loss: {loss:.4f} - acc: {acc:.4f}")
        # loss, acc = model.evaluate_dataset(test_dataset, batch_size=BATCH_SIZE)
        # print(f"Test data       -> loss: {loss:.4f} - acc: {acc:.4f}")

        # model.save(MODEL_SAVE_PATH)
        del model

    if args.eval:
        print("Evaluating model performance...")

        model = build_model()
        model = t3.load_model(model, MODEL_SAVE_PATH)
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

        # # load model from save path
        # print(f"Loading model from {MODEL_SAVE_PATH}")
        # model, optimizer = build_model()
        # model.load(MODEL_SAVE_PATH)
        # model.summary((NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))

        # evaluate performance
        # print("Evaluating preformance...")
        # loss, acc = model.evaluate_dataset(train_dataset, batch_size=BATCH_SIZE)
        # print(f"Training data   -> loss: {loss:.4f} - acc: {acc:.4f}")
        # loss, acc = model.evaluate_dataset(val_dataset, batch_size=BATCH_SIZE)
        # print(f"Validation data -> loss: {loss:.4f} - acc: {acc:.4f}")
        # loss, acc = model.evaluate_dataset(test_dataset, batch_size=BATCH_SIZE)
        # print(f"Test data       -> loss: {loss:.4f} - acc: {acc:.4f}")
        metrics = trainer.evaluate(model, train_dataset)
        print(
            f"  Training dataset  -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}"
        )
        metrics = trainer.evaluate(model, val_dataset)
        print(
            f"  Cross-val dataset  -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}"
        )
        metrics = trainer.evaluate(model, test_dataset)
        print(
            f"  Test dataset  -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}"
        )

        del model

    if args.pred:
        # run predictions
        print("Running predictions...")

        model = build_model()
        model = t3.load_model(model, MODEL_SAVE_PATH)
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

        y_pred, y_true = model.predict_dataset(test_dataset)
        y_pred = np.argmax(y_pred, axis=1)
        print("Sample labels (50): ", y_true[:50])
        print("Sample predictions: ", y_true[:50])
        print("We got %d/%d incorrect!" % ((y_pred != y_true).sum(), len(y_true)))

        # display sample from test dataset
        print("Displaying sample predictions...")
        testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=256, shuffle=True
        )
        data_iter = iter(testloader)
        images, labels = next(data_iter)  # fetch a batch of 64 random images
        preds = np.argmax(model.predict(images), axis=1)
        display_sample(
            images.cpu().numpy(),
            labels.cpu().numpy(),
            predictions=preds,
            num_cols=8,
            plot_title="Sample Images & predictions from test dataset",
        )
        del model


if __name__ == "__main__":
    main()
