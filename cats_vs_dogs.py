#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cats_vs_dogs.py: binary image classification with Pytorch

@author: Manish Bhobe
My experiments with Python, Data Science, ML & Deep Learning
This code is meant for learning purposes only!!
"""

import warnings

warnings.filterwarnings('ignore')

import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from PIL import Image

# Pytorch imports
import torch

print('Using Pytorch version: ', torch.__version__)
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch import optim

# My helper functions for training/evaluating etc.
import pytorch_toolkit as pytk

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use('seaborn')
sns.set(style='darkgrid', context='notebook', font_scale=1.20)

SEED = pytk.seed_all()

# hyper parameters
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 224, 224, 3, 2
NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, L2_REG = 5, 32, 1e-3, 0.001
MODEL_SAVE_NAME = 'cat_or_dog.pyt'
MODEL_SAVE_PATH = os.path.join('./model_states', MODEL_SAVE_NAME)

"""
NOTE: data for this example has been downloaded from Kaggle
    $> kaggle datasets download -d tongpython/cat-and-dog -p ./data
    $> unzip ./data/cat-and-dog.zip
"""
TRAIN_IMAGES_PATH = os.path.join('./data', 'cats-vs-dogs', 'training_set')
TEST_IMAGES_PATH = os.path.join('./data', 'cats-vs-dogs', 'test_set')


def show_sample(images_path=TRAIN_IMAGES_PATH, num_count=10, num_rows=2):
    """ shows a random sample of num_count/2 cat & num_count/2 dog images """
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
            ax[row, col].axis('off')
            index = row * num_cols + col
            if index < num_count:
                image = Image.open(all_images[index])
                ax[row, col].imshow(image)
    plt.suptitle(f"Random sample of {num_count} images")
    plt.show()

def display_sample(images, labels, predictions=None, fig_size=None, num_cols=10, plot_title=None):
    num_images = len(images)
    num_rows = num_images // num_cols
    num_rows = num_rows if (num_images % num_cols == 0) else num_rows + 1

    label_lookup = {
        0: 'Cat',
        1: 'Dog'
    }

    figsize = (22, 22) if fig_size is None else fig_size

    with sns.axes_style("whitegrid"):
        ctx = sns.plotting_context()
        ctx['axes.labelsize'] = 10.0
        ctx['axes.titlesize'] = 12.0
        ctx['font.size'] = 12.0
        sns.set(context='notebook', font_scale=0.65)
        sns.set_style({"font.sans-serif": ["SF Pro Display", "Verdana", "Arial", "Calibri", "DejaVu Sans"]})

        # plt.figure(figsize=figsize)
        fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize,
                               gridspec_kw={"wspace": 0.02, "hspace": 0.25}, squeeze=True)
        fig.tight_layout()
        for row in range(num_rows):
            for col in range(num_cols):
                ax[row, col].axis('off')
                index = row * num_cols + col
                if index < num_images:
                    image = images[index]
                    image = image.transpose((1,2,0))
                    ax[row, col].imshow(image)
                    if predictions is not None:
                        title = label_lookup[labels[index]] if predictions[index] == labels[index] \
                            else label_lookup[predictions[index]] + '/' + label_lookup[labels[index]]
                    else:
                        title = label_lookup[labels[index]]
                    ax[row, col].set_title(title)
        if plot_title is not None:
            plt.suptitle(plot_title)

    plt.show()
    plt.close()

xforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.RandomAffine(degrees=(-30, 30), translate=(0.25, 0.35), scale=(0.5, 1.5), shear=0.30),
        transforms.RandomHorizontalFlip(0.30),
        transforms.RandomVerticalFlip(0.30),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ]),
    'val_or_test': transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
}


class CatOrDogDataset(Dataset):
    def __init__(self, base_path, transforms=None):
        self.cats = glob.glob(base_path + "/cats/*.jpg")
        self.dogs = glob.glob(base_path + "/dogs/*.jpg")
        self.fpaths = self.cats + self.dogs
        # shuffle paths
        random.shuffle(self.fpaths)
        # dog == 1, cat == 0
        self.labels = [int(fpath.split(os.path.sep)[-1].startswith('dog')) for fpath in self.fpaths]
        # print(self.labels[:10], flush=True)
        self.transforms = transforms
        assert (self.transforms is not None)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.fpaths[idx])
        if self.transforms is not None:
            image = self.transforms(image)
        label = torch.tensor(self.labels[idx])
        # print(f"Index {idx} - image.shape: {image.shape} - label.shape: {label.shape}", flush=True)
        return image, label


def get_data(test_perc=0.2):
    assert (xforms['val_or_test'] is not None)
    train_dataset = CatOrDogDataset(TRAIN_IMAGES_PATH, transforms=xforms['train'])
    test_dataset = CatOrDogDataset(TEST_IMAGES_PATH, transforms=xforms['val_or_test'])
    num_images = len(test_dataset)
    num_test_images = int(test_perc * num_images)
    num_val_images = num_images - num_test_images
    print(f"get_data(): {len(test_dataset)} test images split to {num_val_images} eval & {num_test_images} test images")
    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [num_val_images, num_test_images])
    return train_dataset, val_dataset, test_dataset


def build_model(lr=LEARNING_RATE):
    def conv_layer(ino, out, kernel_size, stride=1):
        return nn.Sequential(
            nn.Conv2d(ino, out, kernel_size, stride),
            nn.ReLU(),
            nn.BatchNorm2d(out),
            nn.MaxPool2d(2)
        )

    net = nn.Sequential(
        conv_layer(3, 64, 3),
        conv_layer(64, 512, 3),
        # conv_layer(512, 512, 3),
        # conv_layer(512, 512, 3),
        # conv_layer(512, 512, 3),
        # conv_layer(512, 512, 3),
        nn.Flatten(),
        nn.Linear(512 * 54 * 54, 2),
        # nn.Sigmoid()  # binary classification
    )

    model = pytk.PytkModuleWrapper(net)
    loss_fn = nn.CrossEntropyLoss()  # nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['acc'])
    return model, optimizer

TRAIN_MODEL = True
PREDICT_MODEL = False
DISPLAY_SAMPLE = False

def main():
    train_dataset, val_dataset, test_dataset = get_data()

    if DISPLAY_SAMPLE:
        # show random sample from test_dataset
        print("Displaying random sample of images from test dataset...")
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
        data_iter = iter(testloader)
        images, labels = data_iter.next()  # fetch a batch of 64 random images
        display_sample(images.cpu().numpy(), labels.cpu().numpy(), num_cols=8,
                       plot_title='Sample Images from test dataset')

    if TRAIN_MODEL:
        # build the model
        model, optimizer = build_model()
        model.summary((NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))

        # train the model
        print("Training model...")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        hist = model.fit_dataset(train_dataset, validation_dataset=val_dataset, epochs=NUM_EPOCHS,
                                 batch_size=BATCH_SIZE, lr_scheduler=scheduler, num_workers=5)
        pytk.show_plots(hist, metric='acc')

        # evaluate performance
        print("Evaluating performance...")
        loss, acc = model.evaluate_dataset(train_dataset, batch_size=BATCH_SIZE)
        print(f"Training data   -> loss: {loss:.4f} - acc: {acc:.4f}")
        loss, acc = model.evaluate_dataset(val_dataset, batch_size=BATCH_SIZE)
        print(f"Validation data -> loss: {loss:.4f} - acc: {acc:.4f}")
        loss, acc = model.evaluate_dataset(test_dataset, batch_size=BATCH_SIZE)
        print(f"Test data       -> loss: {loss:.4f} - acc: {acc:.4f}")

        model.save(MODEL_SAVE_PATH)
        del model

    if PREDICT_MODEL:
        # load model from save path
        print(f"Loading model from {MODEL_SAVE_PATH}")
        model, optimizer = build_model()
        model.load(MODEL_SAVE_PATH)
        model.summary((NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))

        # evaluate performance
        print("Evaluating preformance...")
        loss, acc = model.evaluate_dataset(train_dataset, batch_size=BATCH_SIZE)
        print(f"Training data   -> loss: {loss:.4f} - acc: {acc:.4f}")
        loss, acc = model.evaluate_dataset(val_dataset, batch_size=BATCH_SIZE)
        print(f"Validation data -> loss: {loss:.4f} - acc: {acc:.4f}")
        loss, acc = model.evaluate_dataset(test_dataset, batch_size=BATCH_SIZE)
        print(f"Test data       -> loss: {loss:.4f} - acc: {acc:.4f}")

        # run predictions
        print("Running predictions...")
        y_pred, y_true = model.predict_dataset(test_dataset)
        y_pred = np.argmax(y_pred, axis=1)
        print('Sample labels (50): ', y_true[:50])
        print('Sample predictions: ', y_true[:50])
        print('We got %d/%d incorrect!' % ((y_pred != y_true).sum(), len(y_true)))

        # display sample from test dataset
        print('Displaying sample predictions...')
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True)
        data_iter = iter(testloader)
        images, labels = data_iter.next()  # fetch a batch of 64 random images
        preds = np.argmax(model.predict(images), axis=1)
        display_sample(images.cpu().numpy(), labels.cpu().numpy(), predictions=preds,
                       num_cols=8, plot_title='Sample Images & predictions from test dataset')
        del model

if __name__ == "__main__":
    main()
