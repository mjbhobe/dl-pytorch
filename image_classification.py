#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings('ignore')

import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pytorch imports
import torch

print('Using Pytorch version: ', torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
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
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 28, 28, 1, 10
NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, L2_REG = 25, 128, 1e-3, 0.001
MODEL_SAVE_NAME = 'pytk_image_classification.pyt'
MODEL_SAVE_PATH = os.path.join('./model_states', MODEL_SAVE_NAME)

xforms = {
    'train': transforms.Compose([
        transforms.RandomAffine(degrees=(-30,30), translate=(0.25,0.35), scale=(0.5, 1.5), shear=0.30),
        transforms.RandomHorizontalFlip(0.30),
        transforms.RandomVerticalFlip(0.30),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ]),
    'val_or_test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
}

def get_data(val_split=0.2):
    # download the FashionMNIST dataset
    train_dataset = datasets.FashionMNIST(
            './data', download=True, train=True, transform=xforms['train'])
    test_dataset = datasets.FashionMNIST(
            './data', download=True, train=False, transform=xforms['val_or_test'])

    test_dataset_size = int(val_split * len(test_dataset))
    val_dataset_size = len(test_dataset) - test_dataset_size
    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, 
            [val_dataset_size, test_dataset_size])
    print(f"train_dataset: {len(train_dataset)} - val_dataset: {len(val_dataset)}" + 
          f" - test_dataset: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset

def display_sample(sample_images, sample_labels, grid_shape=(10, 10), 
        fig_size=None, plot_title=None, sample_predictions=None):
    # just in case these are not imported!
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn')

    num_rows, num_cols = grid_shape
    assert sample_images.shape[0] == num_rows * num_cols

    # a dict to help encode/decode the labels
    FASHION_LABELS = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot',
    }

    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=0.98)
        sns.set_style(
            {"font.sans-serif": ["SF UI Text", "Calibri", "Arial", "DejaVu Sans", "sans"]})

        f, ax = plt.subplots(num_rows, num_cols, figsize=(14, 10) if fig_size is None else fig_size,
                             gridspec_kw={"wspace": 0.05, "hspace": 0.35}, squeeze=True)  # 0.03, 0.25
        # fig = ax[0].get_figure()
        f.tight_layout()
        f.subplots_adjust(top=0.90)  # 0.93

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")
                # de-normalize image
                sample_images[image_index] = \
                    (sample_images[image_index] * 0.5) / 0.5

                # show selected image
                ax[r, c].imshow(sample_images[image_index].squeeze(),
                                cmap="Greys", interpolation='nearest')

                if sample_predictions is None:
                    # show the text label as image title
                    title = ax[r, c].set_title(
                        f"{FASHION_LABELS[sample_labels[image_index]]}")
                else:
                    pred_matches_actual = (
                        sample_labels[image_index] == sample_predictions[image_index])
                    # show prediction from model as image title
                    title = '%s' % FASHION_LABELS[sample_predictions[image_index]]
                    if pred_matches_actual:
                        # if matches, title color is green
                        title_color = 'g'
                    else:
                        # else title color is red
                        # title = '%s/%s' % (FASHION_LABELS[sample_labels[image_index]],
                        #                    FASHION_LABELS[sample_predictions[image_index]])
                        title_color = 'r'

                    # but show the prediction in the title
                    title = ax[r, c].set_title(title)
                    # if prediction is incorrect title color is red, else green
                    plt.setp(title, color=title_color)

        if plot_title is not None:
            plt.suptitle(plot_title)
        plt.show()
        plt.close()


def build_model(lr=LEARNING_RATE):
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3200, 256),
        nn.ReLU(),
        nn.Linear(256, NUM_CLASSES)
    )
    model = pytk.PytkModuleWrapper(net)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['acc'])
    return model, optimizer

TRAIN_MODEL = False
PREDICT_MODEL = True
DISPLAY_SAMPLE = False


def main():
    # download data
    train_dataset, val_dataset, test_dataset = get_data()

    if DISPLAY_SAMPLE:
        # display sample from train dataset (augmented images)
        print('Displaying sample from train dataset...')
        trainloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=64, shuffle=True)
        data_iter = iter(trainloader)
        images, labels = data_iter.next()  # fetch first batch of 64 images & labels
        display_sample(images.cpu().numpy(), labels.cpu().numpy(),
                       grid_shape=(8, 8), plot_title='Sample Training Images')

    if TRAIN_MODEL:
        # build the model 
        model, optimizer = build_model()
        model.summary((NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)) 

        # train the model
        print("Training model...")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        hist = model.fit_dataset(train_dataset, validation_dataset=val_dataset,
                epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr_scheduler=scheduler, num_workers=5)
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
        display_sample(images.cpu().numpy(), labels.cpu().numpy(), sample_predictions=preds,
                       fig_size=(15, 15), grid_shape=(16, 16), plot_title='Sample Predictions')
        del model

if __name__ == "__main__":
    main()
