#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cifar10_cnn.py - Fashion MNIST dataset classification using CNNs """
import os, sys
import pathlib
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchmetrics
import torchsummary

import torch_training_toolkit as t3
from utils import get_data_cifar10 as get_data, display_sample

np.set_printoptions(precision = 6, linewidth = 1024, suppress = True)
plt.style.use('seaborn')
sns.set(style = 'darkgrid', context = 'notebook', font_scale = 1.20)

SEED = t3.seed_all(t3.T3_FAV_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 32, 32, 3, 10
NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE = 25, 64, 0.001

BASE_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model_states", "cifar10_cnn.pt")


class Net(nn.Module):
    """ our base model (using CNNs) """

    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            t3.Conv2d(3, 32, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            t3.Conv2d(32, 64, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            t3.Conv2d(64, 128, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Dropout(0.30),

            t3.Linear(4 * 4 * 128, 512),
            nn.ReLU(),
            nn.Dropout(0.30),

            t3.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(pathlib.Path(__file__).name)
    parser.add_argument(
        "--do_training", action = "store_true",
        help = "Specify this flag to train the model"
    )
    parser.add_argument(
        "--do_predictions", action = "store_true",
        help = "Specify this flag to run predictions"
    )
    parser.add_argument(
        "--show_sample", action = "store_true",
        help = "Specify this flag to show sample from dataset"
    )
    parser.add_argument(
        "--num_epochs", type = int, default = 25,
        help = "Specifies how many epochs to run training for (default=25)"
    )
    parser.add_argument(
        "--batch_size", type = int, default = 32,
        help = "Specifies batch size for training (default=32)"
    )
    parser.add_argument(
        "--lr", type = float, default = 0.01,
        help = "Specifies learning rate to use for optimizer (default=0.001)"
    )
    parser.add_argument(
        "--l1_reg", type = float, required = False,
        help = "Optional param - specified L1 regularization to be applied"
    )
    parser.add_argument(
        "--l2_reg", type = float, required = False,
        help = "Optional param - specified L2 regularization to be applied"
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        # missing command line args
        parser.print_help()
        parser.exit()

    # DO_TRAINING = args.do_training
    # DO_PREDICTIONS = args.do_predictions
    # SHOW_SAMPLE = args.show_sample
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr

    train_dataset, val_dataset, test_dataset, class_names = get_data(DATA_DIR, debug = True)
    if args.show_sample:
        display_sample(
            test_dataset, class_names, title = "Sample Images from CIFAR10 Test dataset"
        )

    loss_fn = nn.CrossEntropyLoss()
    metrics_map = {
        "acc": torchmetrics.classification.MulticlassAccuracy(num_classes = NUM_CLASSES)
    }
    trainer = t3.Trainer(
        loss_fn, device = DEVICE, metrics_map = metrics_map,
        epochs = NUM_EPOCHS, batch_size = BATCH_SIZE
    )

    if args.do_training:
        print("Training model...")
        model = Net()
        model = model.to(DEVICE)
        input_dim = (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
        print(torchsummary.summary(model, input_dim))
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size = 5, gamma = 0.1, verbose = True
        )

        hist = trainer.fit(
            model, optimizer, train_dataset, validation_dataset = val_dataset,
            # lr_scheduler = lr_scheduler,
            l1_reg = args.l1_reg, l2_reg = args.l2_reg
        )
        hist.plot_metrics(title = "Model Performance")

        # evaluate performance
        metrics = trainer.evaluate(model, train_dataset)
        print(f" Training dataset -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}")
        metrics = trainer.evaluate(model, val_dataset)
        print(f" Cross-val dataset -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}")
        metrics = trainer.evaluate(model, test_dataset)
        print(f" Test dataset -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}")

        t3.save_model(model, MODEL_SAVE_PATH)
        del model

    if args.do_predictions:
        print("Running predictions...")
        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"FATAL ERROR: cannot find model state saved to {MODEL_SAVE_PATH}")
            print("Please train the model before running predictions!")
            parser.print_help()
            parser.exit()

        model = Net()
        model = t3.load_model(model, MODEL_SAVE_PATH)
        model = model.to(DEVICE)
        input_dim = (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
        print(torchsummary.summary(model, input_dim))
        y_pred, y_true = trainer.predict(model, test_dataset)
        y_pred = np.argmax(y_pred, axis = 1)
        correct_preds = (y_pred == y_true).sum()
        pred_acc = (correct_preds / len(y_true)) * 100.0
        print(
            f"We got {correct_preds} of {len(y_true)} correct predictions {pred_acc:.2f}% accuracy!"
        )
        rand_indexes = np.random.randint(0, len(y_true), 64)
        print(f"Actuals (50 random samples)    : {y_true[rand_indexes]}")
        print(f"Predictions (50 random samples): {y_pred[rand_indexes]}")
        del model


if __name__ == "__main__":
    main()
