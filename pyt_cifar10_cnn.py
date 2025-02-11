# -*- coding: utf-8 -*-
"""
pyt_cifar10_dnn.py: multiclass classification of the CIFAR10 image library.

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings

warnings.filterwarnings('ignore')

import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use('seaborn')
sns.set(style='darkgrid', context='notebook', font_scale=1.20)

# Pytorch imports
import torch

print('Using Pytorch version: ', torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim
# My helper functions for training/evaluating etc.
import pytorch_toolkit as pytk

# to ensure that you get consistent results across runs & machines
# @see: https://discuss.pytorch.org/t/reproducibility-over-different-machines/63047
seed = 123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


def load_data():
    """
    load the data using datasets API. We also split the test_dataset into 
    cross-val/test datasets using 80:20 ration
    """
    from torch.utils.data.sampler import SubsetRandomSampler

    means, stds = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    train_xforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])
    val_test_xforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_xforms)

    print("No of training records: %d" % len(train_dataset))

    test_dataset = datasets.CIFAR10(
        './data', train=False, download=True, transform=val_test_xforms)
    print("No of test records: %d" % len(test_dataset))

    # lets split the test dataset into val_dataset & test_dataset -> 8000:2000 records
    val_size = int(0.8 * len(test_dataset))
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(
        test_dataset, [val_size, test_size])
    print("No of cross-val records: %d" % len(val_dataset))
    print("No of test records: %d" % len(test_dataset))

    return train_dataset, val_dataset, test_dataset


def display_sample(sample_images, sample_labels, grid_shape=(10, 10), plot_title=None, sample_predictions=None):
    # just in case these are not imported!
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn')

    num_rows, num_cols = grid_shape
    assert sample_images.shape[0] == num_rows * num_cols

    # a dict to help encode/decode the labels
    CIFAR10_LABELS = {
        0: 'Plane',
        1: 'Auto',
        2: 'Bird',
        3: 'Cat',
        4: 'Deer',
        5: 'Dog',
        6: 'Frog',
        7: 'Horse',
        8: 'Ship',
        9: 'Truck',
    }

    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=0.98)
        sns.set_style(
            {"font.sans-serif": ["SF UI Text", "Calibri", "Arial", "DejaVu Sans", "sans"]})

        f, ax = plt.subplots(num_rows, num_cols, figsize=(8, 8),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.35}, squeeze=True)  # 0.03, 0.25
        # fig = ax[0].get_figure()
        f.tight_layout()
        f.subplots_adjust(top=0.90)  # 0.93
        means, stds = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")
                # de-normalize image
                sample_image = sample_images[image_index].transpose((1, 2, 0))
                sample_image = (sample_image * stds) + means

                # show selected image
                ax[r, c].imshow(sample_image.squeeze(),
                                cmap="Greys", interpolation='nearest')

                if sample_predictions is None:
                    # show the text label as image title
                    title = ax[r, c].set_title(
                        f"{CIFAR10_LABELS[sample_labels[image_index]]}")
                else:
                    pred_matches_actual = (
                        sample_labels[image_index] == sample_predictions[image_index])
                    # show prediction from model as image title

                    if pred_matches_actual:
                        title = '%s' % CIFAR10_LABELS[sample_predictions[image_index]]
                        # if matches, title color is green
                        title_color = 'g'
                    else:
                        # else title color is red
                        title = '%s/%s' % (CIFAR10_LABELS[sample_labels[image_index]],
                                           CIFAR10_LABELS[sample_predictions[image_index]])
                        title_color = 'r'

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


# if you prefer to use Convolutional Neural Network, use the following model definition
class Cifar10ConvNet(pytk.PytkModule):
    def __init__(self):
        super(Cifar10ConvNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Dropout(0.30),

            nn.Linear(4 * 4 * 128, 512),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(512, NUM_CLASSES)
        )
        # self.conv1 = pytk.Conv2d(3, 32, 3, padding=1)
        # self.conv2 = pytk.Conv2d(32, 64, 3, padding=1)
        # self.conv3 = pytk.Conv2d(64, 128, 3, padding=1)
        # self.fc1 = pytk.Linear(4 * 4 * 128, 512)
        # self.out = pytk.Linear(512, NUM_CLASSES)

    def forward(self, x):
        # x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # #x = F.dropout(x, p=0.20, training=self.training)
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # #x = F.dropout(x, p=0.10, training=self.training)
        # x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        # # flatten input (for DNN)
        # x = pytk.Flatten(x)
        # x = F.dropout(x, p=0.30, training=self.training)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.30, training=self.training)
        # # NOTE: nn.CrossEntropyLoss() includes a logsoftmax call, which applies a softmax
        # # function to outputs. So, don't apply one yourself!
        # # x = F.softmax(self.out(x), dim=1)  # -- don't do this!
        # x = self.out(x)
        # return x
        return self.net(x)


def build_model():
    model = Cifar10ConvNet()
    # define the loss function & optimizer that model should
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(),
                           lr=LEARNING_RATE, weight_decay=L2_REG)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['acc'])
    return model


DO_TRAINING = False
DO_EVALUATION = False
DO_PREDICTION = True
SHOW_SAMPLE = True

MODEL_SAVE_NAME = 'pyt_cifar10_cnn.pyt'
MODEL_SAVE_PATH = os.path.join('.', 'model_states', MODEL_SAVE_NAME)
NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, L2_REG = 25, 64, 0.001, 0.0005


def main():
    print('Loading datasets...')
    train_dataset, val_dataset, test_dataset = load_data()

    if SHOW_SAMPLE:
        # display sample from test dataset
        print('Displaying sample from train dataset...')
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
        data_iter = iter(testloader)
        images, labels = next(data_iter)  # fetch first batch of 64 images & labels
        display_sample(images.cpu().numpy(), labels.cpu().numpy(),
                       grid_shape=(8, 8), plot_title='Sample Images')
        sys.exit(-1)

    if DO_TRAINING:
        model = build_model()
        # display Keras like summary
        model.summary((NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))

        # train model
        hist = model.fit_dataset(train_dataset, validation_dataset=val_dataset,
                                 epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        pytk.show_plots(hist, metric='acc', plot_title='Training metrics')

        # save model state
        model.save(MODEL_SAVE_PATH)
        del model

    if DO_EVALUATION:
        # load model state from .pt file
        model = build_model()
        model.load(MODEL_SAVE_PATH)

        # evaluate model performance on train/eval & test datasets
        print('Evaluating model performance...')
        loss, acc = model.evaluate_dataset(train_dataset)
        print('  Training dataset  -> loss: %.4f - acc: %.4f' % (loss, acc))
        loss, acc = model.evaluate_dataset(val_dataset)
        print('  Cross-val dataset -> loss: %.4f - acc: %.4f' % (loss, acc))
        loss, acc = model.evaluate_dataset(test_dataset)
        print('  Test dataset      -> loss: %.4f - acc: %.4f' % (loss, acc))

        # save model state
        model.save(MODEL_SAVE_PATH)
        del model

    if DO_PREDICTION:
        # load model state from .pt file
        model = build_model()
        model.load(MODEL_SAVE_PATH)
        # model = pytk.load_model(MODEL_SAVE_PATH)
        model.summary((NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))

        # evaluate model performance on train/eval & test datasets
        print('Evaluating model performance...')
        loss, acc = model.evaluate_dataset(train_dataset)
        print('  Training dataset  -> loss: %.4f - acc: %.4f' % (loss, acc))
        loss, acc = model.evaluate_dataset(val_dataset)
        print('  Cross-val dataset -> loss: %.4f - acc: %.4f' % (loss, acc))
        loss, acc = model.evaluate_dataset(test_dataset)
        print('  Test dataset      -> loss: %.4f - acc: %.4f' % (loss, acc))

        y_pred, y_true = model.predict_dataset(test_dataset)
        y_pred = np.argmax(y_pred, axis=1)
        print('Sample labels (50): ', y_true[:50])
        print('Sample predictions: ', y_true[:50])
        print('We got %d/%d incorrect!' %
              ((y_pred != y_true).sum(), len(y_true)))

        # display sample from test dataset
        print('Displaying sample predictions...')
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=True)
        data_iter = iter(test_loader)
        images, labels = data_iter.next()  # fetch a batch of 64 random images
        preds = np.argmax(model.predict(images), axis=1)
        display_sample(images.cpu().numpy(), labels.cpu().numpy(), sample_predictions=preds,
                       grid_shape=(8, 8), plot_title='Sample Predictions')


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# Results:
#   CNN with epochs=25, batch-size=32, LR=0.001
#       Training  -> acc: 80.78%
#       Cross-val -> acc: 80.21%
#       Testing   -> acc: 80.62%
# Model is slightly over-fitting, overall performance could be much better!
# -------------------------------------------------------------------
