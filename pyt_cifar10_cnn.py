# -*- coding: utf-8 -*-
"""
pyt_cifar10_cnn.py: Multiclass classification of the CIFAR-10 dataset using Pytorch Convolutional
Neural Network (CNN)

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings
warnings.filterwarnings('ignore')

import sys, os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use('seaborn')
sns.set_style('darkgrid')
sns.set_context('notebook',font_scale=1.10)

# Pytorch imports
import torch
gpu_available = torch.cuda.is_available()
print('Using Pytorch version %s. GPU %s available' % (torch.__version__, "IS" if gpu_available else "IS **NOT**"))
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim
from torchsummary import summary
# My helper functions for training/evaluating etc.
import pyt_helper_funcs as pyt

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
IMGNET_MEAN, IMGNET_STD = 0.5, 0.3  # approximate values

def load_data():
    """
    Load the data using datasets API. 
    We apply some random transforms to training datset as we load the data
    We also split the test_dataset into cross-val/test datasets using 80:20 ration
    """
    xforms = {
        'train' : [
            # add transforms here - scaling, shearing, flipping etc.
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),   
            transforms.ToTensor(),
            transforms.Normalize((IMGNET_MEAN, IMGNET_MEAN, IMGNET_MEAN),
                                 (IMGNET_STD, IMGNET_STD, IMGNET_STD))
        ],
        'test' : [
            transforms.ToTensor(),
            transforms.Normalize((IMGNET_MEAN, IMGNET_MEAN, IMGNET_MEAN),
                                 (IMGNET_STD, IMGNET_STD, IMGNET_STD))
        ],
    }

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                     transform=transforms.Compose(xforms['train']))
    print("No of training records: %d" % len(train_dataset))

    test_dataset = datasets.CIFAR10('./data', download=True, train=False,
                                    transform=transforms.Compose(xforms['test']))
    print("No of test records: %d" % len(test_dataset))

    # lets split the test dataset into val_dataset & test_dataset -> 8000:2000 records
    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [8000, 2000])
    print("No of cross-val records: %d" % len(val_dataset))
    print("No of test records: %d" % len(test_dataset))

    return train_dataset, val_dataset, test_dataset

def display_sample(sample_images, sample_labels, grid_shape=(8, 8),
                   plot_title=None, sample_predictions=None):

    # just in case these are not imported!
    import matplotlib.pyplot as plt
    import seaborn as sns

    # all must be Numpy arrays - use sample_images.cpu().numpy() to convert tensor to Numpy array
    assert ((sample_images is not None) and (isinstance(sample_images, np.ndarray)))
    assert ((sample_labels is not None) and (isinstance(sample_labels, np.ndarray)))
    if sample_predictions is not None:
        assert isinstance(sample_predictions, np.ndarray)

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

    num_rows, num_cols = grid_shape
    assert sample_images.shape[0] >= num_rows * num_cols

    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=0.95)
        sns.set_style({"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})

        f, ax = plt.subplots(num_rows, num_cols, figsize=(14, 11),
            gridspec_kw={"wspace": 0.02, "hspace": 0.35}, squeeze=True)  
        f.tight_layout()
        f.subplots_adjust(top=0.93)

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")

                # show selected image
                sample_image = sample_images[image_index]
                sample_image = sample_image.transpose((1,2,0))
                sample_image = sample_image * IMGNET_STD + IMGNET_MEAN  # since we applied this normalization

                ax[r, c].imshow(sample_image, cmap="Greys")

                if sample_predictions is None:
                    # but show the prediction in the title
                    ax[r, c].set_title("%s" % CIFAR10_LABELS[sample_labels[image_index]])
                else:
                    pred_matches_actual = (
                        sample_labels[image_index] == sample_predictions[image_index])
                    if pred_matches_actual:
                        # show title from prediction or actual in green font
                        title = '%s' % CIFAR10_LABELS[sample_predictions[image_index]]
                        title_color = 'g'
                    else:
                        # show title as actual/prediction in red font
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

# some hyper-parameters
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 32, 32, 3, 10
NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, L2_REG = 100, 64, 0.01, 0.01
MODEL_SAVE_DIR = os.path.join('.', 'model_states')
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'pyt_cifar10_cnn')

# define our network - we are using nn.Sequential
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def build_model(l1, l2, l3):
    model = nn.Sequential(
        pyt.Conv2d(3, l1, 5, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        #nn.Dropout(0.05),

        pyt.Conv2d(l1, l2, 5, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        #nn.Dropout(0.10),

        pyt.Conv2d(l2, l3, 5, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.20),

        Flatten(),

        nn.Linear(2*2*l3, 512),
        nn.Dropout(0.20),        

        nn.Linear(512, 10)
    )
    return model

DO_TRAINING = True
DO_PREDICTION = True
SHOW_SAMPLE = False

MODEL_SAVE_NAME = 'pyt_mnist_dnn'

def main():
    print('Loading datasets...')
    train_dataset, val_dataset, test_dataset = load_data()

    if SHOW_SAMPLE:
        # display sample from test dataset
        testloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        data_iter = iter(testloader)
        images, labels = data_iter.next()  # fetch first batch of 64 images & labels
        display_sample(images.cpu().numpy(), labels.cpu().numpy(), grid_shape=(8, 8),
                       plot_title='Sample Images')

    if DO_TRAINING:
        print('Building model...')
        # build model
        model = pyt.PytModuleWrapper(build_model(16,32,64))
        # define the loss function & optimizer that model should
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(params=model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=L2_REG)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=NUM_EPOCHS//5, gamma=0.1)
        model.compile(loss=loss_fn, optimizer=optimizer, metrics=['acc'])
        # display Keras like summary
        print(model.summary((NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

        # train model
        print('Training model...')
        early_stop = pyt.EarlyStopping(monitor='val_loss', patience=10, verbose=False, save_best_weights=True)
        hist = model.fit_dataset(train_dataset, validation_dataset=val_dataset, lr_scheduler=scheduler,
                                 epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, num_workers=0, early_stopping=early_stop)
        pyt.show_plots(hist)

        # evaluate model performance on train/eval & test datasets
        print('Evaluating model performance...')
        loss, acc = model.evaluate_dataset(train_dataset)
        print('  Training dataset  -> loss: %.4f - acc: %.4f' % (loss, acc))
        loss, acc = model.evaluate_dataset(val_dataset)
        print('  Cross-val dataset -> loss: %.4f - acc: %.4f' % (loss, acc))
        oss, acc = model.evaluate_dataset(test_dataset)
        print('  Test dataset      -> loss: %.4f - acc: %.4f' % (loss, acc))

        # save model state
        model.save(MODEL_SAVE_NAME)
        del model

    if DO_PREDICTION:
        print('Running predictions...')
        # load model state from .pt file
        # model = pyt.load_model(MODEL_SAVE_NAME)
        model = pyt.PytModuleWrapper(pyt.load_model(MODEL_SAVE_NAME))
        # needed for PytModuleWrapper
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(params=model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=NUM_EPOCHS//5, gamma=0.1)        
        model.compile(loss=loss_fn, optimizer=optimizer, metrics=['acc'])

        print('Evaluating model performance...')
        loss, acc = model.evaluate_dataset(train_dataset)
        print('  Training dataset  -> loss: %.4f - acc: %.4f' % (loss, acc))
        loss, acc = model.evaluate_dataset(val_dataset)
        print('  Cross-val dataset -> loss: %.4f - acc: %.4f' % (loss, acc))
        oss, acc = model.evaluate_dataset(test_dataset)
        print('  Test dataset      -> loss: %.4f - acc: %.4f' % (loss, acc))

        # _, all_preds, all_labels = pyt.predict_dataset(model, test_dataset)
        y_pred, y_true = model.predict_dataset(test_dataset)
        y_pred = np.argmax(y_pred, axis=1)
        print('Sample labels (50): ', y_true[:50])
        print('Sample predictions: ', y_true[:50])
        print('We got %d/%d incorrect!' % ((y_pred != y_true).sum(), len(y_true)))

        # display sample from test dataset
        print('Displaying sample predictions...')
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
        data_iter = iter(loader)
        images, labels = data_iter.next()  # fetch a batch of 64 random images
        #_, preds = pyt.predict(model, images)
        preds = np.argmax(model.predict(images), axis=1)
        display_sample(images.cpu().numpy(), labels.cpu().numpy(), sample_predictions=preds,
                       grid_shape=(8, 8), plot_title='Sample Predictions')


if __name__ == "__main__":
    main()

# ----------------------------------------------
# Results: 300 epochs, 32 batch-size, 0.01 LR
#   Training  -> acc 78%
#   Cross-val -> acc 78.33%
#   Testing   -> acc 79.30%
# ----------------------------------------------