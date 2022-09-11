#!/usr/bin/env python
import warnings

warnings.filterwarnings('ignore')

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use('seaborn')
sns.set(style='darkgrid', context='notebook', font_scale=1.20)

# Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch import optim
# My helper functions for training/evaluating etc.
import pytorch_toolkit as pytk

SEED = pytk.seed_all()
print(f"Using seed {SEED}")
print(f"Using Pytorch {torch.__version__}. GPU {'available :)!' if torch.cuda.is_available() else 'not available :('}")

# set up data dirs
THIS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(THIS_DIR, "data", "kaggle", "flowers")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABEL_DATA_FILE = os.path.join(DATA_DIR, "cat_to_name.json")
MODEL_SAVE_DIR = os.path.join(THIS_DIR, "model_states")
# for a modest machine, reduce image sizes to 64 x 64
# IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 224, 224, 3, 102
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 64, 64, 3, 102


data_transforms = {
    'training': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMAGE_WIDTH),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])]),

    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_WIDTH),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])]),

    'testing': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_WIDTH),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
}

# datasets for the folders holding images of flowers
image_datasets = {
    'training': datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['training']),
    'testing': datasets.ImageFolder(TEST_DIR, transform=data_transforms['testing']),
    'validation': datasets.ImageFolder(VALID_DIR, transform=data_transforms['validation'])
}


def display_sample(sample_images, sample_labels, grid_shape=(10, 10), plot_title=None,
                   sample_predictions=None):
    # just in case these are not imported!
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn')

    num_rows, num_cols = grid_shape
    assert sample_images.shape[0] == num_rows * num_cols
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])

    cat2name = None
    with open(LABEL_DATA_FILE, "r") as f:
        cat2name = json.load(f)

    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=0.98)
        sns.set_style(
            {"font.sans-serif": ["SF UI Text", "Calibri", "Arial", "DejaVu Sans", "sans"]})

        f, ax = plt.subplots(num_rows, num_cols, figsize=(14, 10),
                             gridspec_kw={"wspace": 0.35, "hspace": 0.35}, squeeze=True)  # 0.03, 0.25
        f.tight_layout()
        f.subplots_adjust(top=0.90)  # 0.93

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")
                sample_image = sample_images[image_index].clip(0.0, 1.0)
                # transpose from Pytorch (c, w, h) to (w, h, c) format
                sample_image = sample_image.transpose((1, 2, 0))
                # de-normalize image
                sample_image = (sample_image * stds) + means

                # show selected image
                ax[r, c].imshow(sample_image, interpolation='nearest')

                sample_prediction = None
                try:
                    sample_prediction = cat2name[str(sample_labels[image_index])]
                except:
                    sample_prediction = "Unknown"

                if sample_predictions is None:
                    # show the text label as image title
                    title = ax[r, c].set_title(f"{sample_prediction}")
                else:
                    pred_matches_actual = (
                        sample_labels[image_index] == sample_predictions[image_index])
                    # show prediction from model as image title
                    title = '%s' % sample_prediction
                    if pred_matches_actual:
                        # if matches, title color is green
                        title_color = 'g'
                    else:
                        # else title color is red
                        title_color = 'r'

                    # but show the prediction in the title
                    title = ax[r, c].set_title(title)
                    # if prediction is incorrect title color is red, else green
                    plt.setp(title, color=title_color)

        if plot_title is not None:
            plt.suptitle(plot_title)
        plt.show()
        plt.close()


test_dataloader = torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64, shuffle=True)
data_iter = iter(test_dataloader)
sample_images, sample_labels = data_iter.next()
print(f"sample_images.shape {sample_images.shape} - sample_labels.shape {sample_labels.shape}")
display_sample(sample_images.cpu().numpy(), sample_labels.cpu().numpy(),
               grid_shape=(8, 8), plot_title="Sample Data from Test Dataset")

# ---------------- BUILD & TRAIN MODEL -------------------------------------------------------------
NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, L2_REG = 5, 32, 0.001, 0.001
MODEL_SAVE_NAME = 'flowers_vgg19.pyt'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME)
print(f"Model state will be saved to {MODEL_SAVE_PATH}")


def build_model(lr=LEARNING_RATE):

    from torchvision import models

    vgg19 = models.vgg19(pretrained=True)
    print(vgg19)
    for param in vgg19.features.parameters():
        param.require_grad = False  # freeze weight

    # our own classifier - only this part will be re-trained
    # our custom classifier
    classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, NUM_CLASSES),
    )
    # classifier = nn.Sequential(
    #     nn.Flatten(),
    #     # NOTE: classifier.Conv2d layer #28 has 512 filters
    #     nn.Linear(512 * 7 * 7, 1024),
    #     nn.ReLU(),
    #     nn.Dropout(0.20),
    #     nn.Linear(1024, 512),
    #     nn.ReLU(),
    #     nn.Dropout(0.20),
    #     nn.Linear(512, NUM_CLASSES)
    # )
    vgg19.classifier = classifier

    model = pytk.PytkModuleWrapper(vgg19)
    loss_fn = nn.CrossEntropyLoss()  # nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=L2_REG)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['acc'])
    return model, optimizer


try:
    del model
except NameError:
    pass

model, optimizer = build_model()
print(model.summary((NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))
# sys.exit(-1)

# train the model
from torch.optim.lr_scheduler import LambdaLR  # ExponentialLR, StepLR, CyclicLR
import math
INITIAL_LR, DROP_RATE, EPOCHS_DROP = 0.5, 0.8, 10.0
def lambda1(epoch): return INITIAL_LR * math.pow(DROP_RATE, math.floor(epoch / EPOCHS_DROP))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
hist = model.fit_dataset(image_datasets['training'], validation_dataset=image_datasets['validation'],
                         epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr_scheduler=scheduler, num_workers=5)
pytk.show_plots(hist, metric='acc', plot_title='Pretrained VGG19 Model Performance')

# evaluate performance
print('Evaluating model performance...')
loss, acc = model.evaluate_dataset(image_datasets['training'])
print(f'  Training dataset -> loss: {loss:.4f} - acc: {acc:.4f}')
loss, acc = model.evaluate_dataset(image_datasets['validation'])
print(f'  Cross-val dataset -> loss: {loss:.4f} - acc: {acc:.4f}')
loss, acc = model.evaluate_dataset(image_datasets['testing'])
print(f'  Test dataset      -> loss: {loss:.4f} - acc: {acc:.4f}')

model.save(MODEL_SAVE_PATH)
del model
