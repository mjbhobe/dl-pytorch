# /usr/bin/env python

"""
mnist.py - MNIST digits classification with Pytorch Lightning

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
sns.set(style='darkgrid', context='notebook', font_scale=1.10)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim
from torch.utils.data import DataLoader
import torchmetrics as tm

import pytorch_toolkit as pytk
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

print(f"Welcome to Pytorch Lightning {pl.__version__}")
print(f"You are using Pytorch version {torch.__version__}")

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
    transformations = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root='../data', train=True, download=True,
                                   transform=transformations)

    print("No of training records: %d" % len(train_dataset))

    test_dataset = datasets.MNIST('../data', train=False, download=True,
                                  transform=transformations)
    print("No of test records: %d" % len(test_dataset))

    # lets split the test dataset into val_dataset & test_dataset -> 8000:2000 records
    val_dataset, test_dataset = \
        torch.utils.data.random_split(test_dataset, [8000, 2000])
    print("No of cross-val records: %d" % len(val_dataset))
    print("No of test records: %d" % len(test_dataset))

    return train_dataset, val_dataset, test_dataset


def display_sample(sample_images, sample_labels, grid_shape=(10, 10), plot_title=None,
                   sample_predictions=None):
    # just in case these are not imported!
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn')

    num_rows, num_cols = grid_shape
    assert sample_images.shape[0] == num_rows * num_cols

    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=0.90)
        sns.set_style(
            {"font.sans-serif": ["SF Pro Display", "SF UI Text", "Verdana", "Arial", "DejaVu Sans", "sans"]})

        f, ax = plt.subplots(num_rows, num_cols, figsize=(14, 10),
                             gridspec_kw={"wspace": 0.02, "hspace": 0.25}, squeeze=True)
        # fig = ax[0].get_figure()
        f.tight_layout()
        f.subplots_adjust(top=0.90)

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")
                # show selected image
                ax[r, c].imshow(sample_images[image_index].numpy().squeeze(), cmap="Greys")

                if sample_predictions is None:
                    # but show the prediction in the title
                    title = ax[r, c].set_title("No: %d" % sample_labels[image_index])
                else:
                    pred_matches_actual = (sample_labels[image_index] == sample_predictions[image_index])
                    if pred_matches_actual:
                        # show title from prediction or actual in green font
                        title = '%s' % sample_predictions[image_index]
                        title_color = 'g'
                    else:
                        # show title as actual/prediction in red font
                        title = '%s/%s' % ([sample_labels[image_index]],
                                           [sample_predictions[image_index]])
                        title_color = 'r'

                    # but show the prediction in the title
                    title = ax[r, c].set_title(title)
                    # if prediction is incorrect title color is red, else green
                    plt.setp(title, color=title_color)

        if plot_title is not None:
            plt.suptitle(plot_title)
        plt.show()
        plt.close()


# our model
class MNISTModel(pl.LightningModule):
    def __init__(self, lr):
        super(MNISTModel, self).__init__()
        self.convNet = nn.Sequential(
            pytk.Conv2d(1, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.20),

            pytk.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.10),

            nn.Flatten(),

            pytk.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Dropout(p=0.20),

            pytk.Linear(512, NUM_CLASSES)
        )
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = tm.Accuracy()
        self.val_acc = tm.Accuracy()

        self.train_batch_losses = []
        self.val_batch_losses = []
        self.train_batch_accs = []
        self.val_batch_accs = []

        self.history = {
            "loss": [],
            "acc": [],
            "val_loss": [],
            "val_acc": []
        }
        self.log_file = open(os.path.join(os.getcwd(), 'mnist_log.txt'), 'w')

    def forward(self, input):
        return self.convNet(input)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.convNet.parameters(), self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        step_loss = self.loss_fn(outputs, targets)
        step_acc = self.train_acc(outputs, targets)
        self.train_batch_losses.append(step_loss.item())
        self.train_batch_accs.append(step_acc.item())
        self.log('train_loss', step_loss, prog_bar=True)
        self.log('train_acc', step_acc, prog_bar=True)
        print(f"training_step -> batch_idx: {batch_idx} - loss: {step_loss:.3f} - acc: {step_acc:.3f}", file=self.log_file)
        return {"loss": step_loss, "acc": step_acc}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        step_loss = self.loss_fn(outputs, targets)
        step_acc = self.val_acc(outputs, targets)
        self.val_batch_losses.append(step_loss.item())
        self.val_batch_accs.append(step_acc.item())
        self.log('val_loss', step_loss, prog_bar=True)
        self.log('val_acc', step_acc, prog_bar=True)
        print(f"validation_step -> batch_idx: {batch_idx} - loss: {step_loss:.3f} - acc: {step_acc:.3f}", file=self.log_file)
        return {"val_loss": step_loss, "val_acc": step_acc}

    def training_epoch_end(self, outputs) -> None:
        self.train_acc.reset()
        train_epoch_loss = np.array(self.train_batch_losses).mean()
        train_epoch_acc = np.array(self.train_batch_accs).mean()
        self.history["loss"].append(train_epoch_loss)
        self.history["acc"].append(train_epoch_acc)
        self.train_batch_losses.clear()
        self.train_batch_accs.clear()
        print(f"training_epoch_end -> loss: {train_epoch_loss:.3f} - acc: {train_epoch_acc:.3f}", file=self.log_file)

    def validation_epoch_end(self, outputs) -> None:
        self.val_acc.reset()
        val_epoch_loss = np.array(self.val_batch_losses).mean()
        val_epoch_acc = np.array(self.val_batch_accs).mean()
        self.history["val_loss"].append(val_epoch_loss)
        self.history["val_acc"].append(val_epoch_acc)
        self.val_batch_losses.clear()
        self.val_batch_accs.clear()
        print(f"validation_epoch_end -> loss: {val_epoch_loss:.3f} - acc: {val_epoch_acc:.3f}", file=self.log_file)


# some globals
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 28, 28, 1, 10
EPOCHS, BATCH_SIZE, LR = 5, 64, 0.001

# load the datasets
train_dataset, val_dataset, test_dataset = load_data()


# print('Displaying sample from train dataset...')
# trainloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
# data_iter = iter(trainloader)
# images, labels = data_iter.next()  # fetch first batch of 64 images & labels
# display_sample(images, labels, grid_shape=(8, 8), plot_title='Sample Images')

train_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

model = MNISTModel(LR)
trainer = pl.Trainer(max_epochs=EPOCHS)
trainer.fit(model, train_data_loader, val_data_loader)
#pytk.show_plots(model.history, metric='acc', plot_title="Model performance")

print("Epoch metrics:")
print(f"  Train losses: {model.epoch_metrics['train_loss']}")
print(f"  Train accs  : {model.epoch_metrics['train_acc']}")
print(f"  Val losses  : {model.epoch_metrics['val_loss']}")
print(f"  Val accs    : {model.epoch_metrics['val_acc']}")
