# -*- coding: utf-8 -*-
"""
pyt_fashion_mnist_dnn.py: Multiclass classification of Zolando's Fashion MNIST dataset.
Part-3 of the tutorial for torch training toolkit, where we add L1 regularization &
a learning scheduler during training.

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

warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
np.set_printoptions(precision = 4, linewidth = 1024, suppress = True)
plt.style.use('seaborn')
sns.set(style = 'darkgrid', context = 'notebook', font_scale = 1.20)

# Pytorch imports
import torch

print('Using Pytorch version: ', torch.__version__)
import torch.nn as nn
import torchmetrics
import torchsummary
from torchvision import datasets, transforms
# import the Pytorch training toolkit (t3)
import torch_training_toolkit as t3

# to ensure that you get consistent results across runs & machines
seed = 123
t3.seed_all(seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    """
    load the data using datasets API. We also split the test_dataset into 
    cross-val/test datasets using 80:20 ration
    """
    mean, std = 0.5, 0.5
    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )

    train_dataset = datasets.FashionMNIST(
        root = './data', train = True, download = True, transform = transformations
    )

    print("No of training records: %d" % len(train_dataset))

    test_dataset = datasets.FashionMNIST(
        './data', train = False, download = True, transform = transformations
    )
    print("No of test records: %d" % len(test_dataset))

    # lets split the test dataset into val_dataset & test_dataset -> 8000:2000 records
    # val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [8000, 2000])
    val_dataset, test_dataset = t3.split_dataset(test_dataset, split_perc = 0.2)
    print("No of cross-val records: %d" % len(val_dataset))
    print("No of test records: %d" % len(test_dataset))

    return train_dataset, val_dataset, test_dataset


def display_sample(
    sample_images, sample_labels, grid_shape = (10, 10), plot_title = None,
    sample_predictions = None
):
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
        sns.set_context("notebook", font_scale = 0.98)
        sns.set_style(
            {"font.sans-serif": ["SF UI Text", "Calibri", "Arial", "DejaVu Sans", "sans"]}
        )

        f, ax = plt.subplots(
            num_rows, num_cols, figsize = (14, 10),
            gridspec_kw = {"wspace": 0.05, "hspace": 0.35}, squeeze = True
        )  # 0.03, 0.25
        # fig = ax[0].get_figure()
        f.tight_layout()
        f.subplots_adjust(top = 0.90)  # 0.93

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")
                # de-normalize image
                sample_images[image_index] = \
                    (sample_images[image_index] * 0.5) / 0.5

                # show selected image
                ax[r, c].imshow(
                    sample_images[image_index].squeeze(),
                    cmap = "Greys", interpolation = 'nearest'
                )

                if sample_predictions is None:
                    # show the text label as image title
                    title = ax[r, c].set_title(
                        f"{FASHION_LABELS[sample_labels[image_index]]}"
                    )
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
                    plt.setp(title, color = title_color)

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
            t3.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)


# if you prefer to use Convolutional Neural Network, use the following model definition
class FMNISTConvNet(nn.Module):
    def __init__(self):
        super(FMNISTConvNet, self).__init__()
        self.net = nn.Sequential(
            t3.Conv2d(NUM_CHANNELS, 128, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            t3.Conv2d(128, 64, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Flatten(),

            t3.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            t3.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)


DO_TRAINING = True
DO_PREDICTION = True
SHOW_SAMPLE = False
USE_CNN = False  # if False, will use an ANN

MODEL_SAVE_NAME = 'pyt_mnist_cnn.pyt' if USE_CNN else 'pyt_mnist_dnn.pyt'
MODEL_SAVE_PATH = os.path.join('..', 'model_states', MODEL_SAVE_NAME)
NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, L1_REG, L2_REG = 25, 64, 0.001, 5e-4, 2e-4


def main():
    print('Loading datasets...')
    train_dataset, val_dataset, test_dataset = load_data()

    # declare loss functions
    loss_fn = nn.CrossEntropyLoss()
    # metrics map - metrics to track during training
    metrics_map = {
        # accuracy
        "acc": torchmetrics.classification.MulticlassAccuracy(num_classes = NUM_CLASSES),
        # overall f1-score
        # "f1": torchmetrics.classification.MulticlassF1Score(num_classes = NUM_CLASSES)
    }
    # define the trainer
    trainer = t3.Trainer(
        loss_fn = loss_fn, device = DEVICE,
        epochs = NUM_EPOCHS, batch_size = BATCH_SIZE,
        metrics_map = metrics_map
    )

    if SHOW_SAMPLE:
        # display sample from test dataset
        print('Displaying sample from train dataset...')
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = True)
        data_iter = iter(testloader)
        images, labels = next(data_iter)  # fetch first batch of 64 images & labels
        display_sample(
            images.cpu().numpy(), labels.cpu().numpy(),
            grid_shape = (8, 8), plot_title = 'Sample Images'
        )

    if DO_TRAINING:
        print(f'Using {"CNN" if USE_CNN else "ANN"} model...')
        model = FMNISTConvNet() if USE_CNN else FMNISTNet()
        model = model.to(DEVICE)
        # display Keras like summary
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))
        # optimizer is required only during training!
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = L2_REG)
        # add L1 regularization & a learning scheduler
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size = NUM_EPOCHS // 5, gamma = 0.1)
        # add an EarlyStopping
        early_stopping = t3.EarlyStopping(
            model, metrics_map, using_val_dataset = True,
            monitor = "val_loss", patience = 5,
        )
        hist = trainer.fit(
            model, optimizer, train_dataset,
            validation_dataset = val_dataset,
            lr_scheduler = scheduler,
            early_stopping = early_stopping
        )
        # display the tracked metrics
        hist.plot_metrics("Model Performance")

        # evaluate model performance on train/eval & test datasets
        print('Evaluating model performance...')
        metrics = trainer.evaluate(model, train_dataset)
        print(
            f"  Training dataset  -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}"
        )
        metrics = trainer.evaluate(model, val_dataset)
        print(
            f"  Cross-val dataset -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}"
        )
        metrics = trainer.evaluate(model, test_dataset)
        print(
            f"  Testing dataset   -> loss: {metrics['loss']:.4f} - acc: {metrics['acc']:.4f}"
        )
        # save model state
        t3.save_model(model, MODEL_SAVE_PATH)
        del model

    if DO_PREDICTION:
        # load model state from .pt file
        model = FMNISTConvNet() if USE_CNN else FMNISTNet()
        model = t3.load_model(model, MODEL_SAVE_PATH)
        print(torchsummary.summary(model, (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

        y_pred, y_true = trainer.predict(model, test_dataset)
        y_pred = np.argmax(y_pred, axis = 1)
        print('Sample labels (50): ', y_true[:50])
        print('Sample predictions: ', y_true[:50])
        print(
            'We got %d/%d incorrect!' %
            ((y_pred != y_true).sum(), len(y_true))
        )

        # display sample from test dataset
        print('Displaying sample predictions...')
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = True)
        data_iter = iter(testloader)
        images, labels = next(data_iter)
        preds = np.argmax(trainer.predict(model, images.cpu().numpy()), axis = 1)
        display_sample(
            images.cpu().numpy(), labels.cpu().numpy(), sample_predictions = preds,
            grid_shape = (8, 8), plot_title = 'Sample Predictions'
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