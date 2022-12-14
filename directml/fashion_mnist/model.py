""" model.py - the Pytorch model """
import warnings

warnings.filterwarnings('ignore')

import sys

if sys.version_info < (3,):
    raise Exception("pytorch_toolkit does not support Python 2. Please use a Python 3+ interpreter!")

# torch imports
import torch
import torch.nn as nn


def ConvLayer(inp, out, ks=3, s=1, p=1):
    net = nn.Sequential(
        nn.Conv2d(inp, out, kernel_size=ks, stride=s, padding=p),
        nn.BatchNorm2d(out),
        nn.ReLU(),
        nn.Conv2d(out, out, kernel_size=ks, stride=s, padding=p),
        nn.BatchNorm2d(out),
        nn.ReLU()
    )
    return net


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.convNet = nn.Sequential(
            ConvLayer(1, 64, 3),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.20),

            ConvLayer(64, 128, 3),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.20),

            ConvLayer(128, 256, 3),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.20),

            nn.Flatten(),

            nn.Linear(3 * 3 * 256, 1024),
            nn.ReLU(),
            # nn.Dropout(p=0.20),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.20),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x = self.flatten(x)
        x = self.convNet(x)
        return x


class SimpleNet(nn.Module):
    def __init__(self, image_width, image_height, num_channels, num_classes):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(image_width * image_height * num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.net(x)
        return x


def build_model_simple(image_width, image_height, num_channels, num_classes, 
        learning_rate, l2_reg=None):
    model = SimpleNet(image_width, image_height, num_channels, num_classes)
    # model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    return model, loss_fn, optimizer


def build_model(num_classes, learning_rate, l2_reg=None):
    model = ConvNet(num_classes)
    # model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    return model, loss_fn, optimizer
