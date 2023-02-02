# -*- coding: utf-8 -*-
""" layers.py - layers with initializations for torch models """
import warnings

warnings.filterwarnings('ignore')

import sys

if sys.version_info < (2,):
    raise Exception(
        "torch_training_toolkit does not support Python 1. Please use a Python 3+ interpreter!"
    )

# Pytorch imports
import torch
import torch.nn as nn


# print('Using Pytorch version: ', torch.__version__)


# ----------------------------------------------------------------------------------
# convenience functions to create layers with weights & biases initialized
# ----------------------------------------------------------------------------------
def Linear(in_features, out_features, bias = True, device = None, dtype = None):
    """
        (convenience function)
        creates a nn.Linear layer, with weights initiated using xavier_uniform initializer
        and bias, if set, initialized using zeros initializer as is the default in Keras.
        @params:
        - in_nodes: # of nodes from pervious layer
        - out_nodes: # of nodes in this layer
        @returns:
        - an instance of nn.Linear class
    """
    layer = nn.Linear(in_features, out_features, bias, device, dtype)
    # @see: https://msdn.microsoft.com/en-us/magazine/mt833293.aspx for example
    torch.nn.init.xavier_uniform_(layer.weight)
    if bias:
        torch.nn.init.zeros_(layer.bias)
    return layer


def Conv2d(
    in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1,
    dilation = 1, groups = 1, bias = True, padding_mode = 'zeros',
    device = None, dtype = None
):
    """
        (convenience function)
        Creates a nn.Conv2d layer, with weights initiated using xavier_uniform initializer
        and bias, if set, initialized using zeros initializer. This is similar to Keras.
        @params:
            - same as nn.Conv2d params
        @returns:
            - instance of nn.Conv2d layer
    """
    layer = nn.Conv2d(
        in_channels, out_channels, kernel_size = kernel_size,
        stride = stride, padding = padding, dilation = dilation,
        groups = groups, bias = bias, padding_mode = padding_mode,
        device = device, dtype = dtype
    )
    # @see: https://msdn.microsoft.com/en-us/magazine/mt833293.aspx for example
    torch.nn.init.xavier_uniform_(layer.weight)
    if bias:
        torch.nn.init.zeros_(layer.bias)
    return layer
