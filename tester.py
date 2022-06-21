"""
pyt_50startups_regression.py: testing MetricsHistory class from pytorch_toolkit 

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings
warnings.filterwarnings('ignore')

import random
import sys
import os
import pytorch_toolkit as pytk


def main():
    mh = pytk.MetricsHistory(['acc', 'f1_score'], False)
    print(f"Metrics History: history: {mh.history} - metrics list: {mh.metrics_list}")

    mh = pytk.MetricsHistory(['acc', 'f1_score'], True)
    print(f"Metrics History: history: {mh.history} - metrics list: {mh.metrics_list}")


if __name__ == '__main__':
    main()
