# -*- coding: utf-8 -*-
"""
base_model.py - base model, derived from pl.LightningModule that includes all boilerplate
    code I mostly use.

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings

warnings.filterwarnings("ignore")

# Pytorch imports
import pytorch_lightning as pl


class BaseLightningModule(pl.LightningModule):
    def __init__(self):
        super(BaseLightningModule, self).__init__()
        """ you should overload this method in derived class & implement specifics"""

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        """you should overload this function in derived class & implement specifics"""
        raise RuntimeError(
            f"FATAL ERROR: BaseLightningModule.configure_optimizers(...) function has been called."
            f"Did you forget to overload configure_optimizers() in your derived class?"
        )

    def process_batch(self, batch, batch_idx, dataset_name):
        """you should overload this function in derived class & implement specifics"""
        raise RuntimeError(
            f"FATAL ERROR: BaseLightningModule.process_batch(...) function has been called."
            f"Did you forget to overload process_batch() in your derived class?"
        )

    def training_step(self, batch, batch_idx):
        """training step"""
        metrics = self.process_batch(batch, batch_idx, "train")
        return metrics[0]

    def validation_step(self, batch, batch_idx):
        """cross-validation step"""
        metrics = self.process_batch(batch, batch_idx, "val")
        return metrics[0]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """run predictions on a batch"""
        return self.forward(batch)
