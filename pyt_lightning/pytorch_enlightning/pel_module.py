# -*- coding: utf-8 -*-
"""
base_module.py - base model, derived from pl.LightningModule that includes all boilerplate
    code I mostly use.

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings

warnings.filterwarnings("ignore")

# Pytorch imports
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl


class EnLitModule(pl.LightningModule):
    """base class that helps further reduce the code you write in your
    Pythorch Lightning module class derived from LightningModule.
    Derive your module from this class and just override 2 methods
        - def __init__(...) - the constructor, of course. Here you should
            define the network (and call it self.net -- this is a requirement!)
        - def configure_optimizers(self)  -- to define optimizers
        - def process_batch(self, batch, batch_idx, dataset_name) -- common code
            you would normally write to in the training_step, validation_step
            and testing_step methods of LightningModule class. The LightningFlashModule
            overrides these methods of the LightningModule class and calls this
            method, but passes and extra parameter 'dataset_name' with a value
            'train' or 'val' or 'test' depending on whether it is called from the
            training_step, validation_step or testing_step method. You can use this
            value to "customize" your code accordingly (for example, you may want
            to write same code for train/validation, but different for test).
    """

    def __init__(self):
        super(EnLitModule, self).__init__()
        self.net = None
        """ you should overload this method in derived class & implement specifics"""

    def forward(self, x):
        if self.net is None:
            raise RuntimeError(
                f"FATAL ERROR: LightningFlashModule.forward(...) function has been called."
                f"But self.net parameter is not defined. Did you name your network something other "
                f"than self.net?"
            )
        return self.net(x)

    def configure_optimizers(self):
        """you should overload this function in derived class & implement specifics"""
        raise RuntimeError(
            f"FATAL ERROR: LightningFlashModule.configure_optimizers(...) function has been called."
            f"Did you forget to overload configure_optimizers() in your derived class?"
        )

    def process_batch(self, batch, batch_idx, dataset_name) -> STEP_OUTPUT:
        """you should overload this function in derived class & implement specifics"""
        raise RuntimeError(
            f"FATAL ERROR: LightningFlashModule.process_batch(...) function has been called."
            f"Did you forget to overload process_batch() in your derived class?"
        )

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """training step"""
        metrics = self.process_batch(batch, batch_idx, "train")
        return metrics

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """cross-validation step"""
        metrics = self.process_batch(batch, batch_idx, "val")
        return metrics

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """cross-validation step"""
        metrics = self.process_batch(batch, batch_idx, "test")
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """run predictions on a batch"""
        return self.forward(batch)


if __name__ == "__main__":
    raise RuntimeError(
        "FATAL ERROR: this is a re-useable functions module. Cannot run it independently."
    )
