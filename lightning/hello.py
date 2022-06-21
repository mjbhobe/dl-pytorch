# /usr/bin/env python

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

print(f"Welcome to Pytorch Lightning {pl.__version__}")
print(f"You are using Pytorch version {torch.__version__}")

xor_inputs = [
    Variable(torch.Tensor([0, 0])),
    Variable(torch.Tensor([0, 1])),
    Variable(torch.Tensor([1, 0])),
    Variable(torch.Tensor([1, 1]))
]

xor_targets = [
    Variable(torch.Tensor([0])),
    Variable(torch.Tensor([1])),
    Variable(torch.Tensor([1])),
    Variable(torch.Tensor([0]))
]

xor_data = list(zip(xor_inputs, xor_targets))
train_loader = DataLoader(xor_data, batch_size=1)

# this is the model
class XORModel(pl.LightningModule):
    def __init__(self):
        super(XORModel, self).__init__()
        self.input_layer = nn.Linear(2, 4)
        self.output_layer = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()

    def forward(self, input):
        # print("INPUT:", input.shape)
        x = self.input_layer(input)
        # print("FIRST:", x.shape)
        x = self.sigmoid(x)
        # print("SECOND:", x.shape)
        output = self.output_layer(x)
        # print("THIRD:", output.shape)
        return output

    def configure_optimizers(self):
        params = self.parameters()

        optimizer = optim.Adam(params=params, lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        xor_input, xor_target = batch

        # print("XOR INPUT:", xor_input.shape)
        # print("XOR TARGET:", xor_target.shape)
        outputs = self(xor_input)
        # print("XOR OUTPUT:", outputs.shape)
        loss = self.loss(outputs, xor_target)
        return loss


from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

checkpoint_callback = ModelCheckpoint()
model = XORModel()
trainer = pl.Trainer(max_epochs=100, callbacks=[checkpoint_callback])
trainer.fit(model, train_dataloaders=train_loader)
print(checkpoint_callback.best_model_path)

import pytorch_toolkit as pyt
