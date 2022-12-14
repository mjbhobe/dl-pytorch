""" quickstart.py - copy of the Pytorch quick start Fashion MNIST classification example"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchmetrics
import torch_directml

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# Get cpu or gpu device for training.
DEVICE = "cuda" if torch.cuda.is_available() else (
    torch_directml.device(torch_directml.default_device()) if torch_directml.is_available() else "cpu")
print(f"Using {DEVICE} device")

# Define model


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def build_model():
    model = NeuralNetwork()
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer


def accuracy(logits, labels):
    acc = torchmetrics.functional.accuracy(logits, labels)
    return acc


def train(dataloader, model, device, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model = model.to(device)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        acc = accuracy(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, acc, current = loss.item(), acc.item(), batch * len(X)
            print(f"loss: {loss:>7f} - acc: {acc:>7f}  [{current:>5d}/{size:>5d}]")

    model = model.to("cpu")


def test(dataloader, model, device, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model = model.to(device)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    model = model.to("cpu")


# train model
EPOCHS, BATCH_SIZE = 5, 64
DEVICE = torch_directml.device(torch_directml.default_device())
print(f"Using {DEVICE} device")

model, loss_fn, optimizer = build_model()

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, DEVICE, loss_fn, optimizer)
    test(test_dataloader, model, DEVICE, loss_fn)
del model

DEVICE = "cpu"
print(f"Using {DEVICE} device")

model, loss_fn, optimizer = build_model()

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, DEVICE, loss_fn, optimizer)
    test(test_dataloader, model, DEVICE, loss_fn)
del model
