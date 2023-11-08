import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
import pyt_flash as flash

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128 * 5 * 5, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 10),
        )
        # self.conv1 = nn.Conv2d(3, 64, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(64, 128, 5)
        # self.fc1 = nn.Linear(128 * 5 * 5, 256)
        # self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 128 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.net(x)
        return x


# Define data transformations
xforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

NUM_EPOCHS, BATCH_SIZE, LR, NUM_CLASSES, NUM_WORKERS = 10, 128, 0.001, 10, 4

# Load the CIFAR-10 dataset
trainset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=xforms,
)
print(f"trainset -> {len(trainset)} records")

testset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=xforms,
)
print(f"testset -> {len(testset)} records")

# trainset, valset = torch.utils.data.random_split(trainset, [40_000, 10_000])
# print(
#     f"After split -> trainset: {len(trainset)} - valset: {len(valset)} - test: {len(testset)} records"
# )

val_size = int(0.8 * len(testset))
test_size = len(testset) - val_size
valset, testset = torch.utils.data.random_split(testset, [val_size, test_size])
# testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
print(
    f"After split -> trainset: {len(trainset)} - valset: {len(valset)} - test: {len(testset)} records"
)


# Initialize the model, loss function, and optimizer
MODEL_SAVE_PATH = pathlib.Path(__file__).parent / "cifar9_model.pth"
accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES)

DO_TRAIN, DO_EVAL, DO_PRED = True, True, True

if __name__ == "__main__":
    if DO_TRAIN:
        model = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        hist = flash.train_model(
            model,
            criterion,
            optimizer,
            trainset,
            valset=valset,
            metric=accuracy,
            device=DEVICE,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

        # flash.show_metrics_plot(hist)
        flash.save_model(model, MODEL_SAVE_PATH)
        del model

    if DO_EVAL:
        model = Net()
        model = flash.load_model(model, MODEL_SAVE_PATH)
        train_acc = flash.evaluate_model(
            model, trainset, metric=accuracy, device=DEVICE
        )
        val_acc = flash.evaluate_model(model, valset, metric=accuracy, device=DEVICE)
        test_acc = flash.evaluate_model(model, testset, metric=accuracy, device=DEVICE)
        print(
            f"Evaluation -> Train: {train_acc:.3f} - Cross-val: {val_acc:.3f} - Test: {test_acc:.3f}"
        )
        del model

    if DO_PRED:
        model = Net()
        model = flash.load_model(model, MODEL_SAVE_PATH)
        labels, preds = flash.predict_model(model, testset, device=DEVICE)
        print(f"Labels(20): {labels[:20]}")
        print(f"Preds (20): {preds[:20]}")
        correct_count = (preds == labels).sum()
        print(f"We got {correct_count} of {len(testset)} correct!")
