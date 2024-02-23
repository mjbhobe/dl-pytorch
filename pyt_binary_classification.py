"""
pyt_binary_classification.py: binary classification of 2D data

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use("seaborn-v0_8")
sns.set(style="whitegrid", font_scale=1.1, palette="muted")

# Pytorch imports
import torch

print("Using Pytorch version: ", torch.__version__)
import torch.nn as nn
import torchmetrics
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC

print(f"Using torchmetrics: {torchmetrics.__version__}")

# My helper functions for training/evaluating etc.
import torch_training_toolkit as t3

SEED = t3.seed_all()

NUM_EPOCHS = 25
BATCH_SIZE = 1024
LR = 0.01

DATA_FILE = os.path.join(".", "csv_files", "weatherAUS.csv")
print(f"Data file: {DATA_FILE}")
MODEL_SAVE_PATH = os.path.join(".", "model_states", "weather_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# load data, select fields & apply scaling
# ---------------------------------------------------------------------------
def get_data(
    test_split=0.20, shuffle_it=True, balance=False, sampling_strategy=0.85, debug=False
):
    from imblearn.over_sampling import SMOTE

    df = pd.read_csv(DATA_FILE)

    if shuffle_it:
        df = shuffle(df)

    cols = ["Rainfall", "Humidity3pm", "Pressure9am", "RainToday", "RainTomorrow"]
    df = df[cols]

    # convert categorical cols - RainToday & RainTomorrow to numeric
    df["RainToday"].replace({"No": 0, "Yes": 1}, inplace=True)
    df["RainTomorrow"].replace({"No": 0, "Yes": 1}, inplace=True)

    # drop all rows where any cols == Null
    df = df.dropna(how="any")

    # display plot of target
    sns.countplot(data=df, x=df.RainTomorrow)
    plt.title("RainTomorrow: existing counts")
    plt.show()

    X = df.drop(["RainTomorrow"], axis=1).values
    y = df["RainTomorrow"].values
    if debug:
        print(
            f"{'Before balancing ' if balance else ''} X.shape = {X.shape}, "
            f"y.shape = {y.shape}, y-count = {np.bincount(y)}"
        )

    if balance:
        ros = SMOTE(sampling_strategy=sampling_strategy, random_state=SEED)
        X, y = ros.fit_resample(X, y)
        if debug:
            print(
                f"Resampled -> X.shape = {X.shape}, y.shape = {y.shape}, "
                f"y-count = {np.bincount(y)}"
            )

    # display plot of target
    df2 = pd.DataFrame(
        X, columns=["Rainfall", "Humidity3pm", "Pressure9am", "RainToday"]
    )
    df2["RainTomorrow"] = y
    sns.countplot(data=df2, x=df2.RainTomorrow)
    plt.title("RainTomorrow: after re-balancing")
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=SEED
    )
    if debug:
        print(
            f"Split data -> X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}, "
            f"X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}"
        )

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    # y_train = np.expand_dims(y_train, axis=1)
    # y_test = np.expand_dims(y_test, axis=1)

    # NOTE: BCELoss() expects labels to be floats - why???
    # y_train = y_train.astype(np.float32)
    # y_test = y_test.astype(np.float32)

    y_train = y_train[:, np.newaxis]
    y_test = y_test[:, np.newaxis]

    return (X_train, y_train), (X_test, y_test)


class WeatherDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_file_path,
        shuffle_it=True,
        balance=True,
        sampling_strategy=0.85,
        seed=SEED,
    ):
        assert os.path.exists(
            data_file_path
        ), f"FATAL: {data_file_path} - file does not exist!"
        df = pd.read_csv(data_file_path)
        if shuffle_it:
            df = shuffle(df)
        cols = ["Rainfall", "Humidity3pm", "Pressure9am", "RainToday", "RainTomorrow"]
        df = df[cols]
        # convert categorical cols - RainToday & RainTomorrow to numeric
        df["RainToday"].replace({"No": 0, "Yes": 1}, inplace=True)
        df["RainTomorrow"].replace({"No": 0, "Yes": 1}, inplace=True)
        # drop all rows where any cols == Null
        df = df.dropna(how="any")

        # assign X & y
        self.X = df.drop(["RainTomorrow"], axis=1).values
        self.y = df["RainTomorrow"].values
        if balance:
            ros = SMOTE(sampling_strategy=sampling_strategy, random_state=seed)
            self.X, self.y = ros.fit_resample(self.X, self.y)
        ss = StandardScaler()
        self.X = ss.fit_transform(self.X)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx, :]
        label = self.y[idx, :]
        return features, label


class Net(nn.Module):
    def __init__(self, num_features):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            t3.Linear(num_features, 16),
            nn.ReLU(),
            # t3.Linear(32, 16),
            # nn.ReLU(),
            t3.Linear(16, 8),
            nn.ReLU(),
            t3.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# DO_TRAINING = True
# DO_EVAL = True
# DO_PREDICTION = True

import pickle


def main():
    parser = t3.TrainingArgsParser()
    args = parser.parse_args()

    # # load & preprocess data
    # (X_train, y_train), (X_test, y_test) = get_data(balance = True, sampling_strategy = 0.90,
    #                                                 debug = True)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    dataset = WeatherDataset(DATA_FILE)
    print(f"Loaded {len(dataset)} records", flush=True)
    # set aside 10% as test dataset
    train_dataset, test_dataset = t3.split_dataset(dataset, split_perc=0.1)
    print(
        f"train_dataset: {len(train_dataset)} recs, test_dataset: {len(test_dataset)} recs"
    )
    metrics_map = {
        "acc": BinaryAccuracy(),
        "f1": BinaryF1Score(),
        "roc_auc": BinaryAUROC(thresholds=None),
    }

    loss_fn = nn.BCELoss()
    trainer = t3.Trainer(
        loss_fn=loss_fn,
        device=DEVICE,
        metrics_map=metrics_map,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    if args.train:
        # build model
        model = Net(4)
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        hist = trainer.fit(model, optimizer, train_dataset, validation_split=0.2)
        hist_pkl = os.path.join(
            os.path.dirname(__file__), "model_states", "history2.pkl"
        )
        with open(hist_pkl, "wb") as f:
            pickle.dump(hist, f)
        # sys.exit(-1)
        hist.plot_metrics(title="Model Performance", fig_size=(16, 8))
        t3.save_model(model, MODEL_SAVE_PATH)
        del model

    if args.eval:
        model = Net(4)
        model = t3.load_model(model, MODEL_SAVE_PATH)
        print(model)
        # evaluate performance
        print("Evaluating performance...")
        print("Training dataset")
        # evaluate training dataset (just re-confirming similar results as during training)
        metrics = trainer.evaluate(model, train_dataset)
        print(f"Training metrics: {metrics}")
        print("Testing dataset")
        # evaluate test dataset (for a good model, these should not be much different from training)
        metrics = trainer.evaluate(model, test_dataset)
        print(f"Testing metrics: {metrics}")
        del model

    if args.pred:
        model = Net(4)
        model = t3.load_model(model, MODEL_SAVE_PATH)
        print(model)

        preds, actuals = trainer.predict(model, test_dataset)
        preds = np.round(preds).ravel()
        actuals = actuals.ravel()
        incorrect_counts = (preds != actuals).sum()
        print(f"We got {incorrect_counts} of {len(actuals)} predictions wrong!")
        print(classification_report(actuals, preds))
        t3.plot_confusion_matrix(
            confusion_matrix(actuals, preds),
            class_names=["No Rain", "Rain"],
            title="Rain prediction for tomorrow",
        )
        del model


if __name__ == "__main__":
    main()

# Results:
#   Training (1000 epochs)
#       - loss: 0.377  acc: 84.0%
#   Training (1000 epochs)
#       - loss: 0.377 acc:  84.1%
#   Conclusion: No overfitting, but accuracy is low. Possibly due to very imbalanced data
#
#   Training (1000 epochs) with re-sampling
#       - loss: 0.377  acc: 84.0%
#   Training (1000 epochs)
#       - loss: 0.377 acc:  84.1%
#   Conclusion: No overfitting, but accuracy is low. Possibly due to very imbalanced data
