import warnings

warnings.filterwarnings('ignore')

import sys, os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import glob
from PIL import Image, ImageDraw

# tweaks for libraries
np.set_printoptions(precision = 6, linewidth = 1024, suppress = True)
plt.style.use('seaborn')
sns.set_style('darkgrid')
sns.set_context('notebook', font_scale = 1.10)

# Pytorch imports
import torch

gpu_available = torch.cuda.is_available()
# print('Using Pytorch version %s. GPU %s available' % (torch.__version__, "IS" if gpu_available else "IS **NOT**"))
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim
from torchsummary import summary

# import the Pytorch Toolkit here....
import pytorch_toolkit as pytk

# to ensure that you get consistent results across various machines
# @see: https://discuss.pytorch.org/t/reproducibility-over-different-machines/63047
seed = 123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

DATA_FOLDER = os.path.join('.', 'data', 'Kaggle', 'histo')
DATA_FILE_NAME = 'train_labels.csv'
DATA_FILE_PATH = os.path.join(DATA_FOLDER, DATA_FILE_NAME)
IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_CLASSES = 96, 96, 3, 2
NUM_EPOCHS, BATCH_SIZE, LR_RATE, L2_REG = 100, 64, 3e-4, 5e-5
MODEL_SAVE_DIR = './model_states'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "pyt_hosto.pt")


def get_data():
    print(f"Loading data from {DATA_FILE_PATH}")
    df = pd.read_csv(DATA_FILE_PATH)
    # df['label'].value_counts().plot(kind='bar', title="Diagnosis Distribution")
    plt.show()

    # shuffle the dataframe
    df = df.sample(frac = 1).reset_index(drop = True)
    # split into train/cross-val/test datasets
    X_train, X_test, y_train, y_test = train_test_split(df['id'], df['label'], test_size = 0.20, random_state = seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.10, random_state = seed)
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    X_train_image_paths = [os.path.join(DATA_FOLDER, 'train', image_id + '.tif') for image_id in X_train.values]
    X_val_image_paths = [os.path.join(DATA_FOLDER, 'train', image_id + '.tif') for image_id in X_val.values]
    X_test_image_paths = [os.path.join(DATA_FOLDER, 'train', image_id + '.tif') for image_id in X_test.values]

    return (X_train_image_paths, y_train.values), (X_val_image_paths, y_val.values), \
           (X_test_image_paths, y_test.values)


def display_sample(sample_images, sample_labels, sample_predictions = None, grid_shape = (8, 8),
                   plot_title = None, fig_size = None):
    """ display a random selection of images & corresponding labels, optionally with predictions
        The display is laid out in a grid of num_rows x num_col cells
        If sample_predictions are provided, then each cell's title displays the prediction
        (if it matches actual) or actual/prediction if there is a mismatch
    """
    from PIL import Image
    import seaborn as sns

    num_rows, num_cols = grid_shape

    assert len(sample_images) == num_rows * num_cols

    # a dict to help encode/decode the labels
    LABELS = {
        1: 'Malignant',
        0: 'Benign'
    }

    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale = 1.1)
        sns.set_style({"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})

        f, ax = plt.subplots(num_rows, num_cols, figsize = ((20, 20) if fig_size is None else fig_size),
                             gridspec_kw = {"wspace": 0.02, "hspace": 0.25}, squeeze = True)
        # fig = ax[0].get_figure()
        f.tight_layout()
        f.subplots_adjust(top = 0.95)

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")

                # show selected image
                sample_image = sample_images[image_index]
                # got image as (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
                sample_image = sample_image.transpose((1, 2, 0))
                sample_image = sample_image * 0.5 + 0.5  # since we applied this normalization
                # sample_image *= 255.0

                ax[r, c].imshow(sample_image)

                if sample_predictions is None:
                    # show the actual labels in the cell title
                    title = ax[r, c].set_title("%s" % LABELS[sample_labels[image_index]])
                else:
                    # else check if prediction matches actual value
                    true_label = sample_labels[image_index]
                    pred_label = sample_predictions[image_index]
                    prediction_matches_true = (true_label == pred_label)
                    if prediction_matches_true:
                        # if actual == prediction, cell title is prediction shown in green font
                        title = LABELS[true_label]
                        title_color = 'g'
                    else:
                        # if actual != prediction, cell title is actua/prediction in red font
                        title = '%s\n%s' % (LABELS[true_label], LABELS[pred_label])
                        title_color = 'r'
                    # display cell title
                    title = ax[r, c].set_title(title)
                    plt.setp(title, color = title_color)
        # set plot title, if one specified
        if plot_title is not None:
            f.suptitle(plot_title)

        plt.show()
        plt.close()


# define the custom dataset
from torch.utils.data.dataset import Dataset


class HistoDataset(Dataset):
    def __init__(self, image_paths, labels, transforms):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Open a PIL image, resize to (IMAGE_HEIGHT, IMAGE_WIDTH), apply transforms (if any) & convert to Numpy array
        and return array and label at index
        """
        image_path = self.image_paths[index]
        assert (os.path.exists(image_path)), f'Invalid path - {image_path} does not exist!'
        img = Image.open(image_path)
        if self.transforms is not None:
            img = self.transforms(img)
        # img = np.array(img).astype(np.float32)
        # img /= 255.0
        # label = int(self.labels[index])
        label = torch.LongTensor(self.labels[index])
        return (img, label)


# we are scaling all images to same size + converting them to tensors & normalizing data
xforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(96, scale = (0.8, 1.0), ratio = (1.0, 1.0)),
        transforms.ToTensor(),
    ]),
    'eval': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
    ]),
}


def build_model():
    cnn_model = nn.Sequential(
        nn.Conv2d(NUM_CHANNELS, 8, 3, padding = 1),
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.MaxPool2d(kernel_size = 2, stride = 2),

        nn.Conv2d(8, 16, 3, padding = 1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(kernel_size = 2, stride = 2),

        nn.Conv2d(16, 32, 3, padding = 0),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size = 2, stride = 2),

        nn.Conv2d(32, 64, 3, padding = 0),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size = 2, stride = 2),

        nn.Flatten(),
        nn.ReLU(),
        nn.Dropout(0.5),

        # nn.Linear(64*4*4, 1024),
        # nn.ReLU(),
        # nn.Dropout(0.5),

        nn.Linear(64 * 4 * 4, 100),
        nn.ReLU(),
        nn.Dropout(0.25),

        nn.Linear(100, NUM_CLASSES)
    )

    model = pytk.PytkModuleWrapper(cnn_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = model.parameters(), lr = LR_RATE, weight_decay = L2_REG)
    model.compile(loss = criterion, optimizer = optimizer, metrics = ['acc'])
    return model


SHOW_SAMPLE = True


def main():
    (X_train_image_paths, y_train), (X_val_image_paths, y_val), \
    (X_test_image_paths, y_test) = get_data()

    # define our datasets
    train_dataset = HistoDataset(X_train_image_paths, y_train, xforms['train'])
    eval_dataset = HistoDataset(X_val_image_paths, y_val, xforms['eval'])
    test_dataset = HistoDataset(X_test_image_paths, y_test, xforms['test'])
    print(len(train_dataset), len(X_train_image_paths), len(eval_dataset), len(X_val_image_paths), \
          len(test_dataset), len(X_test_image_paths))

    if SHOW_SAMPLE:
        # display a sample of 64 images/labels from the test dataset
        loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = True)
        data_iter = iter(loader)
        sample_images, sample_labels = data_iter.next()  # fetch first batch of 64 images & labels
        print(f'Dataset: image.shape = {sample_images.shape}, labels.shape = {sample_labels.shape}')
        display_sample(sample_images.cpu().numpy(), sample_labels.cpu().numpy(), plot_title = "Sample Test Images")

    # define our model
    # model = pytk.PytkModuleWrapper(cnn_model)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(params=model.parameters(), lr=LR_RATE, weight_decay=L2_REG)
    # model.compile(loss=criterion, optimizer=optimizer, metrics=['acc'])
    model = build_model()
    print(model.summary((NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

    # train the model
    from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=1)
    lr_scheduler = StepLR(optimizer, step_size = 20, gamma = 0.5, verbose = 1)

    hist = model.fit_dataset(train_dataset, validation_dataset = eval_dataset, epochs = NUM_EPOCHS,
                             batch_size = BATCH_SIZE, num_workers = 3, verbose = 1)
    pytk.show_plots(hist, metric = 'acc')

    # evaluate performance
    print('Evaluating model performance after training...')
    loss, acc = model.evaluate_dataset(train_dataset)
    print('  Training data  -> loss: %.3f, acc: %.3f' % (loss, acc))
    loss, acc = model.evaluate_dataset(eval_dataset)
    print('  Cross-val data -> loss: %.3f, acc: %.3f' % (loss, acc))
    loss, acc = model.evaluate_dataset(test_dataset)
    print('  Testing data   -> loss: %.3f, acc: %.3f' % (loss, acc))

    model.save(MODEL_SAVE_PATH)
    del model

    # load model from saved state
    # model = pytk.PytkModuleWrapper(pytk.load_model(MODEL_SAVE_PATH))
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(params=model.parameters(), lr=LR_RATE, weight_decay=L2_REG)
    # model.compile(loss=criterion, optimizer=optimizer, metrics=['acc'])
    model = build_model()
    model.load(MODEL_SAVE_PATH)
    print(model.summary((NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

    # run predictions
    # display sample from test dataset
    print('Running predictions....')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

    actuals, predictions = [], []

    for batch_no, (images, labels) in enumerate(test_loader):
        # images, labels = data_iter.next()  # fetch first batch of 64 images & labels
        preds = model.predict(images)
        actuals.extend(labels.cpu().numpy().ravel())
        predictions.extend(np.argmax(preds, axis = 1).ravel())

    actuals = np.array(actuals)
    predictions = np.array(predictions)

    print('Sample actual values & predictions...')
    print('  - Acutal values: ', actuals[:25])
    print('  - Predictions  : ', predictions[:25])
    correct_preds = (actuals == predictions).sum()
    acc = correct_preds / len(actuals)
    print('  We got %d of %d correct (%.3f accuracy)' % (correct_preds, len(actuals), acc))

    from sklearn.metrics import confusion_matrix, classification_report
    print(classification_report(actuals, predictions))

    # display predictions
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)
    data_iter = iter(test_loader)
    images, labels = data_iter.next()

    preds = model.predict(images)
    preds = np.argmax(preds, axis = 1)

    print(images.shape, labels.shape, preds.shape)
    display_sample(images.cpu().numpy(), labels.cpu().numpy(), sample_predictions = preds,
                   grid_shape = (8, 8), fig_size = (16, 20), plot_title = 'Sample Predictions')

    del model


if __name__ == "__main__":
    main()
