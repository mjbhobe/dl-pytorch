# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 18:06:07 2022

@author: manis
"""

import warnings
warnings.filterwarnings('ignore')

import sys, os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from PIL import Image, ImageDraw
%matplotlib inline

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use('seaborn')
sns.set_style('darkgrid')
sns.set_context('notebook',font_scale=1.10)

# Pytorch imports
import torch
gpu_available = torch.cuda.is_available()
print('Using Pytorch version %s. GPU %s available' % (torch.__version__, "IS" if gpu_available else "IS **NOT**"))
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

MODEL_SAVE_DIR = os.path.join('.','model_states')
DATA_FOLDER = os.path.join('C:\\Dev\\Code\\git-projects\\dl-pytorch','data','Kaggle','histo')
print(DATA_FOLDER)

DATA_FILE_NAME = 'train_labels.csv'
DATA_FILE_PATH = os.path.join(DATA_FOLDER, DATA_FILE_NAME)
print(f"DATA_FILE_PATH: {DATA_FILE_PATH}")
assert os.path.exists(DATA_FILE_PATH), f"{DATA_FILE_PATH} - path does not exist!"

labels = pd.read_csv(DATA_FILE_PATH)
labels.head()

labels['label'].value_counts().plot(kind='bar')

malignant_df = labels.loc[labels['label'] == 1]
malignant_image_names = malignant_df['id'].values

nrows, ncols = 5, 5
rand_ids = np.random.randint(0, len(malignant_image_names), nrows * ncols)
random_image_ids = malignant_image_names[rand_ids]

old_figsize = plt.rcParams['figure.figsize']

plt.rcParams['figure.figsize'] = (15.0, 15.0)
plt.subplots_adjust(wspace=0, hspace=0)

# let's display 25 randomly selected images in a 5x5 grid
for i, image_id in enumerate(random_image_ids):
    image_path = os.path.join(DATA_FOLDER, 'train', image_id + '.tif')
    # load the image & draw
    img = Image.open(image_path)
    # draw a 32*32 rectangle
    draw = ImageDraw.Draw(img)
    draw.rectangle(((32, 32), (64, 64)),outline="green")
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(np.array(img))
    plt.axis('off')

plt.rcParams['figure.figsize'] = old_figsize

labels = labels.sample(frac=1).reset_index(drop=True)
labels.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(labels['id'], labels['label'], test_size=0.30, random_state=seed)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.20, random_state=seed)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

y_train, y_val, y_test = y_train.values, y_val.values, y_test.values

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
        Open a PIL image, resize to (IMAGE_HEIGHT, IMAGE_WIDTH), apply transforms (if any) &
        convert to Numpy array and return array and label at index
        """
        image_path = self.image_paths[index]
        assert(os.path.exists(image_path)), f'Invalid path - {image_path} does not exist!'
        img = Image.open(image_path)
        if self.transforms is not None:
            img = self.transforms(img)
        img = np.array(img).astype(np.float32)
        img /= 255.0
        label = int(self.labels[index])
        return (img, label)


# we are scaling all images to same size + converting them to tensors & normalizing data
xforms = {
    'train': transforms.Compose([
        transforms.RandomAffine(0, shear=0.2),         # random shear 0.2
#         transforms.randomaffine(0, scale=(0.8, 1.2)),  # random zoom 0.2
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]),
    'eval': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
    ]),
}


X_train_image_paths = [os.path.join(DATA_FOLDER, 'train', image_id + '.tif') \
                       for image_id in X_train.values]
X_val_image_paths = [os.path.join(DATA_FOLDER, 'train', image_id + '.tif') \
                     for image_id in X_val.values]
X_test_image_paths = [os.path.join(DATA_FOLDER, 'train', image_id + '.tif') \
                      for image_id in X_test.values]

def display_sample(sample_images, sample_labels, sample_predictions=None, grid_shape=(8,8),
                   plot_title=None, fig_size=None):
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
        1 : 'Malignant',
        0 : 'Benign'
    }

    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=1.1)
        sns.set_style({"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})

        f, ax = plt.subplots(num_rows, num_cols, figsize=((20, 20) if fig_size is None else fig_size),
            gridspec_kw={"wspace": 0.02, "hspace": 0.25}, squeeze=True)
        #fig = ax[0].get_figure()
        f.tight_layout()
        f.subplots_adjust(top=0.95)

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")

                # show selected image
                sample_image = sample_images[image_index]
                # got image as (NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
                sample_image = sample_image.transpose((1,2,0))
                #sample_image = sample_image * 0.5 + 0.5  # since we applied this normalization
                sample_image *= 255.0

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
                    plt.setp(title, color=title_color)
        # set plot title, if one specified
        if plot_title is not None:
            f.suptitle(plot_title)

        plt.show()
        plt.close()


train_dataset = HistoDataset(X_train_image_paths, y_train, xforms['train'])
eval_dataset = HistoDataset(X_val_image_paths, y_val, xforms['eval'])
test_dataset = HistoDataset(X_test_image_paths, y_test, xforms['test'])
print(len(train_dataset), len(X_train_image_paths), len(eval_dataset), len(X_val_image_paths), \
      len(test_dataset), len(X_test_image_paths))

IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_CLASSES = 96, 96, 3, 2
NUM_EPOCHS, BATCH_SIZE, LR_RATE, L2_REG = 25, 64, 0.001, 0.0005

cnn_model = nn.Sequential(
    pytk.Conv2d(NUM_CHANNELS, 8, 3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(8),
    nn.MaxPool2d(kernel_size=2, stride=2),

    pytk.Conv2d(8, 16, 3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(16),
    nn.MaxPool2d(kernel_size=2, stride=2),

    pytk.Conv2d(16, 32, 3, padding=0),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.MaxPool2d(kernel_size=2, stride=2),

    pytk.Conv2d(32, 64, 3, padding=0),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Flatten(),

    nn.Linear(64*4*4, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.25),

    nn.Linear(512, NUM_CLASSES)
)

model = pytk.PytkModuleWrapper(cnn_model)
print(model.summary((NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)))

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(params=model.parameters(), lr=LR_RATE, momentum=0.8, nesterov=False, weight_decay=L2_REG)
optimizer = optim.Adam(params=model.parameters(), lr=LR_RATE, weight_decay=L2_REG)
model.compile(loss=criterion, optimizer=optimizer, metrics=['acc'])


hist = model.fit_dataset(train_dataset, validation_dataset=eval_dataset, epochs=2,
                         batch_size=BATCH_SIZE, verbose=1)
