"""
pyt_breast_cancer.py: Binary classification of Wisconsin Breast Cancer dataset

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings
warnings.filterwarnings('ignore')

import os, sys, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use('seaborn')
sns.set_style('darkgrid')
sns.set_context('notebook',font_scale=1.10)

# Pytorch imports
import torch
print('Using Pytorch version: ', torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchsummary import summary
# My helper functions for training/evaluating etc.
import pyt_helper_funcs as pyt

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed);
data_file_path = './data/wisconsin_breast_cancer.csv'

def download_data_file():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'


    df_cols = [
        "id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean",
        "smoothness_mean","compactness_mean","concavity_mean","concave points_mean",
        "symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se",
        "area_se","smoothness_se","compactness_se","concavity_se","concave points_se",
        "symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst",
        "area_worst","smoothness_worst","compactness_worst","concavity_worst",
        "concave points_worst","symmetry_worst","fractal_dimension_worst",
    ]

    print('Downloading data from %s...' % url)
    wis_df = pd.read_csv(url, header=None, names=df_cols, index_col=0)
    wis_df.to_csv(data_file_path)

def load_data(val_split=0.20, test_split=0.10):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if not os.path.exists(data_file_path):
        download_data_file()

    assert os.path.exists(data_file_path), "%s - unable to open file!" % data_file_path

    wis_df = pd.read_csv(data_file_path, index_col=0)

    # diagnosis is the target col - char
    wis_df['diagnosis'] = wis_df['diagnosis'].map({'M': 1, 'B': 0})
    f_names = wis_df.columns[wis_df.columns != 'diagnosis']
    
    X = wis_df.drop(['diagnosis'], axis=1).values
    y = wis_df['diagnosis'].values

    # split into train/test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=val_split, random_state=seed)
    
    # scale data
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    # split into train/eval sets
    X_val, X_test, y_val, y_test = \
        train_test_split(X_test, y_test, test_size=test_split, random_state=seed)
    
    # train_df = pd.DataFrame(X_train, columns=f_names)
    # train_df['diagnosis'] = y_train

    # eval_df = pd.DataFrame(X_val, columns=f_names)
    # eval_df['diagnosis'] = y_val

    # test_df = pd.DataFrame(X_test, columns=f_names)
    # test_df['diagnosis'] = y_test

    # return (pyt.PandasDataset(train_df, target_col_name='diagnosis'),
    #         pyt.PandasDataset(eval_df, target_col_name='diagnosis'),
    #         pyt.PandasDataset(test_df, target_col_name='diagnosis'))
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# our ANN
class WBCNet(pyt.PytModule):
    def __init__(self, inp_size, hidden1, hidden2, num_classes):
        super(WBCNet, self).__init__()
        self.fc1 = nn.Linear(inp_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden2, num_classes)
        self.dropout = nn.Dropout(0.20)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        x = self.fc1(inp)
        x = self.relu1(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        #x = self.dropout(x)
        # NOTE: nn.CrossEntropyLoss() includes a logsoftmax call, which applies a softmax
        # function to outputs. So, don't apply one yourself!
        #x = F.softmax(self.out(x), dim=1)  # -- don't do this!
        x = self.out(x)
        x = self.sigmoid(x)
        return x

DO_TRAINING = True
DO_TESTING = True
DO_PREDICTION = True
MODEL_SAVE_NAME = 'pyt_wbc_ann'

# Hyper-parameters
NUM_FEATURES = 30
NUM_CLASSES = 1
NUM_EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001

def main():

    # train_dataset, eval_dataset, test_dataset = load_data()
    # print('Loaded -> len(train_dataset): %d - len(eval_dataset): %d - len(test_dataset): %d' %
    #             (len(train_dataset), len(eval_dataset), len(test_dataset)))
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
    # NOTE: BCELoss() expects y to be of type float
    y_train, y_val, y_test = y_train.astype(np.float), y_val.astype(np.float), y_test.astype(np.float)

    if DO_TRAINING:
        print('Building model...')
        model = WBCNet(NUM_FEATURES, 30, 30, NUM_CLASSES)
        # define the loss function & optimizer that model should
        loss_fn = nn.BCELoss() #nn.CrossEntropyLoss()
        #optimizer = optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=0.001)
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                nesterov=True, momentum=0.9, dampening=0) #, weight_decay=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        # model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        model.compile(loss=loss_fn, optimizer=optimizer, metrics=['acc','f1'])
        print(model)

        # train model
        print('Training model...')
        #metrics_list = ['acc','f1'] 
        hist = model.fit(X_train, y_train, val_set=(X_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        #hist = model.fit_dataset(train_dataset, val_dataset=eval_dataset, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        # hist = pyt.train_model(model,train_dataset, loss_fn=loss_fn, optimizer=optimizer, #lr_scheduler=lr_scheduler,
        #             #val_dataset=eval_dataset,
        #             epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, metrics=metrics_list) #,'roc_auc'])
        pyt.show_plots(hist)

        # evaluate model performance on train/eval & test datasets
        print('\nEvaluating model performance...')
        loss, acc, f1 = model.evaluate(X_train, y_train)
        #loss, acc, f1 = model.evaluate_dataset(train_dataset)
        #loss, acc, f1 = pyt.evaluate_model(model, train_dataset, loss_fn, metrics=metrics_list)
        print('  Training dataset  -> loss: %.4f - acc: %.4f - f1: %.4f' % (loss, acc, f1))
        loss, acc, f1 = model.evaluate(X_val, y_val)
        #loss, acc, f1 = model.evaluate_dataset(eval_dataset)
        #loss, acc, f1 = pyt.evaluate_model(model, eval_dataset, loss_fn, metrics=metrics_list)
        print('  Cross-val dataset  -> loss: %.4f - acc: %.4f - f1: %.4f' % (loss, acc, f1))
        loss, acc, f1 = model.evaluate(X_test, y_test)
        #loss, acc, f1 = model.evaluate_dataset(test_dataset)
        #loss, acc, f1 = pyt.evaluate_model(model, test_dataset, loss_fn, metrics=metrics_list)
        print('  Test dataset  -> loss: %.4f - acc: %.4f - f1: %.4f' % (loss, acc, f1))

        # loss, acc = model.evaluate(eval_dataset, loss_fn)
        # print('  Cross-val dataset -> loss: %.4f - acc: %.4f' % (loss, acc)) 
        # loss, acc = model.evaluate(test_dataset, loss_fn)
        # print('  Test dataset      -> loss: %.4f - acc: %.4f' % (loss, acc))

        # save model state
        model.save(MODEL_SAVE_NAME)
        del model

    if DO_PREDICTION:
        print('\nRunning predictions...')
        model = pyt.load_model(MODEL_SAVE_NAME)
        #model.compile(loss=loss_fn, optimizer=optimizer, metrics=['acc','f1'])

        # _, all_preds, all_labels = pyt.predict_dataset(model, test_dataset)
        #y_pred, y_true = model.predict_dataset(test_dataset)
        y_pred = model.predict(X_test)
        # we have just 30 elements in dataset, showing ALL
        print('Sample labels: ', y_test)
        print('Sample predictions: ', y_pred)
        print('We got %d/%d incorrect!' % ((y_test != y_pred).sum(), len(y_test)))

if __name__ == "__main__":
    main()





