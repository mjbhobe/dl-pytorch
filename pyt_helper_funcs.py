# encoding='utf-8'
"""
pyt_helper_funcs.py - generic helper functions that can be used across Pytorch code
    - provides helper functions to create various layers (pre-initialized weights & biases)
    - provides functions to train/evaluate models
    - provides a list of metrics that we can use during training (acc, precision, recall, mse etc.)
    - provides PytModule class, which helps provide a Keras-like training mechanism
    - provides classes for custom Datasets (for use with pandas DataFrame & Numpy arrays)

@author: Manish Bhobe
This code is shared with MIT license for educational purposes only. Use at your own risk!!
I am not responsible if your computer explodes of GPU gets fried :P

Usage:
  - Copy this file into a directory in sys.path
  - import the file into your code - I use this syntax
       import pyt_helper_funcs as pyt
"""
import warnings
warnings.filterwarnings('ignore')

import sys, os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# torch imports
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary
from torch.utils.data.dataset import Dataset
 
# seed
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed);

# -----------------------------------------------------------------------------
# helper function to create various layers of model
# -----------------------------------------------------------------------------
def Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
           dilation=1, groups=1, bias=True, padding_mode='zeros'):
    """
    (convenience function)
    creates a nn.Conv2d layer, with weights initiated using glorot_uniform initializer
    and bias initialized using zeros initializer
    @params:
        - same as nn.Conv2d params
    @returns:
        - instance of nn.Conv2d layer, with weights initialized using xavier_uniform
          initializer and bias initialized using zeros initializer
    """
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias, padding_mode=padding_mode)
    # @see: https://msdn.microsoft.com/en-us/magazine/mt833293.aspx for example
    torch.nn.init.xavier_uniform_(layer.weight)
    if bias: torch.nn.init.zeros_(layer.bias)
    return layer

def Dense(in_nodes, out_nodes, bias=True):
    """
    (convenience function)
    creates a fully connected layer
    @params:
      - in_nodes: # of nodes from pervious layer
      - out_nodes: # of nodes in this layer
    @returns:
      - an instance of nn.Linear class with weights (params) initialized
        using xavier_uniform initializer & bias initialized with zeros
    """
    layer = nn.Linear(in_nodes, out_nodes)
    # @see: https://msdn.microsoft.com/en-us/magazine/mt833293.aspx for example
    torch.nn.init.xavier_uniform_(layer.weight)
    if bias: torch.nn.init.zeros_(layer.bias)
    return layer

def Linear(in_nodes, out_nodes, bias=True):
    """
    another shortcut for dense(in_nodes, out_nodes)
    """
    return Dense(in_nodes, out_nodes, bias)

def Flatten(x):
    """
    (convenience function)
    Flattens out the previous layer. Normally used between Conv2D/MaxPooling2D or LSTM layers
    and Linear/Dense layers
    """
    return x.view(x.shape[0],-1)

# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def epsilon():
    return torch.tensor(1e-7)

def accuracy(logits, labels):
    """
    computes accuracy given logits (computed probabilities) & labels (actual values)
    @params:
        - logits: predictions computed from call to model.forward(...) (Tensor)
        - labels: actual values (labels) (Tensor)
    @returns:
        computed accuracy value
           accuracy = (correct_predictions) / len(labels)
    """
    if logits.size()[1] == 1:
        # binary classification case (just 2 classes)
        predicted = torch.round(logits.data).reshape(-1)
    else:
        vals, predicted = torch.max(logits.data, 1)
    
    #_, predicted = torch.max(logits.data, 1)
    total_count = labels.size(0)

    y_pred = predicted.long()
    if len(labels.shape) > 1:
        y_true = labels.reshape(-1).long()  # flatten
    else:
        y_true = labels.long()

    correct_predictions = (y_pred == y_true).sum().item()
    # accuracy is the fraction of correct predictions to total_count
    acc = (correct_predictions / total_count)
    return acc

def precision(logits, labels):
    """
    computes precision metric (for BINARY classification)
        - logits: predictions computed from call to model.forward(...) (Tensor)
        - labels: actual values (labels) (Tensor)
    NOTE: in this version, we limit precision/recall/f1-score calc to binary classification
    @returns:    # eps=1e-10
    # prec = true_positives / (predicted_positives + eps)
        computed precision value
           precision = true_positives / (predicted_positives + epsilon)
    """
    if logits.size()[1] == 1:
        # binary classification case (just 2 classes)
        predicted = torch.round(logits.data).reshape(-1)
    else:
        vals, predicted = torch.max(logits.data, 1)

    y_pred = predicted.long()
    if len(labels.shape) > 1:
        y_true = labels.reshape(-1).long()  # flatten
    else:
        y_true = labels.long()
    true_positives = torch.sum(torch.clamp(y_true * y_pred, 0, 1))
    predicted_positives = torch.sum(torch.clamp(y_pred, 0, 1))

    prec = true_positives / (predicted_positives + epsilon())
    return prec    

def recall(logits, labels):
    """
    computes precision metric (for binary classification)
        - logits: predictions computed from call to model.forward(...) (Tensor)
        - labels: actual values (labels) (Tensor)
    @returns:
        computed recall value
           precision = true_positives / (all_positives + epsilon)
    """
    if logits.size()[1] == 1:
        # binary classification case (just 2 classes)
        predicted = torch.round(logits.data).reshape(-1)
    else:
        vals, predicted = torch.max(logits.data, 1)

    y_pred = predicted.long()
    if len(labels.shape) > 1:
        y_true = labels.reshape(-1).long()   # flatten
    else:
        y_true = labels.long()
    true_positives = torch.sum(torch.clamp(y_true * y_pred, 0, 1))
    all_positives = torch.sum(torch.clamp(y_true, 0, 1))

    rec = true_positives / (all_positives + epsilon())
    return rec    

def f1_score2(logits, labels):
    """
    computes F1 score (for binary classification)
        - logits: predictions computed from call to model.forward(...) (Tensor)
        - labels: actual values (labels) (Tensor)
    @returns:
        computed F1 score value
           f1 = 2*((precision*recall)/(precision+recall))
    """

    prec = precision(logits, labels)
    rec = recall(logits, labels)
    f1 = 2 * ((prec * rec) / (prec + rec + epsilon()))
    return f1

def roc_auc(logits, labels):
    """
    computes roc_auc score (for BINARY classification)
        - logits: predictions computed from call to model.forward(...) (Tensor)
        - labels: actual values (labels) (Tensor)
    @returns:
        computed roc_auc score
    """
    _, predicted = torch.max(logits.data, 1)
    y_true = labels.cpu().numpy()
    y_pred = predicted.cpu().numpy()
    rcac = roc_auc_score(y_true, y_pred)
    return rcac

def mse(predictions, actuals):
    """
    computes mean-squared-error
        - predictions: predictions computed from call to model.forward(...) (Tensor)
        - actuals: actual values (labels) (Tensor)
    @returns:
        computed mse = sum((actuals - predictions)**2) / actuals.size(0)
    """
    diff = actuals - predictions
    mse_err = torch.sum(diff * diff) / (diff.numel() + epsilon())
    return mse_err

def rmse(predictions, actuals):
    """
    computes root-mean-squared-error
        - predictions: predictions computed from call to model.forward(...) (Tensor)
        - actuals: actual values (labels) (Tensor)
    @returns:
        computed rmse = sqrt(mse(predictions, actuals))
    """    
    rmse_err = torch.sqrt(mse(predictions, actuals))
    return rmse_err

def mae(predictions, actuals):
    """
    computes mean absolute error
        - predictions: predictions computed from call to model.forward(...) (Tensor)
        - actuals: actual values (labels) (Tensor)
    @returns:
        computed mae = sum(abs(predictions - actuals)) / n
    """    
    diff = actuals - prediction
    mae_err = torch.mean(torch.abs(diff))
    return mae_err

METRICS_MAP = {
    'acc': accuracy,
    'accuracy': accuracy,
    'prec': precision,
    'precision': precision,
    'rec': recall,
    'recall': recall,
    'f1': f1_score2,  # f1_score2 to avoid conflict with scklern.metrics.f1_score
    'f1_score': f1_score2,
    'roc_auc': roc_auc,
    'mse': mse,
    'rmse': rmse,
    'mae': mae
}

# -------------------------------------------------------------------------------------
# helper class to implement early stopping
# based on Bjarten's implementation (@see: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py)
# -------------------------------------------------------------------------------------
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, monitor='val_loss', min_delta=0, patience=5, mode='min', verbose=False,
                 save_best_weights=False):
        """
        Args:
            monitor (str): which metric should be monitored (default: 'val_loss')
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement. (default: 0)
            patience (int): How many epochs to wait until after last validation loss improvement. (default: 5)
            mode (str): one of {'min','max'} (default='min') In 'min' mode, training will stop when the quantity 
                monitored has stopped decreasing; in 'max' mode it will stop when the quantity monitored has 
                stopped increasing;
            verbose (bool): If True, prints a message for each validation loss improvement. (default: False)
            save_best_weights (bool): Save state with best weights so far (default: False)
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        if mode not in ['min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown. Using \'min\' instead!' % mode)
            self.mode = 'min'
        self.verbose = verbose
        self.save_best_weights = save_best_weights

        self.monitor_op = np.less if self.mode == 'min' else np.greater
        self.min_delta *= -1 if self.monitor_op == np.less else 1
        self.best_score = np.Inf if self.monitor_op == np.less else -np.Inf
        self.counter = 0
        self.best_epoch = 0
        self.checkpoint_file_path = os.path.join('.', 'model_states', 'checkpoint.pt')
        self.metrics_log = []

        self.early_stop = False

    def __call__(self, model, curr_metric_val, epoch):
        # if not (isinstance(model, PytModule) or isinstance(model, PytModuleWrapper)):
        #     raise TypeError("model should be derived from PytModule or PytModuleWrapper")

        #self.is_wrapped = isinstance(model, PytModuleWrapper)
        
        if self.monitor_op(curr_metric_val - self.min_delta, self.best_score):
            if self.save_best_weights:
                # save model state for restore later
                self.save_checkpoint(model, self.monitor, curr_metric_val)
            self.best_score = curr_metric_val
            self.counter = 0
            self.metrics_log = []
            self.best_epoch = epoch + 1
            if self.verbose:
                print('   EarlyStopping (log): patience counter reset to 0 at epoch %d where best score of \'%s\' is %.3f' % (
                        epoch, self.monitor, self.best_score))
        else:
            self.counter += 1
            if self.verbose:
                print('   EarlyStopping (log): patience counter increased to %d - best_score of \'%s\' is %.3f at epoch %d' % (
                    self.counter, self.monitor, self.best_score, self.best_epoch))
            if self.counter >= self.patience:
                self.early_stop = True
                print('   EarlyStopping: Early stopping training at epoch %d. \'%s\' has not improved for past %d epochs.' % (
                    epoch, self.monitor, self.patience))
                print('     - Best score: %.4f at epoch %d. Last %d scores -> %s' % (
                    self.best_score, self.best_epoch, len(self.metrics_log), self.metrics_log))
            else:
                self.metrics_log.append(curr_metric_val)

    def save_checkpoint(self, model, metric_name, curr_metric_val):
        '''Saves model when validation loss decrease.'''
        if self.verbose: 
            print('   EarlyStopping (log): \'%s\' metric has \'improved\' - from %.4f to %.4f. Saving checkpoint...' % (
                    metric_name, self.best_score, curr_metric_val))
        mod = model
        if isinstance(model, PytModuleWrapper):
            mod = model.model
        torch.save(mod.state_dict(), self.checkpoint_file_path)

# -------------------------------------------------------------------------------------
# helper functions for training model, evaluating performance & making predictions
# -------------------------------------------------------------------------------------
def check_attribs__(model, loss_fn, optimizer=None, check_only_loss=False):
    """ internal helper function - checks various attributes of "model" """
    if loss_fn is None:
        # model instance must have self.loss_fn attribute defined
        try:
            l = model.loss_fn
            if l is None:
                # defined in model, but set to None
                raise ValueError('FATAL ERROR: it appears that you have not set a value for loss_fn ' +
                    'Detected None value for both the loss_fn parameter and module.loss_fn attribute!')
        except AttributeError as e:
            print("FATAL ERROR: when loss_fn parameter is None, the model's instance is expected " +
              "to have the loss function defined with attribute self.loss_fn!\n" +
              "This model's instance does not have a self.loss_fn attribute defined.")
            raise e

    if not check_only_loss:
        if optimizer is None:
            # model instance must have self.optimizer attribute defined
            try:
                o = model.optimizer
                if o is None:
                    raise ValueError('FATAL ERROR: it appears that you have not set a value '  +
                        'for optimizer. Detected None value for both the optimizer parameter and ' +
                        'module.optimizer attribute!')
            except AttributeError as e:
                print("FATAL ERROR: when optimizer parameter is None, the model's instance is expected " +
                  "to have the optimizer function defined with attribute self.optimizer!\n" +
                  "This model's instance does not have a self.optimizer attribute defined.")
                raise e

def compute_metrics__(logits, labels, metrics, batch_metrics, validation_dataset=False):
    """ internal helper functions - computes metrics in an epoch loop """
    for metric_name in metrics:
        metric_value = METRICS_MAP[metric_name](logits, labels)
        if validation_dataset:
            batch_metrics['val_%s' % metric_name] = metric_value #.append(metric_value)
        else:
            batch_metrics[metric_name] = metric_value

def accumulate_metrics__(metrics, cum_metrics, batch_metrics, validation_dataset=False):
    """ internal helper function - "sums" metrics across batches """
    if metrics is not None:
        for metric in metrics:
            if validation_dataset:
                cum_metrics['val_%s' % metric] += batch_metrics['val_%s' % metric]
            else:
                cum_metrics[metric] += batch_metrics[metric]

    # check for loss separately
    if 'loss' not in metrics:
        if validation_dataset:
            cum_metrics['val_loss'] += batch_metrics['val_loss']
        else:
            cum_metrics['loss'] += batch_metrics['loss']
    return cum_metrics

def get_metrics_str__(metrics_list, batch_or_cum_metrics, validation_dataset=False):
    """ internal helper functions: formats metrics for printing to console """
    metrics_str = ''

    for i, metric in enumerate(metrics_list):
        if i > 0:
            metrics_str += ' - %s: %.4f' % (metrics_list[i], batch_or_cum_metrics[metric])
        else:
            metrics_str += '%s: %.4f' % (metrics_list[i], batch_or_cum_metrics[metric])
            
    # append validation metrics too
    if validation_dataset:
        for i, metric in enumerate(metrics_list):
            metrics_str += ' - val_%s: %.4f' % (metrics_list[i], batch_or_cum_metrics['val_%s' % metric])

    return metrics_str

def create_hist_and_metrics_ds__(metrics, include_val_metrics=True):
    """ internal helper functions - create data structures to log epoch metrics, 
        batch metrics & cumulative betch metrics """
    history = {'loss': []}
    batch_metrics = {'loss': 0.0}
    cum_metrics = {'loss': 0.0}

    if include_val_metrics:
        history['val_loss'] = []
        batch_metrics['val_loss'] = 0.0
        cum_metrics['val_loss'] = 0.0

    if metrics is not None and len(metrics) > 0:
        # walk list of metric names & create one entry per metric
        for metric_name in metrics:
            if metric_name not in METRICS_MAP.keys():
                raise ValueError('%s - unrecognized metric!' % metric_name)
            else:
                history[metric_name] = []
                batch_metrics[metric_name] = 0.0
                cum_metrics[metric_name] = 0.0

                if include_val_metrics:
                    history['val_%s' % metric_name] = []
                    batch_metrics['val_%s' % metric_name] = 0.0
                    cum_metrics['val_%s' % metric_name] = 0.0

    return history, batch_metrics, cum_metrics

def train_model(model, train_dataset, loss_fn=None, optimizer=None, validation_split=0.0, validation_dataset=None,
                lr_scheduler=None, epochs=25, batch_size=64, metrics=None, shuffle=True,
                num_workers=0, early_stopping=None):
    """
    Trains model (derived from nn.Module) across epochs using specified loss function,
    optimizer, validation dataset (if any), learning rate scheduler, epochs and batch size etc.
    @parms:
        - model: instance of a model derived from torch.nn.Module class
        - train_dataset: training dataset derived from torchvision.dataset
        - loss_fn: loss function to use when training (optional, default=None)
            if loss_fn == None, then the model class must define an attribute self.loss_fn
            defined as an instance of a loss function
        - optimizer: Pytorch optimizer to use during training (instance of any optimizer
          from the torch.optim package)
            if optimizer == None, then model must define an attribute self.optimizer, which is 
            an instance of any optimizer defined in the torch.optim package
        - validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. 
           The model will set apart this fraction of the training data, will not train on it, and will 
           evaluate the loss and any model metrics on this data at the end of each epoch.
        - validation_dataset: cross-validation dataset to use during training (optional, default=None)
            if validation_dataset is not None, then model is cross trained on test dataset & 
            cross-validation dataset (good practice to always use a cross-validation dataset!)
            (NOTE: validation_dataset will supercede validation_split if both specified)
        - lr_scheduler: learning rate scheduler to use during training (optional, default=None)
            if specified, should be one of the learning rate schedulers (e.g. StepLR) defined
            in the torch.optim.lr_scheduler package
        - epochs (int): number of epochs for which the model should be trained (optional, default=25)
        - batch_size (int): batch size to split the training & cross-validation datasets during 
          training (optional, default=32)
        - metrics (list of strings): metrics to compute (optional, default=None)
            pass a list of strings from one or more of the following ['acc','prec','rec','f1']
            when metrics = None, only loss is computed for training set (and validation set, if any)
            when metrics not None, in addition to loss all specified metrics are computed for training
            set (and validation set, if specified)
        - shuffle (boolean, default = True): determines if the training dataset shuould be shuffled between
            epochs or not. 
        - num_workers (int, default=0): number of worker threads to use when loading datasets internally using 
            DataLoader objects.
        - early_stopping(EarlyStopping, default=None): instance of EarlyStopping class to be passed in if training
            has to be early-stopped based on parameters used to construct instance of EarlyStopping
    @returns:
        - history: dictionary of the loss & accuracy metrics across epochs
            Metrics are saved by key name (see METRICS_MAP) 
            Metrics (across epochs) are accessed by key (e.g. hist['loss'] accesses training loss
            and hist['val_acc'] accesses validation accuracy
            Validation metrics are available ONLY if validation set is provided during training
    """
    try:
        # checks for parameters
        assert isinstance(model, nn.Module), "train_model() works with instances of nn.Module only!"
        assert isinstance(train_dataset, torch.utils.data.Dataset), \
            "train_dataset must be a subclass of torch.utils.data.Dataset"
        assert (0.0 <= validation_split < 1.0), "validation_split must be a float between (0.0, 1.0]"
        if validation_dataset is not None:
            assert isinstance(validation_dataset, torch.utils.data.Dataset), \
                "validation_dataset must be a subclass of torch.utils.data.Dataset"
        check_attribs__(model, loss_fn, optimizer)
        if loss_fn is None: loss_fn = model.loss_fn
        if loss_fn is None:
            raise ValueError("Loss function is not defined. Must pass as paf rameter or define in class")
        if optimizer is None: optimizer = model.optimizer
        if optimizer is None:
            raise ValueError("Optimizer is not defined. Must pass as parameter or define in class")
        if lr_scheduler is not None:
            assert isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler), \
                "lr_scheduler: incorrect type. Expecting class derived from torch.optim._LRScheduler"
        early_stopping_metric = None
        if early_stopping is not None:
            assert isinstance(early_stopping, EarlyStopping), \
                "early_stopping: incorrect type. Expecting instance of EarlyStopping class"
            early_stopping_metric = early_stopping.monitor

        # if validation_split is provided by user, then split train_dataset
        if (validation_split > 0.0) and (validation_dataset is None):
            # NOTE: validation_dataset supercedes validation_split
            num_recs = len(train_dataset)
            train_count = int((1.0 - validation_split) * num_recs)
            val_count = num_recs - train_count
            train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_count, val_count])
            assert (train_dataset is not None) and (len(train_dataset) == train_count), \
                "Something is wrong with validation_split - getting incorrect train_dataset counts!!"
            assert (validation_dataset is not None) and (len(validation_dataset) == val_count), \
                "Something is wrong with validation_split - getting incorrect validation_dataset counts!!"

        # train on GPU if available
        gpu_available = torch.cuda.is_available()

        print('Training on %s...' % ('GPU' if gpu_available else 'CPU'))
        model = model.cuda() if gpu_available else model.cpu()

        if validation_dataset is not None:
            print('Training on %d samples, cross-validating on %d samples' %
                    (len(train_dataset), len(validation_dataset)))
        else:
            print('Training on %d samples' % len(train_dataset))

        tot_samples = len(train_dataset)
        len_tot_samples = len(str(tot_samples))

        # create data structurs to hold batch metrics, epoch metrics etc.
        history, batch_metrics, cum_metrics = \
            create_hist_and_metrics_ds__(metrics, validation_dataset is not None)

        metrics_list = ['loss']
        if metrics is not None:
            metrics_list = metrics_list + metrics
            if early_stopping_metric is not None:
                metrics_list_check = []
                for metric in metrics_list:
                    metrics_list_check.append(metric)
                    metrics_list_check.append('val_%s' % metric)
                assert early_stopping_metric in metrics_list_check, \
                    "early stopping metric (%s) is not logged during training!" % early_stopping_metric

        len_num_epochs = len(str(epochs))

        curr_lr = None

        if lr_scheduler is not None:
            curr_lr = lr_scheduler.get_lr()
            print('Using learning rate {}'.format(curr_lr))

        for epoch in range(epochs):
            model.train()  # model is training, so batch normalization & dropouts can be applied
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       shuffle=shuffle, num_workers=num_workers)
            num_batches = 0
            samples = 0

            # zero out batch & cum metrics for next epoch
            for metric_name in metrics_list:
                batch_metrics[metric_name] = 0.0
                cum_metrics[metric_name] = 0.0
                if validation_dataset is not None:
                    batch_metrics['val_%s' % metric_name] = 0.0
                    cum_metrics['val_%s' % metric_name] = 0.0

            # iterate over the training dataset
            for batch_no, (data, labels) in enumerate(train_loader):
                # move to GPU if available
                data = data.cuda() if gpu_available else data.cpu()
                labels = labels.cuda() if gpu_available else labels.cpu()

                # clear accummulated gradients
                optimizer.zero_grad()
                # make a forward pass
                logits = model(data)
                # apply loss function
                loss_tensor = loss_fn(logits, labels)
                # compute gradients
                loss_tensor.backward()
                # update weights
                optimizer.step()
                batch_loss = loss_tensor.item()

                # compute metrics for batch + accumulate metrics across batches
                batch_metrics['loss'] = batch_loss
                if metrics is not None:
                    compute_metrics__(logits, labels, metrics, batch_metrics, validation_dataset=False)
                # same as cum_netrics[metric_name] += batch_metric[metric_name] across all metrics
                cum_metrics = accumulate_metrics__(metrics_list, cum_metrics, batch_metrics, validation_dataset=False)

                samples += len(labels)
                num_batches += 1

                # display progress
                metrics_str = get_metrics_str__(metrics_list, batch_metrics, validation_dataset=False)
                print('\rEpoch (%*d/%*d): (%*d/%*d) -> %s' %
                            (len_num_epochs, epoch+1, len_num_epochs, epochs,
                             len_tot_samples, samples, len_tot_samples, tot_samples,
                             metrics_str),
                        end='', flush=True)
            else:
                # compute average metrics across all batches of train_loader
                for metric_name in metrics_list:
                    cum_metrics[metric_name] = cum_metrics[metric_name] / num_batches
                    history[metric_name].append(cum_metrics[metric_name])

                # display average training metrics for this epoch
                metrics_str = get_metrics_str__(metrics_list, cum_metrics, validation_dataset=False)
                print('\rEpoch (%*d/%*d): (%*d/%*d) -> %s' %
                        (len_num_epochs, epoch+1, len_num_epochs, epochs,
                            len_tot_samples, samples, len_tot_samples, tot_samples,
                            metrics_str),
                    end='' if validation_dataset is not None else '\n', flush=True)

                if validation_dataset is not None:
                    model.eval()  # mark model as evaluating - don't apply any dropouts
                    with torch.no_grad():
                        # run through the validation dataset
                        val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,
                                                                 shuffle=False, num_workers=num_workers)
                        num_val_batches = 0

                        for val_data, val_labels in val_loader:
                            #val_data, val_labels = val_data.to(device), val_labels.to(device)
                            val_data = val_data.cuda() if gpu_available else val_data.cpu()
                            val_labels = val_labels.cuda() if gpu_available else val_labels.cpu()

                            # forward pass
                            val_logits = model(val_data)
                            # apply loss function
                            loss_tensor = loss_fn(val_logits, val_labels)
                            batch_loss = loss_tensor.item()

                            # calculate all metrics for validation dataset batch
                            batch_metrics['val_loss'] = batch_loss
                            if metrics is not None:
                                compute_metrics__(val_logits, val_labels, metrics, batch_metrics, validation_dataset=True)
                            # same as cum_metrics[val_metric_name] += batch_metrics[val_metric_name] for all metrics
                            cum_metrics = accumulate_metrics__(metrics_list, cum_metrics, batch_metrics, validation_dataset=True)

                            num_val_batches += 1
                        else:
                            # average metrics across all val-dataset batches
                            for metric_name in metrics_list:
                                cum_metrics['val_%s' % metric_name] = cum_metrics['val_%s' % metric_name] / num_val_batches
                                history['val_%s' % metric_name].append(cum_metrics['val_%s' % metric_name])

                            # display train + val set metrics    
                            metrics_str = get_metrics_str__(metrics_list, cum_metrics, validation_dataset=True)
                            print('\rEpoch (%*d/%*d): (%*d/%*d) -> %s' %
                                        (len_num_epochs, epoch+1, len_num_epochs, epochs,
                                         len_tot_samples, samples, len_tot_samples, tot_samples,
                                         metrics_str),
                                    flush=True)
            
            # check for early stopping
            if early_stopping is not None:
                curr_metric_val = history[early_stopping_metric][-1]
                early_stopping(model, curr_metric_val, epoch)
                if early_stopping.early_stop:
                    print("Early stopping training at epoch %d" % epoch)
                    if early_stopping.save_best_weights:
                        mod = model
                        if isinstance(model, PytModuleWrapper):
                            mod = model.model
                        mod.load_state_dict(torch.load(early_stopping.checkpoint_file_path))
                    return history
                
            # step the learning rate scheduler at end of epoch
            if (lr_scheduler is not None) and (epoch <= epochs-1):
                lr_scheduler.step()
                step_lr = lr_scheduler.get_lr()
                #print('   StepLR (log): curr_lr = {}, new_lr = {}'.format(curr_lr, step_lr))
                if np.round(np.array(step_lr),10) != np.round(np.array(curr_lr),10):
                    print('Stepping learning rate to {}'.format(step_lr))
                    curr_lr = step_lr

        return history
    finally:
        model = model.cpu()

def train_model_xy(model, X_train, y_train, loss_fn=None, optimizer=None, validation_split=0.0, validation_dataset=None,
                   lr_scheduler=None, epochs=25, batch_size=64, metrics=None, shuffle=True):
    raise NotImplementedError("train_model_xy should not be used!!")
    try:
        # checks for parameters
        assert isinstance(model, nn.Module), "train_model() works with instances of nn.Module only!"
        assert isinstance(X_train, np.ndarray), \
            "X_train must be an instance of Numpy array"
        assert isinstance(y_train, np.ndarray), \
            "y_train must be an instance of Numpy array or torch.Tensor"
        if validation_dataset is not None:
            assert isinstance(validation_dataset, tuple) and (len(validation_dataset) == 2), "validation_dataset must be a 2 element tuple"
            assert (isinstance(validation_dataset[0], np.ndarray) and isinstance(validation_dataset[1], np.ndarray)), \
                "validation_dataset must hold only Numpy arrays"
        check_attribs__(model, loss_fn, optimizer)
        if loss_fn is None: loss_fn = model.loss_fn
        if loss_fn is None:
            raise ValueError("Loss function is not defined. Must pass as parameter or define in class")
        if optimizer is None: optimizer = model.optimizer
        if optimizer is None:
            raise ValueError("Optimizer is not defined. Must pass as parameter or define in class")
        if metrics is None: metrics = model.metrics_list
        if lr_scheduler is not None:
            assert isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler), \
                "lr_scheduler: incorrect type. Expecting class derived from torch.optim._LRScheduler"

        # train on GPU if available
        gpu_available = torch.cuda.is_available()

        print('Training on %s...' % ('GPU' if gpu_available else 'CPU'))
        model = model.cuda() if gpu_available else model.cpu()
        X_val, y_val = None, None

        if validation_dataset is not None:
            X_val, y_val = validation_dataset[0], validation_dataset[1]
            print('Training on %d samples, cross-validating on %d samples' %
                    (X_train.shape[0], X_val.shape[0]))
        else:
            print('Training on %d samples' % X_train.shape[0])

        tot_samples = X_train.shape[0]
        len_tot_samples = len(str(tot_samples))

        # create data structurs to hold batch metrics, epoch metrics etc.
        history, batch_metrics, cum_metrics = \
            create_hist_and_metrics_ds__(metrics, validation_dataset is not None)

        metrics_list = ['loss']
        if metrics is not None:
            metrics_list = metrics_list + metrics

        len_num_epochs = len(str(epochs))

        if lr_scheduler is not None:
            print('Using learning rate {}'.format(lr_scheduler.get_lr()))

        num_batches = tot_samples // batch_size
        num_batches += 1 if tot_samples % batch_size > 0 else 0

        for epoch in range(epochs):
            model.train()  # model is training, so batch normalization & dropouts can be applied
            samples = 0

            if shuffle:
                (X_train, y_train) = shuffle(X_train, y_train)

            # zero out batch & cum metrics for next epoch
            for metric_name in metrics_list:
                batch_metrics[metric_name] = 0.0
                cum_metrics[metric_name] = 0.0
                if validation_dataset is not None:
                    batch_metrics['val_%s' % metric_name] = 0.0
                    cum_metrics['val_%s' % metric_name] = 0.0

            # iterate over the training dataset
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                # grab the next batch of data
                X_batch = X_train[start:end, :]
                y_batch = y_train[start:end]
                data = Variable(torch.FloatTensor(X_batch))
                if (y_batch.dtype == np.int) or (y_batch.dtype == np.long):
                    labels = Variable(torch.LongTensor(y_batch))
                else:
                    labels = Variable(torch.FloatTensor(y_batch))

                # move to GPU if available
                data = data.cuda() if gpu_available else data.cpu()
                labels = labels.cuda() if gpu_available else labels.cpu()

                # clear accummulated gradients
                optimizer.zero_grad()
                # make a forward pass
                logits = model(data)
                # apply loss function
                loss_tensor = loss_fn(logits, labels)
                # compute gradients
                loss_tensor.backward()
                # update weights
                optimizer.step()
                batch_loss = loss_tensor.item()

                # compute metrics for batch + accumulate metrics across batches
                batch_metrics['loss'] = batch_loss
                if metrics is not None:
                    compute_metrics__(logits, labels, metrics, batch_metrics, validation_dataset=False)
                # same as cum_netrics[metric_name] += batch_metric[metric_name] across all metrics
                cum_metrics = accumulate_metrics__(metrics_list, cum_metrics, batch_metrics, validation_dataset=False)

                samples += len(labels)

                # display progress
                metrics_str = get_metrics_str__(metrics_list, batch_metrics, validation_dataset=False)
                print('\rEpoch (%*d/%*d): (%*d/%*d) -> %s' %
                            (len_num_epochs, epoch+1, len_num_epochs, epochs,
                             len_tot_samples, samples, len_tot_samples, tot_samples,
                             metrics_str),
                        end='', flush=True)
            else:
                # compute average metrics across all batches of train_loader
                for metric_name in metrics_list:
                    cum_metrics[metric_name] = cum_metrics[metric_name] / num_batches
                    history[metric_name].append(cum_metrics[metric_name])

                # display average training metrics for this epoch
                metrics_str = get_metrics_str__(metrics_list, cum_metrics, validation_dataset=False)
                print('\rEpoch (%*d/%*d): (%*d/%*d) -> %s' %
                        (len_num_epochs, epoch+1, len_num_epochs, epochs,
                            len_tot_samples, samples, len_tot_samples, tot_samples,
                            metrics_str),
                    end='' if validation_dataset is not None else '\n', flush=True)

                if validation_dataset is not None:
                    model.eval()  # mark model as evaluating - don't apply any dropouts
                    with torch.no_grad():
                        # run through the validation dataset
                        num_val_batches = 0

                        for batch_i in range(0, X_val.shape[0], batch_size):
                            # grab the next batch of data
                            X_val_batch = X_val[batch_i:batch_i + batch_size,:]
                            y_val_batch = y_val[batch_i:batch_i + batch_size]
                            val_data = Variable(torch.FloatTensor(X_val_batch))
                            if (y_val_batch.dtype == np.int) or (y_val_batch.dtype == np.long):
                                val_labels = Variable(torch.LongTensor(y_val_batch))
                            else:
                                val_labels = Variable(torch.FloatTensor(y_val_batch))

                            val_data = val_data.cuda() if gpu_available else val_data.cpu()
                            val_labels = val_labels.cuda() if gpu_available else val_labels.cpu()

                            # forward pass
                            val_logits = model(val_data)
                            # apply loss function
                            loss_tensor = loss_fn(val_logits, val_labels)
                            batch_loss = loss_tensor.item()

                            # calculate all metrics for validation dataset batch
                            batch_metrics['val_loss'] = batch_loss
                            if metrics is not None:
                                compute_metrics__(val_logits, val_labels, metrics, batch_metrics, validation_dataset=True)
                            # same as cum_metrics[val_metric_name] += batch_metrics[val_metric_name] for all metrics
                            cum_metrics = accumulate_metrics__(metrics_list, cum_metrics, batch_metrics, validation_dataset=True)

                            num_val_batches += 1
                        else:
                            # average metrics across all val-dataset batches
                            for metric_name in metrics_list:
                                cum_metrics['val_%s' % metric_name] = cum_metrics['val_%s' % metric_name] / num_val_batches
                                history['val_%s' % metric_name].append(cum_metrics['val_%s' % metric_name])

                            # display train + val set metrics    
                            metrics_str = get_metrics_str__(metrics_list, cum_metrics, validation_dataset=True)
                            print('\rEpoch (%*d/%*d): (%*d/%*d) -> %s' %
                                        (len_num_epochs, epoch+1, len_num_epochs, epochs,
                                         len_tot_samples, samples, len_tot_samples, tot_samples,
                                         metrics_str),
                                    flush=True)
            
            # step the learning rate scheduler at end of epoch
            if lr_scheduler is not None:
                lr_scheduler.step()
                print('Stepping learning rate to {}'.format(lr_scheduler.get_lr()))
        return history
    finally:
        model = model.cpu()

def evaluate_model(model, dataset, loss_fn=None, batch_size=64, metrics=None, num_workers=0):
    """ evaluate's model performance against dataset provided
    @params:
        - model: instance of model derived from nn.Model (or instance of pyt.PytModel or pyt.PytSequential)
        - dataset: instance of dataset to evaluate against (derived from torch.utils.data.Dataset)
        - loss_fn (optional, default=None): loss function to use during evaluation
             If not provided the model's loss function (i.e. model.loss_fn) is used
             asserts error if both are not provided!
        - batch_size (optional, default=64): batch size to use during evaluation
        - metrics (optional, default=None): list of metrics to evaluate 
            (e.g.: metrics=['acc','f1']) evaluates accuracy & f1-score
            Loss is ALWAYS evaluated, even when metrics=None
    @returns:
        - value of loss across dataset, if metrics=None (single value)
        - value of loss + list of metrics (in order provided), if metrics list is provided
          (e.g. if metrics=['acc', 'f1'], then a list of 3 values will be returned loss, accuracy & f1-score)
    """
    try:
        assert isinstance(model, nn.Module), \
            "evaluate_model() works with instances of nn.Module only!"
        assert isinstance(dataset, torch.utils.data.Dataset), \
            "dataset must be a subclass of torch.utils.data.Dataset"
        check_attribs__(model, loss_fn, check_only_loss=True)
        if loss_fn is None: loss_fn = model.loss_fn

        # evaluate on GPU if available
        gpu_available = torch.cuda.is_available()
        model = model.cuda() if gpu_available else model.cpu()

        samples, num_batches = 0, 0
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        tot_samples = len(dataset)
        len_tot_samples = len(str(tot_samples))

        history, batch_metrics, cum_metrics = \
            create_hist_and_metrics_ds__(metrics, dataset is not None)

        metrics_list = ['loss']
        if metrics is not None:
            metrics_list = metrics_list + metrics

        with torch.no_grad():
            model.eval()
            for data, labels in loader:
                data = data.cuda() if gpu_available else data.cpu()
                labels = labels.cuda() if gpu_available else labels.cpu()

                # forward pass
                logits = model(data)
                # compute batch loss
                loss_tensor = loss_fn(logits, labels)
                batch_loss = loss_tensor.item()

                # compute all metrics for this batch
                compute_metrics__(logits, labels, metrics, batch_metrics, validation_dataset=False)
                batch_metrics['loss'] = batch_loss
                # same as cum_metrics[metric_name] += batch_metrics[metric_name] for all metrics
                cum_metrics = accumulate_metrics__(metrics_list, cum_metrics, batch_metrics, validation_dataset=False)

                samples += len(labels)
                num_batches += 1

                # display progress for this batch
                metrics_str = get_metrics_str__(metrics_list, batch_metrics, validation_dataset=False)
                print('\rEvaluating (%*d/%*d) -> %s' %
                        (len_tot_samples, samples, len_tot_samples, tot_samples,
                         metrics_str),
                      end='', flush=True)
            else:
                # compute average of all metrics provided in metrics list
                for metric_name in metrics_list:
                    cum_metrics[metric_name] = cum_metrics[metric_name] / num_batches
                
                metrics_str = get_metrics_str__(metrics_list, cum_metrics, validation_dataset=False)
                print('\rEvaluating (%*d/%*d) -> %s' %
                        (len_tot_samples, tot_samples, len_tot_samples, tot_samples,
                         metrics_str),
                      flush=True)
                      
        if metrics is None:
            return cum_metrics['loss']
        else:
            ret_metrics = []
            for metric_name in metrics_list:
                ret_metrics.append(cum_metrics[metric_name])
            return ret_metrics
    finally:
        model = model.cpu()

def predict_dataset(model, dataset, batch_size=64, num_workers=0):
    """ runs prediction on dataset (use for classification ONLY)
    @params:
        - model: instance of model derived from nn.Model (or instance of pyt.PytModel or pyt.PytSequential)
        - dataset: instance of dataset to evaluate against (derived from torch.utils.data.Dataset)
        - batch_size (optional, default=64): batch size to use during evaluation
    @returns:
        - tuple of Numpy Arrays of class predictions & labels
    """
    try:
        assert isinstance(model, nn.Module), \
            "predict_dataset() works with instances of nn.Module only!"
        assert isinstance(dataset, torch.utils.data.Dataset), \
            "dataset must be a subclass of torch.utils.data.Dataset"

        # run on GPU, if available
        gpu_available = torch.cuda.is_available()
        model = model.cuda() if gpu_available else model.cpu()

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        preds, actuals = [], []

        for images, labels in loader:
            images = images.cuda() if gpu_available else images.cpu()
            labels = labels.cuda() if gpu_available else labels.cpu()

            # run prediction
            with torch.no_grad():
                model.eval()
                logits = model(images)
                batch_preds = list(logits.to("cpu").numpy())
                batch_actuals = list(labels.to("cpu").numpy())
                preds.extend(batch_preds)
                actuals.extend(batch_actuals)
        return np.array(preds), np.array(actuals)
    finally:
        model = model.cpu()

def predict(model, data):
    """ runs predictions on Numpy Array (use for classification ONLY)
    @params:
        - model: instance of model derived from nn.Model (or instance of pyt.PytModel or pyt.PytSequential)
        - data: Numpy array of values on which predictions should be run
    @returns:
        - Numpy array of class predictions
    """
    try:
        assert isinstance(model, nn.Module), \
            "predict() works with instances of nn.Module only!"
        assert ((isinstance(data, np.ndarray)) or (isinstance(data, torch.Tensor))), \
            "data must be an instance of Numpy ndarray or torch.tensor"
        # train on GPU if you can
        gpu_available = torch.cuda.is_available()
        model = model.cuda() if gpu_available else model.cpu()

        # run prediction
        with torch.no_grad():
            model.eval()
            if isinstance(data, np.ndarray):
                #data = data.astype(np.float32)
                data = torch.tensor(data, dtype=torch.float32)
            data = data.cuda() if gpu_available else data.cpu()
            # forward pass
            logits = model(data)
            # take max for each row along columns to get class predictions
            #vals, preds = torch.max(logits.data, 1) #torch.max(logits, 1)
            #preds = np.array(preds.cpu().numpy())
            preds = np.array(logits.cpu().numpy())
        return preds
    finally:
        model = model.cpu()

def save_model(model, model_save_name, model_save_dir=os.path.join('.','model_states')):
    """ saves Pytorch model to disk (file with .pt extension) 
    @params:
        - model: instance of model derived from nn.Model (or instance of pyt.PytModel or pyt.PytSequential)
        - model_save_name: name of file or complete path of file to save model to 
          (NOTE: this file is overwritten without warning!)
        - model_save_dir (optional, defaul='./model_states'): folder to save Pytorch model to
           used only if model_save_name is just a name of file
           ignored if model_save_name is complete path to a file
    """
    if not model_save_name.endswith('.pt'):
        model_save_name = model_save_name + '.pt'

    # model_save_name could be just a file name or complete path
    if (len(os.path.dirname(model_save_name)) == 0):
        # only file name specified e.g. pyt_model.pt. We'll use model_save_dir to save
        if not os.path.exists(model_save_dir):
            # check if save_dir exists, else create it
            try:
                os.mkdir(model_save_dir)
            except OSError as err:
                print("Unable to create folder {} to save Keras model. Can't continue!".format(model_save_dir))
                raise err
        model_save_path = os.path.join(model_save_dir, model_save_name)
    else:
        # user passed in complete path e.g. './save_states/kr_model.h5'
        model_save_path = model_save_name

    torch.save(model, model_save_path)
    print('Pytorch model saved to %s' % model_save_path)

def load_model(model_save_name, model_save_dir='./model_states'):
    """ loads model from disk and create a complete instance from saved state
    @params:
        - model_save_name: name of file or complete path of file to save model to 
          (NOTE: this file is overwritten without warning!)
        - model_save_dir (optional, defaul='./model_states'): folder to save Pytorch model to
           used only if model_save_name is just a name of file
           ignored if model_save_name is complete path to a file
    @returns:
        - 'ready-to-go' instance of model restored from saved state
    """
    if not model_save_name.endswith('.pt'):
        model_save_name = model_save_name + '.pt'

    # base_file_name could be just a file name or complete path
    if (len(os.path.dirname(model_save_name)) == 0):
        # only file name specified e.g. pyt_model.pt
        model_save_path = os.path.join(model_save_dir, model_save_name)
    else:
        # user passed in complete path e.g. './save_states/kr_model.h5'
        model_save_path = model_save_name

    if not os.path.exists(model_save_path):
        raise IOError('Cannot find model state file at %s!' % model_save_path)

    model = torch.load(model_save_path)
    model.eval()
    print('Pytorch model loaded from %s' % model_save_path)
    return model

def show_plots(history, plot_title=None, fig_size=None):
    """ Useful function to view plot of loss values & accuracies across the various epochs
        Works with the history object returned by the train_model(...) call """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use('seaborn')
    sns.set_style('darkgrid')

    assert type(history) is dict

    # NOTE: the history object should always have loss (for training data), but MAY have
    # val_loss for validation data
    loss_vals = history['loss']
    val_loss_vals = history['val_loss'] if 'val_loss' in history.keys() else None

    # accuracy is an optional metric chosen by user
    acc_vals = history['acc'] if 'acc' in history.keys() else None
    if acc_vals is None:
        # try 'accuracy' key, could be using Tensorflow 2.0 backend!
        acc_vals = history['accuracy'] if 'acc' in history.keys() else None

    val_acc_vals = history['val_acc'] if 'val_acc' in history.keys() else None
    if val_acc_vals is None:
        # try 'val_accuracy' key, could be using Tensorflow 2.0 backend!
        val_acc_vals = history['val_accuracy'] if 'val_accuracy' in history.keys() else None

    epochs = range(1, len(history['loss']) + 1)

    col_count = 1 if ((acc_vals is None) and (val_acc_vals is None)) else 2

    with sns.axes_style("darkgrid"):
        sns.set_context("notebook", font_scale=1.1)
        sns.set_style({"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})

        f, ax = plt.subplots(nrows=1, ncols=col_count, figsize=((16, 5) if fig_size is None else fig_size))

        axs = ax[0] if col_count == 2 else ax

        # plot losses on ax[0]
        axs.plot(epochs, loss_vals, label='Training Loss')
        if val_loss_vals is not None:
            axs.plot(epochs, val_loss_vals, label='Validation Loss')
            axs.set_title('Training & Validation Loss')
            axs.legend(loc='best')
        else:
            axs.set_title('Training Loss')

        axs.set_xlabel('Epochs')
        axs.set_ylabel('Loss')
        axs.grid(True)

        # plot accuracies, if exist
        if col_count == 2:
            ax[1].plot(epochs, acc_vals, label='Training Accuracy')
            if val_acc_vals is not None:
                #ax[1].plot(epochs, val_acc_vals, color='firebrick', marker='*', label='Validation Accuracy')
                ax[1].plot(epochs, val_acc_vals, label='Validation Accuracy')
                ax[1].set_title('Training & Validation Accuracy')
                ax[1].legend(loc='best')
            else:
                ax[1].set_title('Training Accuracy')

            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Accuracy')
            ax[1].grid(True)

        if plot_title is not None:
            plt.suptitle(plot_title)

        plt.show()
        plt.close()

    # delete locals from heap before exiting (to save some memory!)
    del loss_vals, epochs, acc_vals
    if val_loss_vals is not None:
        del val_loss_vals
    if val_acc_vals is not None:
        del val_acc_vals

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# --------------------------------------------------------------------------------------------
# Utility classes
# --------------------------------------------------------------------------------------------

class PandasDataset(Dataset):
    """ Dataset class from a pandas Dataframe """
    def __init__(self, df, target_col_name, target_col_type=np.int):
        """ 
        @params:
            - df: pandas dataframe from which Dataset is to be created
            - target_col_name: column name of Pandas dataframe to use as target/label
            - target_col_type (optional, default: np.int): datatype of target/label column 
        """
        assert isinstance(df, pd.DataFrame), "df parameter should be an instance of pd.DataFrame"
        assert isinstance(target_col_name, str), "target_col_name parameter should be a string"

        self.df = df
        assert target_col_name in self.df.columns, "%s - not a valid column name" % target_col_name
        self.target_col_name = target_col_name
        self.target_col_type = target_col_type
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        ## NOTE: set data.type = np.float32 & label to np.int, else I get wierd errors!
        data = row[self.df.columns != self.target_col_name].values.astype(np.float32)
        label = row[self.target_col_name].astype(self.target_col_type)
        return (data, label)

class XyDataset(Dataset):
    """ Dataset class from a Numpy arrays 
    @params:
        - X: numpy array for the features (m rows X n feature cols Numpy array)
        - y: numpy array for labels/targets (m rows X 1 array OR a flattened array of m values)
    """
    def __init__(self, X, y, y_dtype):
        assert isinstance(X, np.ndarray), "X parameter should be an instance of Numpy array"
        assert isinstance(y, np.ndarray), "y parameter should be an instance of Numpy array"

        self.X = X
        self.y = y
        self.y_dtype = y_dtype
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        ## NOTE: set data.type = np.float32 & label to np.int, else I get wierd errors!
        data = self.X[index].astype(np.float32)
        label = self.y[index].astype(self.y_dtype)
        return (data, label)

class PytModule(nn.Module):
    """
    A class that you can inherit from to define your project's model
    Inheriting from this class provides a Keras-like interface for training model, evaluating model performance
    and for generating predictions.
        - As usual, you must override the constructor and the forward() method in your derived class.
        - You may provide a compile() function to set loss, optimizer and metrics at one location, else
          you will have to provide these as parameters to the fit(), evaluate() calls
    
    This class provides the following convenience methods that call functions defined above
    You call this class's functions with the same parameters, except model, which is passed as 'self'
       - compile(loss, optimizer, metrics=None) - keras compile() like function. Sets the loss function,
            optimizer and metrics (if any) to use during testing. Note that loss is always measured.
       - fit() - trains the model on numpy arrays (X = data & y = labels). 
       - fit_dataset() - trains model on torch.utils.data.Dataset instance. 
       - evaluate() - evaluate on numpy arrays (X & y)
       - evaluate_dataset() - evaluate on torch.utils.data.Dataset
       - predict() - generates class predictions
       - save() - same as save_model(). Saved model's state to disk
       - summary() - provides a Keras like summary of model
       NOTE:
       - a load() function, to load model's state from disk is not implemented, use stand-alone load_model() 
         function instead
    """
    def __init__(self):
        super(PytModule, self).__init__()
        self.loss_fn = None
        self.optimizer = None
        self.metrics_list = None

    def compile(self, loss, optimizer, metrics=None):
        assert loss is not None, "ERROR: loss function must be a valid loss function!"
        assert optimizer is not None, "ERROR: optimizer must be a valid optimizer function"
        self.loss_fn = loss
        self.optimizer = optimizer
        self.metrics_list = metrics

    def forward(self, input):
        raise NotImplementedError("You have landed up calling PytModule.forward(). " +
            "You must re-implement this menthod in your derived class!")

    def fit_dataset(self, train_dataset, loss_fn=None, optimizer=None, validation_split=0.0,
                    validation_dataset=None, lr_scheduler=None, epochs=25, batch_size=64, metrics=None,
                    shuffle=True, num_workers=0, early_stopping=None):
        """ 
        train model on instance of torch.utils.data.Dataset
        @params:
            - train_dataset: instance of torch.utils.data.Dataset on which the model trains
            - loss_fn (optional, default=None): instance of one of the loss functions defined in Pytorch
              You could pass loss functions as a parameter to this function or pre-set it using the compile function.
              Value passed into this parameter takes precedence over value set in compile(...) call
            - optimizer (optional, default=None): instance of any optimizer defined by Pytorch
              You could pass optimizer as a parameter to this function or pre-set it using the compile function.
              Value passed into this parameter takes precedence over value set in compile(...) call
            - validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. 
              The model will set apart this fraction of the training data, will not train on it, and will 
              evaluate the loss and any model metrics on this data at the end of each epoch.
            - validation_dataset (optional, default=None) - instance of torch.utils.data.Dataset used for cross-validation
              If you pass a valid instance, then model is cross-trained on train_dataset and validation_dataset, else model
              is trained on just train_dataset. As a best practice, it is advisible to use cross training.
            - lr_scheduler (optional, default=NOne) - learning rate scheduler, used to step the learning rate across epochs 
               as model trains. Instance of any scheduler defined in torch.optim.lr_scheduler package
            - epochs (optional, default=25): no of epochs for which model is trained
            - batch_size (optional, default=64): batch size to use
            - metrics (optional, default=None): list of metrics to measure across epochs as model trains. 
              Following metrics are supported (each identified by a key)
                'acc' or 'accuracy' - accuracy
                'prec' or 'precision' - precision
                'rec' or 'recall' - recall
                'f1' or 'f1_score' - f1_score
                'roc_auc' - roc_auc_score
                'mse' - mean squared error
                'rmse' - root mean squared error
               metrics are provided as a list (e.g. ['acc','f1'])
               Loss is ALWAYS measures, even if you don't provide a list of metrics
               NOTE: if validation_dataset is provided, each metric is also measured for the validaton dataset
            - num_workers: no of worker threads to use to load datasets
            - early_stopping: instance of EarlyStopping class if early stopping is to be used (default: None)
        @returns:
           - history object (which is a map of metrics measured across epochs).
             Each metric list is accessed as hist[metric_name] (e.g. hist['loss'] or hist['acc'])
             If validation_dataset is also provided, it will return corresponding metrics for validation dataset too
             (e.g. hist['val_acc'], hist['val_loss'])
        """ 
        p_loss_fn = self.loss_fn if loss_fn is None else loss_fn
        p_optimizer = self.optimizer if optimizer is None else optimizer
        p_metrics_list = self.metrics_list if metrics is None else metrics
        return train_model(self, train_dataset, loss_fn = p_loss_fn, optimizer = p_optimizer,
                           validation_split = validation_split, validation_dataset = validation_dataset,
                           lr_scheduler = lr_scheduler, epochs = epochs, batch_size = batch_size,
                           metrics = p_metrics_list, shuffle = shuffle, num_workers = num_workers,
                           early_stopping=early_stopping)

    def fit(self, X_train, y_train, loss_fn=None, optimizer=None, validation_split=0.0, validation_data=None,
            lr_scheduler=None, epochs=25, batch_size=64, metrics=None, shuffle=True, num_workers=0, early_stopping=None):

        assert ((X_train is not None) and (isinstance(X_train, np.ndarray))), \
            "Parameter error: X_train is None or is NOT an instance of np.ndarray"
        assert ((y_train is not None) and (isinstance(y_train, np.ndarray))), \
            "Parameter error: y_train is None or is NOT an instance of np.ndarray"
        if (y_train.dtype == np.int) or (y_train.dtype == np.long):
            y_dtype = np.long
        else:
            y_dtype = np.float32

        validation_dataset = None
        if validation_data is not None:
            assert isinstance(validation_data, tuple)
            assert isinstance(validation_data[0], np.ndarray), "Expecting validation_dataset[0] to be a Numpy array"
            assert isinstance(validation_data[1], np.ndarray), "Expecting validation_dataset[1] to be a Numpy array"
            if (validation_data[1].dtype == np.int) or (validation_data[1].dtype == np.long):
                y_val_dtype = np.long
            else:
                y_val_dtype = np.float32
            validation_dataset = XyDataset(validation_data[0], validation_data[1], y_val_dtype)

        train_dataset = XyDataset(X_train, y_train, y_dtype)
        p_loss_fn = self.loss_fn if loss_fn is None else loss_fn
        p_optimizer = self.optimizer if optimizer is None else optimizer
        p_metrics_list = self.metrics_list if metrics is None else metrics
        return self.fit_dataset(train_dataset, loss_fn=p_loss_fn, optimizer=p_optimizer,
                                validation_split=validation_split, validation_dataset=validation_dataset,
                                lr_scheduler=lr_scheduler,
                                epochs=epochs, batch_size=batch_size, metrics=p_metrics_list,
                                shuffle=shuffle, num_workers=num_workers, early_stopping=early_stopping)

    def evaluate_dataset(self, dataset, loss_fn=None, batch_size=64, metrics=None, num_workers=0):
        p_loss_fn = self.loss_fn if loss_fn is None else loss_fn
        p_metrics_list = self.metrics_list if metrics is None else metrics
        return evaluate_model(self, dataset, loss_fn=p_loss_fn, batch_size=batch_size, metrics=p_metrics_list,
                              num_workers=num_workers)

    def evaluate(self, X, y, loss_fn=None, batch_size=64, metrics=None, num_workers=0):
        assert ((X is not None) and (isinstance(X, np.ndarray))), \
            "Parameter error: X is None or is NOT an instance of np.ndarray"
        assert ((y is not None) and (isinstance(y, np.ndarray))), \
            "Parameter error: y is None or is NOT an instance of np.ndarray"

        if (y.dtype == np.int) or (y.dtype == np.long):
            y_dtype = np.long
        else:
            y_dtype = np.float32

        p_dataset = XyDataset(X, y, y_dtype)
        return self.evaluate_dataset(p_dataset, loss_fn=loss_fn, batch_size=batch_size,
                                     metrics=metrics, num_workers=num_workers)

    def predict_dataset(self, dataset, batch_size=32, num_workers=0):
        assert dataset is not None
        assert isinstance(dataset, torch.utils.data.Dataset)
        return predict_dataset(self, dataset, batch_size, num_workers=num_workers)

    def predict(self, data):
        assert data is not None
        assert ((isinstance(data, np.ndarray)) or (isinstance(data, torch.Tensor))), \
            "data must be an instance of Numpy ndarray or torch.tensor"
        return predict(self, data)

    def save(self, model_save_name, model_save_dir='./model_states'):
        save_model(self, model_save_name, model_save_dir)

    # NOTE: load() is not implemented. Use standalone load_model() function instead

    def summary(self, input_shape):
        if torch.cuda.is_available():
            summary(self.cuda(), input_shape)
        else:
            summary(self.cpu(), input_shape)

class PytModuleWrapper():
    """
    Utility class that wraps an instance of nn.Module or nn.Sequential or a pre-trained Pytorch module
    and provides a Keras-like interface to train, evaluate & predict results from model.
    """
    def __init__(self, model):
        super(PytModuleWrapper, self).__init__()
        assert (model is not None) and isinstance(model, nn.Module), \
            "model parameter is None or not of type nn.Module"
        self.model = model
        self.loss_fn = None
        self.optimizer = None
        self.metrics_list = None
        
    def compile(self, loss, optimizer, metrics=None):
        assert loss is not None, "ERROR: loss function must be a valid loss function!"
        assert optimizer is not None, "ERROR: optimizer must be a valid optimizer function"
        self.loss_fn = loss
        self.optimizer = optimizer
        self.metrics_list = metrics

    def forward(self, input):
        return self.model.forward(input)

    def parameters(self, recurse=True):
        return self.model.parameters(recurse)

    def fit_dataset(self, train_dataset, loss_fn=None, optimizer=None, validation_split=0.0,
                    validation_dataset=None, lr_scheduler=None, epochs=25, batch_size=64, metrics=None,
                    shuffle=True, num_workers=0, early_stopping=None):
        p_loss_fn = self.loss_fn if loss_fn is None else loss_fn
        p_optimizer = self.optimizer if optimizer is None else optimizer
        p_metrics_list = self.metrics_list if metrics is None else metrics

        return train_model(self.model, train_dataset, loss_fn=p_loss_fn,
            optimizer=p_optimizer, validation_split=validation_split, validation_dataset=validation_dataset,
            lr_scheduler=lr_scheduler, epochs=epochs, batch_size=batch_size, metrics=p_metrics_list,
            shuffle=shuffle, num_workers=num_workers, early_stopping=early_stopping)

    def fit(self, X_train, y_train, loss_fn=None, optimizer=None, validation_split=0.0, validation_data=None,
            lr_scheduler=None, epochs=25, batch_size=64, metrics=None, shuffle=True, num_workers=0, early_stopping=None):

        assert ((X_train is not None) and (isinstance(X_train, np.ndarray))), \
            "Parameter error: X_train is None or is NOT an instance of np.ndarray"
        assert ((y_train is not None) and (isinstance(y_train, np.ndarray))), \
            "Parameter error: y_train is None or is NOT an instance of np.ndarray"
        if (y_train.dtype == np.int) or (y_train.dtype == np.long):
            y_dtype = np.long
        else:
            y_dtype = np.float32

        validation_dataset = None
        if validation_data is not None:
            assert isinstance(validation_data, tuple)
            assert isinstance(validation_data[0], np.ndarray), "Expecting validation_dataset[0] to be a Numpy array"
            assert isinstance(validation_data[1], np.ndarray), "Expecting validation_dataset[1] to be a Numpy array"
            if (validation_data[1].dtype == np.int) or (validation_data[1].dtype == np.long):
                y_val_dtype = np.long
            else:
                y_val_dtype = np.float32
            validation_dataset = XyDataset(validation_data[0], validation_data[1], y_val_dtype)

        train_dataset = XyDataset(X_train, y_train, y_dtype)
        p_loss_fn = self.loss_fn if loss_fn is None else loss_fn
        p_optimizer = self.optimizer if optimizer is None else optimizer
        p_metrics_list = self.metrics_list if metrics is None else metrics
        
        return self.fit_dataset(train_dataset, loss_fn=p_loss_fn, optimizer=p_optimizer,
                                validation_split=validation_split, validation_dataset=validation_dataset,
                                lr_scheduler=lr_scheduler, epochs=epochs, batch_size=batch_size, metrics=p_metrics_list,
                                shuffle=shuffle, num_workers=num_workers, early_stopping=early_stopping)

    def evaluate_dataset(self, dataset, loss_fn=None, batch_size=64, metrics=None, num_workers=0):
        p_loss_fn = self.loss_fn if loss_fn is None else loss_fn
        p_metrics_list = self.metrics_list if metrics is None else metrics
        return evaluate_model(self.model, dataset, loss_fn=p_loss_fn, batch_size=batch_size,
                              metrics=p_metrics_list, num_workers=num_workers)

    def evaluate(self, X, y, loss_fn=None, batch_size=64, metrics=None, num_workers=0):
        assert ((X is not None) and (isinstance(X, np.ndarray))), \
            "Parameter error: X is None or is NOT an instance of np.ndarray"
        assert ((y is not None) and (isinstance(y, np.ndarray))), \
            "Parameter error: y is None or is NOT an instance of np.ndarray"
        if (y.dtype == np.int) or (y.dtype == np.long):
            y_dtype = np.long
        else:
            y_dtype = np.float32

        p_dataset = XyDataset(X, y, y_dtype)
        return self.evaluate_dataset(p_dataset, loss_fn=loss_fn, batch_size=batch_size,
                                     metrics=metrics, num_workers=num_workers)

    def predict_dataset(self, dataset, batch_size=32, num_workers=0):
        assert dataset is not None
        assert isinstance(dataset, torch.utils.data.Dataset)
        return predict_dataset(self.model, dataset, batch_size, num_workers=num_workers)

    def predict(self, data):
        assert data is not None
        assert ((isinstance(data, np.ndarray)) or (isinstance(data, torch.Tensor))), \
            "data must be an instance of Numpy ndarray or torch.tensor"
        return predict(self.model, data)

    def save(self, model_save_name, model_save_dir='./model_states'):
        save_model(self.model, model_save_name, model_save_dir)

    # NOTE: load() is not implemented    def summary(self, input_shape):

    def summary(self, input_shape):
        if torch.cuda.is_available():
            summary(self.model.cuda(), input_shape)
        else:
            summary(self.model.cpu(), input_shape)


