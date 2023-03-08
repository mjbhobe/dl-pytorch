# -*- coding: utf-8 -*-
""" training.py - core functions to help with cross-training, evaluation & testing of Pytorch models"""
import warnings

import torch.utils.data

warnings.filterwarnings('ignore')

import sys

if sys.version_info < (2,):
    raise Exception(
        "torch_training_toolkit does not support Python 1. Please use a Python 3+ interpreter!"
    )

import os
import pathlib

from .metrics_history import *
from .dataset_utils import *
from .layers import *

# custom data types
from typing import Union, Dict, Tuple
import torchmetrics

# LossFxnType = Callable[[torch.tensor, torch.tensor], torch.tensor]
LRSchedulerType = torch.optim.lr_scheduler._LRScheduler
ReduceLROnPlateauType = torch.optim.lr_scheduler.ReduceLROnPlateau
NumpyArrayTuple = Tuple[np.ndarray, np.ndarray]
MetricsMapType = Dict[str, torchmetrics.Metric]
MetricsValType = Dict[str, float]


def cross_train_module(
    model: nn.Module,
    dataset: Union[NumpyArrayTuple, torch.utils.data.Dataset],
    loss_fxn,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    validation_split: float = 0.0,
    validation_dataset: Union[NumpyArrayTuple, torch.utils.data.Dataset] = None,
    metrics_map: MetricsMapType = None,
    epochs: int = 5,
    batch_size: int = 64,
    l1_reg: float = None,
    reporting_interval: int = 1,
    lr_scheduler: Union[LRSchedulerType, ReduceLROnPlateauType] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    verbose: bool = True
) -> MetricsHistory:
    """
        Cross-trains model (derived from nn.Module) across epochs using specified loss function,
        optimizer, validation dataset (if any), learning rate scheduler, epochs and batch size etc.

        This function is used internally by the Trainer class - see Trainer.fit(...)
    """
    # validate parameters passed into function
    assert (0.0 <= validation_split < 1.0), \
        "cross_train_module: 'validation_split' must be a float between (0.0, 1.0]"
    if loss_fxn is None:
        raise ValueError("cross_train_module: 'loss_fxn' cannot be None")
    if optimizer is None:
        raise ValueError("cross_train_module: 'optimizer' cannot be None")

    reporting_interval = 1 if reporting_interval < 1 else reporting_interval
    reporting_interval = 1 if reporting_interval >= epochs else reporting_interval

    train_dataset, val_dataset = dataset, validation_dataset

    if isinstance(train_dataset, tuple):
        # train dataset was a tuple of np.ndarrays - convert to Dataset
        torch_X_train = torch.from_numpy(train_dataset[0]).type(torch.FloatTensor)
        torch_y_train = torch.from_numpy(train_dataset[1]).type(
            torch.LongTensor
            if train_dataset[1].dtype in [np.int, np.long] else torch.FloatTensor
        )
        train_dataset = torch.utils.data.TensorDataset(torch_X_train, torch_y_train)

    if (val_dataset is not None) and isinstance(val_dataset, tuple):
        # cross-val dataset was a tuple of np.ndarrays - convert to Dataset
        torch_X_val = torch.from_numpy(val_dataset[0]).type(torch.FloatTensor)
        torch_y_val = torch.from_numpy(val_dataset[1]).type(
            torch.LongTensor
            if val_dataset[1].dtype in [np.int, np.long] else torch.FloatTensor
        )
        val_dataset = torch.utils.data.TensorDataset(torch_X_val, torch_y_val)

    # split the dataset if validation_split > 0.0
    if (validation_split > 0.0) and (validation_dataset is None):
        # NOTE: validation_dataset supersedes validation_split!!
        # Use validation_split only if validation_dataset is None
        train_dataset, val_dataset = \
            split_dataset(train_dataset, validation_split)

    if val_dataset is not None:
        print(
            f"Cross training on \'{device}\' with {len(train_dataset)} training and " +
            f"{len(val_dataset)} cross-validation records...", flush = True
        )
    else:
        print(
            f"Training on \'{device}\' with {len(train_dataset)} records...",
            flush = True
        )

    if reporting_interval != 1:
        print(
            f"NOTE: progress will be reported every {reporting_interval} epoch!"
        )

    l1_penalty = None if l1_reg is None else torch.nn.L1Loss()
    if l1_reg is not None:
        print(f"Adding L1 regularization with lambda = {l1_reg}")

    history = None

    try:
        model = model.to(device)
        tot_samples = len(train_dataset)
        len_num_epochs, len_tot_samples = len(str(epochs)), len(str(tot_samples))
        # create metrics history
        history = MetricsHistory(metrics_map, (val_dataset is not None))
        train_batch_size = batch_size if batch_size != -1 else len(train_dataset)

        for epoch in range(epochs):
            model.train()
            # reset metrics
            history.clear_batch_metrics()
            # loop over records in training dataset (use DataLoader)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size = train_batch_size,
                shuffle = shuffle,
                num_workers = num_workers
            )
            num_batches, samples = 0, 0

            for batch_no, (X, y) in enumerate(train_dataloader):
                X = X.to(device)
                y = y.to(device)
                # clear accumulated gradients
                optimizer.zero_grad()
                # make forward pass
                preds = model(X)
                # calculate loss
                loss_tensor = loss_fxn(preds, y)
                # add L1 if mentioned
                if l1_reg is not None:
                    reg_loss = 0
                    for param in model.parameters():
                        reg_loss += l1_penalty(param)
                    loss_tensor += reg_loss * l1_reg
                # compute gradients
                loss_tensor.backward()
                # update weights
                optimizer.step()

                # compute batch metric(s)
                preds = preds.to(device)
                history.calculate_batch_metrics(
                    preds.to("cpu"), y.to("cpu"), loss_tensor.item(),
                    val_metrics = False
                )

                num_batches += 1
                samples += len(X)

                if (reporting_interval == 1) and verbose:
                    # display progress with batch metrics - will display line like this:
                    # Epoch (  3/100): (  45/1024) -> loss: 3.456 - acc: 0.275
                    metricsStr = history.get_metrics_str(
                        batch_metrics = True,
                        include_val_metrics = False
                    )
                    print(
                        "\rEpoch (%*d/%*d): (%*d/%*d) -> %s" %
                        (len_num_epochs, epoch + 1, len_num_epochs, epochs,
                         len_tot_samples, samples, len_tot_samples, tot_samples,
                         metricsStr), end = '', flush = True
                    )
            else:
                # all train batches are over - display average train metrics
                history.calculate_epoch_metrics(val_metrics = False)
                if val_dataset is None:
                    if (epoch == 0) or ((epoch + 1) % reporting_interval == 0) \
                        or ((epoch + 1) == epochs):
                        metricsStr = history.get_metrics_str(
                            batch_metrics = False,
                            include_val_metrics = False
                        )
                        print(
                            "\rEpoch (%*d/%*d): (%*d/%*d) -> %s" %
                            (len_num_epochs, epoch + 1, len_num_epochs, epochs,
                             len_tot_samples, samples, len_tot_samples,
                             tot_samples,
                             metricsStr), flush = True
                        )
                        # training ends here as there is no cross-validation dataset
                else:
                    # we have a validation dataset
                    # same print as above except for trailing ... and end=''
                    if (epoch == 0) or ((epoch + 1) % reporting_interval == 0) \
                        or ((epoch + 1) == epochs):
                        metricsStr = history.get_metrics_str(
                            batch_metrics = False,
                            include_val_metrics = False
                        )
                        print(
                            "\rEpoch (%*d/%*d): (%*d/%*d) -> %s..." %
                            (len_num_epochs, epoch + 1, len_num_epochs, epochs,
                             len_tot_samples, samples, len_tot_samples,
                             tot_samples,
                             metricsStr),
                            end = '', flush = True
                        )

                    val_batch_size = batch_size if batch_size != -1 else len(val_dataset)
                    model.eval()
                    with torch.no_grad():
                        # val_dataloader = None if val_dataset is None else \
                        val_dataloader = torch.utils.data.DataLoader(
                            val_dataset,
                            batch_size = val_batch_size,
                            shuffle = shuffle,
                            num_workers = num_workers
                        )
                        num_val_batches = 0

                        for val_X, val_y in val_dataloader:
                            val_X = val_X.to(device)
                            val_y = val_y.to(device)
                            val_preds = model(val_X)
                            val_batch_loss = loss_fxn(val_preds, val_y).item()
                            history.calculate_batch_metrics(
                                val_preds.to("cpu"), val_y.to("cpu"), val_batch_loss,
                                val_metrics = True
                            )
                            num_val_batches += 1
                        else:
                            # loop over val_dataset completed - compute val average metrics
                            history.calculate_epoch_metrics(val_metrics = True)
                            # display final metrics
                            if (epoch == 0) or ((epoch + 1) % reporting_interval == 0) \
                                or ((epoch + 1) == epochs):
                                metricsStr = history.get_metrics_str(
                                    batch_metrics = False,
                                    include_val_metrics = True
                                )
                                print(
                                    "\rEpoch (%*d/%*d): (%*d/%*d) -> %s" %
                                    (len_num_epochs, epoch + 1, len_num_epochs,
                                     epochs,
                                     len_tot_samples, samples, len_tot_samples,
                                     tot_samples,
                                     metricsStr), flush = True
                                )

            # step the learning rate scheduler at end of epoch
            if (lr_scheduler is not None) and (epoch < epochs - 1):
                # have to go to these hoops as ReduceLROnPlateau requires a metric for step()
                if isinstance(
                    lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    lr_metric = history.metrics_history["loss"]["epoch_vals"][-1] \
                        if val_dataset is not None \
                        else history.metrics_history["val_loss"]["epoch_vals"][-1]
                    lr_scheduler.step(lr_metric)
                else:
                    lr_scheduler.step()
        return history
    finally:
        model = model.to('cpu')


def evaluate_module(
    model: nn.Module,
    dataset: Union[NumpyArrayTuple, torch.utils.data.Dataset],
    loss_fn,
    device: torch.device,
    metrics_map: MetricsMapType = None,
    batch_size: int = 64,
    verbose: bool = True
) -> MetricsValType:
    try:
        model = model.to(device)

        # if dataset is a tuple of np.ndarrays, convert to torch Dataset
        if isinstance(dataset, tuple):
            X = torch.from_numpy(dataset[0]).type(torch.FloatTensor)
            y = torch.from_numpy(dataset[1]).type(
                torch.LongTensor
                if dataset[1].dtype in [np.int, np.long] else torch.FloatTensor
            )
            dataset = torch.utils.data.TensorDataset(X, y)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size = batch_size,
            shuffle = False
        )

        tot_samples, samples, num_batches = len(dataset), 0, 0
        len_tot_samples = len(str(tot_samples))

        # create metrics history
        history = MetricsHistory(metrics_map)

        with torch.no_grad():
            model.eval()
            for X, y in loader:
                X = X.to(device)
                y = y.to(device)

                # forward pass
                preds = model(X)
                # compute batch loss
                batch_loss = loss_fn(preds, y).item()
                history.calculate_batch_metrics(
                    preds.to("cpu"), y.to("cpu"), batch_loss,
                    val_metrics = False
                )
                samples += len(y)
                num_batches += 1
                if verbose:
                    metricsStr = history.get_metrics_str(
                        batch_metrics = True,
                        include_val_metrics = False
                    )
                    print(
                        "\rEvaluating (%*d/%*d) -> %s" %
                        (len_tot_samples, samples, len_tot_samples, tot_samples,
                         metricsStr), end = '', flush = True
                    )
            else:
                # iteration over batch completed
                # calculate average metrics across all batches
                history.calculate_epoch_metrics(val_metrics = False)
                metricsStr = history.get_metrics_str(
                    batch_metrics = False,
                    include_val_metrics = False
                )
                print(
                    "\rEvaluating (%*d/%*d) -> %s" %
                    (len_tot_samples, samples, len_tot_samples, tot_samples,
                     metricsStr), flush = True
                )
        return history.get_metric_vals(history.tracked_metrics())
    finally:
        model = model.to('cpu')


def predict_module(
    model: nn.Module,
    dataset: Union[NumpyArrayTuple, torch.utils.data.Dataset],
    device: torch.device,
    batch_size: int = 64
) -> NumpyArrayTuple:
    try:
        model = model.to(device)

        # if dataset is a tuple of np.ndarrays, convert to torch Dataset
        if isinstance(dataset, tuple):
            X = torch.from_numpy(dataset[0]).type(torch.FloatTensor)
            y = torch.from_numpy(dataset[1]).type(
                torch.LongTensor
                if dataset[1].dtype in [np.int, np.long] else torch.FloatTensor
            )
            dataset = torch.utils.data.TensorDataset(X, y)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size = batch_size,
            shuffle = False
        )
        preds, actuals = [], []

        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            with torch.no_grad():
                model.eval()
                batch_preds = list(model(X).to("cpu").numpy())
                batch_actuals = list(y.to("cpu").numpy())
                preds.extend(batch_preds)
                actuals.extend(batch_actuals)
        return (np.array(preds), np.array(actuals))
    finally:
        model = model.to('cpu')


def predict_array(
    model: nn.Module,
    data: np.ndarray,
    device: torch.device,
    batch_size: int = 64
) -> np.ndarray:
    """
        runs predictions on Numpy Array (use for classification ONLY)
        @params:
            - model: instance of model derived from nn.Module (or instance of pyt.PytModel or pyt.PytSequential)
            - data: Numpy array of values on which predictions should be run
        @returns:
            - Numpy array of class predictions (probabilities)
            NOTE: to convert to classes use np.max(...,axis=1) after this call.
    """
    try:
        assert isinstance(model, nn.Module), \
            "predict() works with instances of nn.Module only!"
        assert ((isinstance(data, np.ndarray)) or (isinstance(data, torch.Tensor))), \
            "data must be an instance of Numpy ndarray or torch.tensor"
        # train on GPU if you can
        model = model.to(device)

        # run prediction
        with torch.no_grad():
            model.eval()
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype = torch.float32)
            data = data.to(device)
            # forward pass
            logits = model(data)
            preds = np.array(logits.cpu().numpy())
        return preds
    finally:
        model = model.cpu()


def save_model(model: nn.Module, model_save_path: str, verbose: bool = True):
    """ saves Pytorch state (state_dict) to disk
        @params:
            - model: instance of model derived from nn.Module (or instance of pytk.PytModel or pytk.PytSequential)
            - model_save_path: absolute or relative path where model's state-dict should be saved
              (NOTE:
                 - the model_save_path file is overwritten at destination without warning
                 - if `model_save_path` is just a file name, then model saved to current dir
                 - if `model_save_path` contains directory that does not exist, the function attempts to create
                   the directories
              )
    """
    save_dir, _ = os.path.split(model_save_path)

    if not os.path.exists(save_dir):
        # create directory from file_path, if it does not exist
        # e.g. if model_save_path = '/home/user_name/pytorch/model_states/model.pyt' and the
        # directory '/home/user_name/pytorch/model_states' does not exist, it is created
        try:
            os.mkdir(save_dir)
        except OSError as err:
            print(
                f"Unable to create folder/directory {save_dir} to save model!"
            )
            raise err

    # now save the model to file_path
    # NOTE: a good practice is to move the model to cpu before saving the state dict
    # as this will save tensors as CPU tensors, rather than CUDA tensors, which will
    # help load model on any machine that may or may not have GPUs
    torch.save(model.to("cpu").state_dict(), model_save_path)
    if verbose:
        print(f"Pytorch model saved to {model_save_path}")


def load_model(model: nn.Module, model_state_dict_path: str, verbose: bool = True) -> nn.Module:
    """ loads model's state dict from file on disk
        @params:
            - model: instance of model derived from nn.Module
            - model_state_dict_path: complete/relative path from where model's state dict
              should be loaded. This should be a valid path (i.e. should exist),
              else an IOError is raised.
        @returns:
            - an nn.Module with state dict (weights & structure) loaded from disk
    """

    # convert model_state_dict_path to absolute path
    model_save_path = pathlib.Path(model_state_dict_path).absolute()
    if not os.path.exists(model_save_path):
        raise IOError(
            f"ERROR: can't load model from {model_state_dict_path} - file does not exist!"
        )

    # load state dict from path
    state_dict = torch.load(model_save_path)
    model.load_state_dict(state_dict)
    if verbose:
        print(f"Pytorch model loaded from {model_state_dict_path}")
    model.eval()
    return model


class Trainer:
    def __init__(
        self,
        loss_fn,
        device: torch.device,
        metrics_map: MetricsMapType = None,
        epochs: int = 5, batch_size: int = 64, reporting_interval: int = 1,
        shuffle: bool = True,
        num_workers: int = 0
    ):
        """ constructs a Trainer object to be used for cross-training, evaluation of and
            getting predictions from an nn.Module instance.
            @params:
             - loss_fn: loss function to be used during training & evaluation (pass an instance
                 of one of PyTorch's loss functions (e.g. nn.BCELoss(), nn.CrossEntropyLoss())
             - device (type torch.device): the device on which training/evaluation etc.
                 should happen (e.g., torch.device("cuda") for GPU or torch.device("cpu"))
             - metrics_map (optional, default None): declares the metrics ** IN ADDITION TO ** loss
                 that should be tracked across epochs as the model trains/evaluates. It is optional,
                 which means that only loss will be tracked, else loss + all metrics defined in the map
                 will be tracked across epochs.
                 A metrics_map is of type `Dict[str, torchmetric.Metric]` & you can define several
                 metrics to be monitored during the training process. Each metric has a user-defined
                 abbreviation (the `str` part of the Dict) and an associated torchmetric.Metric _formula_
                 to calculate the metric
                 Example of metrics_maps:
                    # measures accuracy (abbreviated 'acc' [user can choose any abbreviation here!])
                    metric_map = {
                        "acc": torchmetrics.classification.BinaryAccuracy()
                    }
                    # this one trackes 3 metrics - accuracy (acc),
                    metrics_map = {
                        "acc": torchmetrics.classification.BinaryAccuracy(),
                        "f1": torchmetrics.classification.BinaryF1Score(),
                        "roc_auc": torchmetrics.classification.BinaryAUROC(thresholds = None)
                    }
             - epochs (int, default 5) - number of epochs for which the module should train
             - batch_size (int, default 64) - size of batch used during training
             - reporting_interval (int, default 1) - interval (i.e. number of epochs) after
                 which progress must be reported when training. By default, the class
                 reports progress after each epoch. Set this value to change the interval
                 for example, if reporting_interval = 5, progress will be reported on the
                 first, then every fifth and the last epoch.
             - shuffle (bool, default=True) - should the data be shuffled during training?
                 If True, it is shuffled else not. It is a good practice to always shuffle
                 data between epochs. This setting should be left to the default value,
                 unless you have a valid reason NOT to shuffle data (training and cross-validation)
        """
        if loss_fn is None:
            raise ValueError("FATAL ERROR: Trainer() -> 'loss_fn' cannot be None")
        if device is None:
            raise ValueError("FATAL ERROR: Trainer() -> 'device' cannot be None")
        if epochs < 1:
            raise ValueError("FATAL ERROR: Trainer() -> 'epochs' >= 1")
        # batch_size can be -ve
        batch_size = -1 if batch_size < 0 else batch_size
        reporting_interval = 1 if reporting_interval < 1 else reporting_interval
        assert num_workers >= 0, \
            "FATAL ERROR: Trainer() -> 'num_workers' must be >= 0"

        self.loss_fn = loss_fn
        self.device = device
        self.metrics_map = metrics_map
        self.epochs = epochs
        self.batch_size = batch_size
        self.reporting_interval = reporting_interval
        self.shuffle = shuffle
        self.num_workers = num_workers

    def fit(
        self, model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataset: Union[NumpyArrayTuple, torch.utils.data.Dataset],
        validation_dataset: Union[NumpyArrayTuple, torch.utils.data.Dataset] = None,
        validation_split: float = 0.0,
        l1_reg = None,
        lr_scheduler: Union[LRSchedulerType, ReduceLROnPlateauType] = None,
        verbose: bool = True
    ) -> MetricsHistory:
        """ fits (i.e cross-trains) the model using provided training data (and optionally
            cross-validation data).
            @params:
              - model (type nn.Module, required) - an instance of the model (derived from nn.Module) to
                  be trained.
              - optimizer (type torch.optim.Optimizer, required) - the optimizer used to adjust the weights
                  during training. Use an instance of any optimizer from torch.optim package
              - train_dataset (a Numpy `ndarray tuple` OR `torch.utils.data.Dataset`, required) - the data
                  or dataset to be used for training. It can be a tuple of Numpy arrays (e.g., (X_train, y_train)
                  or an instance of torch.utils.data.Dataset class.
                  It is easier to use the tuple of Numpy arrays paradigm when you are loading structured tabular
                  data - most common scikit datasets would fall into this category (e.g. Iris, Wisconsin etc.)
                  For more complex data types, such as images, audio or text, you would use the Dataset alternative.
               - validation_dataset (optional, default = None) [type a Numpy `ndarray tuple` OR `torch.utils.data.Dataset`
                  class]. Optionally defined the validation data/dataset to be used. Similar to train_dataset
               - validation_split (float, default=0.2) - used when you have just training data, but want to internally
                  split into train & cross-validation datasets. Specify a percentage (between 0.0 and 1.0) of train
                  dataset that would be set aside for cross-validation.
                  For example, if validation_split = 0.2, then 20% of train dataset will be set aside for cross-validation.
                  NOTE: if both validation_dataset and validation_split are specified, the former is used and latter IGNORED.
        """
        assert model is not None, \
            "FATAL ERROR: Trainer.fit() -> 'model' cannot be None"
        assert optimizer is not None, \
            "FATAL ERROR: Trainer.fit() -> 'optimizer' cannot be None"
        assert train_dataset is not None, \
            "FATAL ERROR: Trainer.fit() -> 'train_dataset' cannot be None"
        if lr_scheduler is not None:
            # NOTE:  ReduceLROnPlateau is NOT derived from _LRScheduler, but from object, which
            # is odd as all other schedulers derive from _LRScheduler
            assert (isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler) or
                    isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)), \
                "lr_scheduler: incorrect type. Expecting class derived from " \
                "torch.optim._LRScheduler or ReduceLROnPlateau"

        history = cross_train_module(
            model, train_dataset, self.loss_fn, optimizer, device = self.device,
            validation_split = validation_split, validation_dataset = validation_dataset,
            metrics_map = self.metrics_map, epochs = self.epochs, batch_size = self.batch_size,
            l1_reg = l1_reg,
            reporting_interval = self.reporting_interval, lr_scheduler = lr_scheduler,
            shuffle = self.shuffle, num_workers = self.num_workers, verbose = verbose
        )
        return history

    def evaluate(
        self,
        model: nn.Module,
        dataset: Union[NumpyArrayTuple, torch.utils.data.Dataset],
        verbose: bool = True
    ) -> dict:
        return evaluate_module(
            model, dataset, self.loss_fn, device = self.device, metrics_map = self.metrics_map,
            batch_size = self.batch_size, verbose = verbose
        )

    def predict(
        self,
        model: nn.Module,
        dataset: Union[np.ndarray, NumpyArrayTuple, torch.utils.data.Dataset]
    ) -> Union[np.ndarray, NumpyArrayTuple]:
        if isinstance(dataset, np.ndarray):
            return predict_array(model, dataset, self.device, self.batch_size)
        else:
            return predict_module(model, dataset, self.device, self.batch_size)
