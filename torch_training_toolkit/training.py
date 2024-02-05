# -*- coding: utf-8 -*-
""" training.py - core functions to help with cross-training, evaluation & testing of Pytorch models"""
import warnings

warnings.filterwarnings("ignore")

import sys
import os

if sys.version_info < (2,):
    raise Exception(
        "torch_training_toolkit does not support Python 2.x. Please use a Python 3+ interpreter!"
    )


from .metrics_history import *
from .dataset_utils import *
from .layers import *
from .early_stopping import *

# custom data types
from typing import Union, Dict, Tuple
import torch.utils.data
import torchmetrics
import logging

# LossFxnType = Callable[[torch.tensor, torch.tensor], torch.tensor]
LRSchedulerType = torch.optim.lr_scheduler._LRScheduler
ReduceLROnPlateauType = torch.optim.lr_scheduler.ReduceLROnPlateau
NumpyArrayTuple = Tuple[np.ndarray, np.ndarray]
MetricsMapType = Dict[str, torchmetrics.Metric]
MetricsValType = Dict[str, float]


def cross_train_module(
    model: nn.Module,
    dataset: Union[
        NumpyArrayTuple, torch.utils.data.Dataset, torch.utils.data.DataLoader
    ],
    loss_fxn,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    validation_split: float = 0.0,
    validation_dataset: Union[
        NumpyArrayTuple, torch.utils.data.Dataset, torch.utils.data.DataLoader
    ] = None,
    metrics_map: MetricsMapType = None,
    epochs: int = 5,
    batch_size: int = 64,
    l1_reg: float = None,
    reporting_interval: int = 1,
    lr_scheduler: Union[LRSchedulerType, ReduceLROnPlateauType] = None,
    early_stopping: EarlyStopping = None,
    shuffle: bool = True,
    num_workers: int = 4,
    verbose: bool = True,
    logger: logging.Logger = None,
    seed=41,
) -> MetricsHistory:
    """
    Cross-trains model (derived from nn.Module) across epochs using specified loss function,
    optimizer, validation dataset (if any), learning rate scheduler, epochs and batch size etc.

    This function is used internally by the Trainer class - see Trainer.fit(...)
    """
    torch.manual_seed(seed)

    if logger is not None:
        logger.debug(f"Entering cross_train_module() function")
        # display info on training parameters
        train_params = "Training Parameters:\n"
        train_params += f"   - will train on {device} for {epochs} epochs and batch_size of {batch_size}."
        train_params += "\n" if early_stopping is None else " Early stopping applied!\n"
        train_params += (
            f"   - progress will be reported every {reporting_interval} epochs.\n"
        )
        train_params += (
            f"   - train dataset type: {type(dataset)} with {len(dataset)} records.\n"
        )
        if validation_split > 0.0:
            train_params += f"   - using validation_split = {validation_split}."
            train_params += (
                "\n"
                if validation_dataset is None
                else " NOTE: validation_dataset will be ignored!\n"
            )
        elif validation_dataset is not None:
            train_params += f"   - validation dataset type: {type(validation_dataset)} with {len(validation_dataset)} records.\n"
        else:
            train_params += f"   - NOTE: validation_dataset NOT provided!\n"
        train_params += (
            f"   - using {type(loss_fxn)} loss and {type(optimizer)} optimizer.\n"
        )
        train_params += f"   - Additional metrics tracked: {str(metrics_map)}.\n"
        if l1_reg is not None:
            train_params += f"   - using L1 regularization: {l1_reg}.\n"
        if lr_scheduler is not None:
            train_params += (
                f"   - using learning rate scheduler: {type(lr_scheduler)}.\n"
            )
        logger.debug(train_params)

    # validate parameters passed into function
    if not (0.0 <= validation_split < 1.0):
        if logger is not None:
            logger.fatal(
                f"Exception -> ValueError(\"cross_train_module: 'validation_split' must be a float between (0.0, 1.0]\")"
            )
        raise ValueError(
            "cross_train_module: 'validation_split' must be a float between (0.0, 1.0]"
        )

    if loss_fxn is None:
        if logger is not None:
            logger.fatal(
                "Exception -> ValueError(\"cross_train_module: 'loss_fxn' cannot be None\")"
            )
        raise ValueError("cross_train_module: 'loss_fxn' cannot be None")
    if optimizer is None:
        if logger is not None:
            logger.fatal(
                "Exception -> ValueError(\"cross_train_module: 'optimizer' cannot be None\")"
            )
        raise ValueError("cross_train_module: 'optimizer' cannot be None")

    reporting_interval = 1 if reporting_interval < 1 else reporting_interval
    reporting_interval = 1 if reporting_interval >= epochs else reporting_interval

    train_dataset, val_dataset = dataset, validation_dataset

    if isinstance(train_dataset, tuple):
        # check that we have a tuple of numpy ndarrays
        if not (isinstance(train_dataset[0], np.ndarray)) and (
            isinstance(train_dataset[1], np.ndarray)
        ):
            if logger is not None:
                logger.fatal(
                    f'Exception -> ValueError("dataset passed in as tuple must have np.ndarray elements only!")'
                )
            raise ValueError(
                f"FATAL: dataset passed in as tuple must have np.ndarray elements only!"
            )
        # train dataset was a tuple of np.ndarrays - convert to Dataset
        torch_X_train = torch.from_numpy(train_dataset[0]).type(torch.FloatTensor)
        torch_y_train = torch.from_numpy(train_dataset[1]).type(
            torch.LongTensor
            if train_dataset[1].dtype in [int, np.int32, np.int64]  # , np.long]
            else torch.FloatTensor
        )
        train_dataset = torch.utils.data.TensorDataset(torch_X_train, torch_y_train)
        if logger is not None:
            logger.debug(
                "Detected training dataset as instance of tuples of np.ndarrays. Converted to TensorDatasets"
            )

    if (val_dataset is not None) and isinstance(val_dataset, tuple):
        # check that we have a tuple of numpy ndarrays
        if not (isinstance(val_dataset[0], np.ndarray)) and (
            isinstance(val_dataset[1], np.ndarray)
        ):
            if logger is not None:
                logger.fatal(
                    f'Exception -> ValueError("validation dataset passed in as tuple must have np.ndarray elements only!")'
                )
            raise ValueError(
                f"FATAL: validation dataset passed in as tuple must have np.ndarray elements only!"
            )
        # cross-val dataset was a tuple of np.ndarrays - convert to Dataset
        torch_X_val = torch.from_numpy(val_dataset[0]).type(torch.FloatTensor)
        torch_y_val = torch.from_numpy(val_dataset[1]).type(
            torch.LongTensor
            if val_dataset[1].dtype in [int, np.int32, np.int64]  # , np.long]
            else torch.FloatTensor
        )
        val_dataset = torch.utils.data.TensorDataset(torch_X_val, torch_y_val)
        if logger is not None:
            logger.debug(
                "Detected validation dataset as instance of tuples of np.ndarrays. Converted to TensorDatasets"
            )

    # split the dataset if validation_split > 0.0
    # NOTE: train_dataset must be an instance of torch.util.data.Dataset, not DataLoader
    if (
        (validation_split > 0.0)
        # and isinstance(validation_dataset, torch.utils.data.Dataset)
        and (validation_dataset is None)
    ):
        # NOTE: validation_dataset supersedes validation_split!!
        # Use validation_split only if validation_dataset is None
        # if both are specified (i.e. validation_split > 0.0 and validation_dataset is not None)
        # then validation_split value will be ignored!
        if not isinstance(train_dataset, torch.utils.data.Dataset):
            raise ValueError(
                f"FATAL ERROR: for validation_split > 0, expecting training data to be a Dataset not DataLoader!"
            )
        train_dataset, val_dataset = split_dataset(train_dataset, validation_split)
        if logger is not None:
            logger.debug(
                f"NOTE: using validation_split (= {validation_split}) to split training dataset. Will ignore validation_dataset parameter!"
            )

    if val_dataset is not None:
        print(
            f"Cross training on '{device}' with {len(train_dataset):,} training and "
            + f"{len(val_dataset):,} cross-validation records...",
            flush=True,
        )
    else:
        print(
            f"Training on '{device}' with {len(train_dataset):,} records...", flush=True
        )

    if reporting_interval != 1:
        print(f"NOTE: progress will be reported every {reporting_interval} epoch!")

    l1_penalty = None if l1_reg is None else torch.nn.L1Loss()
    if (l1_reg is not None) and (logger is not None):
        logger.debug(f"Adding L1 regularization with lambda = {l1_reg}")

    history = None

    try:
        model = model.to(device)
        tot_samples = len(train_dataset)
        len_num_epochs, len_tot_samples = len(str(epochs)), len(str(tot_samples))
        # create metrics history
        history = MetricsHistory(metrics_map, (val_dataset is not None))
        train_batch_size = batch_size if batch_size != -1 else len(train_dataset)

        # NOTE: for some reason creating DataLoaders on Windows with num_workers > 0
        # takes an insane amount of time, slowing down training significantly.
        # On Windows, will ignore the num_workers parameter (i.e. force pass in
        # a zero, which is the default value)
        # @see: https://stackoverflow.com/questions/73777647/pytorch-custom-dataset-is-super-slow
        num_workers_hack = 0 if os.name == "nt" else num_workers
        pin_mem_hack = True if torch.cuda.is_available() else False
        # print(
        #     f"On {os.name} using {num_workers_hack} workers & pin_memory is {pin_mem_hack}",
        #     flush=True,
        # )
        # sys.exit(-1)

        # create the train & validation dataloaders
        train_dataloader = (
            # convert dataset to dataloader
            torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=shuffle,
                num_workers=num_workers_hack,
                pin_memory=pin_mem_hack,
            )
            if isinstance(train_dataset, torch.utils.data.Dataset)
            # or use dataloader as-is
            else train_dataset
        )

        # create validation dataloader only if val_dataset is present!
        if val_dataset is not None:
            val_batch_size = batch_size if batch_size != -1 else len(val_dataset)
            val_dataloader = (
                torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=val_batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers_hack,
                    pin_memory=pin_mem_hack,
                )
                if isinstance(val_dataset, torch.utils.data.Dataset)
                else val_dataset
            )

        for epoch in range(epochs):
            model.train()
            # reset metrics
            history.clear_batch_metrics()
            # loop over records in training dataset (use DataLoader)
            # train_dataloader = (
            #     # convert dataset to dataloader
            #     torch.utils.data.DataLoader(
            #         train_dataset,
            #         batch_size=train_batch_size,
            #         shuffle=shuffle,
            #         num_workers=num_workers,
            #     )
            #     if isinstance(train_dataset, torch.utils.data.Dataset)
            #     # or use dataloader as-is
            #     else train_dataset
            # )
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
                # add L1 loss, if specified - L2 Regularization is handled by optimizer!
                if l1_reg is not None:
                    # reg_loss = 0.0
                    reg_loss = sum([l1_penalty(param) for param in model.parameters()])
                    # for param in model.parameters():
                    #     reg_loss += l1_penalty(param)
                    loss_tensor += reg_loss * l1_reg
                # compute gradients
                loss_tensor.backward()
                # update weights
                optimizer.step()

                # compute batch metric(s)
                preds = preds.to(device)
                history.calculate_batch_metrics(
                    preds.to("cpu"), y.to("cpu"), loss_tensor.item(), val_metrics=False
                )

                num_batches += 1
                samples += len(X)

                if (reporting_interval == 1) and verbose:
                    # display progress with batch metrics - will display line like this:
                    # Epoch (  3/100): (  45/1024) -> loss: 3.456 [- acc: 0.275...]
                    metricsStr = history.get_metrics_str(
                        batch_metrics=True, include_val_metrics=False
                    )
                    print(
                        "\rEpoch (%*d/%*d): (%*d/%*d) -> %s"
                        % (
                            len_num_epochs,
                            epoch + 1,
                            len_num_epochs,
                            epochs,
                            len_tot_samples,
                            samples,
                            len_tot_samples,
                            tot_samples,
                            metricsStr,
                        ),
                        end="",
                        flush=True,
                    )
            else:
                # all train batches are over - display average train metrics
                history.calculate_epoch_metrics(val_metrics=False)
                if val_dataset is None:
                    if (
                        (epoch == 0)
                        or ((epoch + 1) % reporting_interval == 0)
                        or ((epoch + 1) == epochs)
                    ):
                        metricsStr = history.get_metrics_str(
                            batch_metrics=False, include_val_metrics=False
                        )
                        prog_str = "\rEpoch (%*d/%*d): (%*d/%*d) -> %s" % (
                            len_num_epochs,
                            epoch + 1,
                            len_num_epochs,
                            epochs,
                            len_tot_samples,
                            samples,
                            len_tot_samples,
                            tot_samples,
                            metricsStr,
                        )
                        print(prog_str, flush=True)
                        if logger is not None:
                            logger.debug(prog_str)
                        # training ends here as there is no cross-validation dataset
                else:
                    # we have a validation dataset
                    # same print as above except for trailing ... and end=''
                    if (
                        (epoch == 0)
                        or ((epoch + 1) % reporting_interval == 0)
                        or ((epoch + 1) == epochs)
                    ):
                        metricsStr = history.get_metrics_str(
                            batch_metrics=False, include_val_metrics=False
                        )
                        print(
                            "\rEpoch (%*d/%*d): (%*d/%*d) -> %s..."
                            % (
                                len_num_epochs,
                                epoch + 1,
                                len_num_epochs,
                                epochs,
                                len_tot_samples,
                                samples,
                                len_tot_samples,
                                tot_samples,
                                metricsStr,
                            ),
                            end="",
                            flush=True,
                        )

                    # val_batch_size = (
                    #     batch_size if batch_size != -1 else len(val_dataset)
                    # )
                    model.eval()
                    with torch.no_grad():
                        # val_dataloader = None if val_dataset is None else \
                        # val_dataloader = (
                        #     torch.utils.data.DataLoader(
                        #         val_dataset,
                        #         batch_size=val_batch_size,
                        #         shuffle=shuffle,
                        #         num_workers=num_workers,
                        #     )
                        #     if isinstance(val_dataset, torch.utils.data.Dataset)
                        #     else val_dataset
                        # )
                        num_val_batches = 0

                        for val_X, val_y in val_dataloader:
                            val_X = val_X.to(device)
                            val_y = val_y.to(device)
                            val_preds = model(val_X)
                            val_batch_loss = loss_fxn(val_preds, val_y).item()
                            history.calculate_batch_metrics(
                                val_preds.to("cpu"),
                                val_y.to("cpu"),
                                val_batch_loss,
                                val_metrics=True,
                            )
                            num_val_batches += 1
                        else:
                            # loop over val_dataset completed - compute val average metrics
                            history.calculate_epoch_metrics(val_metrics=True)
                            # display final metrics
                            if (
                                (epoch == 0)
                                or ((epoch + 1) % reporting_interval == 0)
                                or ((epoch + 1) == epochs)
                            ):
                                metricsStr = history.get_metrics_str(
                                    batch_metrics=False, include_val_metrics=True
                                )
                                prog_str = "\rEpoch (%*d/%*d): (%*d/%*d) -> %s" % (
                                    len_num_epochs,
                                    epoch + 1,
                                    len_num_epochs,
                                    epochs,
                                    len_tot_samples,
                                    samples,
                                    len_tot_samples,
                                    tot_samples,
                                    metricsStr,
                                )
                                print(prog_str, flush=True)
                                if logger is not None:
                                    logger.debug(prog_str)

            if (early_stopping is not None) and (val_dataset is not None):
                # early stooping test only if validation dataset is used
                monitored_metric = early_stopping.monitored_metric()
                if monitored_metric not in history.metrics_history.keys():
                    raise ValueError(
                        f"FATAL ERROR: EarlyStopping tracks {monitored_metric}, which is NOT tracked during training!\n"
                        f"Either add this metric to metrics_map or check metric tracked by EarlyStopping"
                    )
                last_metric_val = history.metrics_history[monitored_metric][
                    "epoch_vals"
                ][-1]
                early_stopping(model, last_metric_val, epoch)
                if early_stopping.early_stop:
                    # load last state
                    model.load_state_dict(torch.load(early_stopping.checkpoint_path()))
                    model.eval()
                    if logger is not None:
                        logger.debug("Early stopping the training loop.")
                    break

            # step the learning rate scheduler at end of epoch
            if (lr_scheduler is not None) and (epoch < epochs - 1):
                # have to go to these hoops as ReduceLROnPlateau requires a metric for step()
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_metric = (
                        history.metrics_history["loss"]["epoch_vals"][-1]
                        if val_dataset is not None
                        else history.metrics_history["val_loss"]["epoch_vals"][-1]
                    )
                    lr_scheduler.step(lr_metric)
                else:
                    lr_scheduler.step()

        return history
    finally:
        model = model.to("cpu")


def evaluate_module(
    model: nn.Module,
    dataset: Union[
        NumpyArrayTuple, torch.utils.data.Dataset, torch.utils.data.DataLoader
    ],
    loss_fn,
    device: torch.device,
    metrics_map: MetricsMapType = None,
    batch_size: int = 64,
    verbose: bool = True,
    logger: logging.Logger = None,
    seed=41,
) -> MetricsValType:
    """evaluate module performance.
    Internal function used by Trainer.evaluate() - please see Trainer class for details
    """
    torch.manual_seed(seed)
    try:
        model = model.to(device)

        # if dataset is a tuple of np.ndarrays, convert to torch Dataset
        if isinstance(dataset, tuple):
            X = torch.from_numpy(dataset[0]).type(torch.FloatTensor)
            y = torch.from_numpy(dataset[1]).type(
                torch.LongTensor
                if dataset[1].dtype in [int, np.int32, np.int64]  # , np.long]
                else torch.FloatTensor
            )
            dataset = torch.utils.data.TensorDataset(X, y)

        loader = (
            torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            if isinstance(dataset, torch.utils.data.Dataset)
            else dataset
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
                    preds.to("cpu"), y.to("cpu"), batch_loss, val_metrics=False
                )
                samples += len(y)
                num_batches += 1
                if verbose:
                    metricsStr = history.get_metrics_str(
                        batch_metrics=True, include_val_metrics=False
                    )
                    print(
                        "\rEvaluating (%*d/%*d) -> %s"
                        % (
                            len_tot_samples,
                            samples,
                            len_tot_samples,
                            tot_samples,
                            metricsStr,
                        ),
                        end="",
                        flush=True,
                    )
            else:
                # iteration over batch completed
                # calculate average metrics across all batches
                history.calculate_epoch_metrics(val_metrics=False)
                metricsStr = history.get_metrics_str(
                    batch_metrics=False, include_val_metrics=False
                )
                prog_str = (
                    "\rEvaluating (%*d/%*d) -> %s"
                    % (
                        len_tot_samples,
                        samples,
                        len_tot_samples,
                        tot_samples,
                        metricsStr,
                    ),
                )
                print(prog_str, flush=True)
        return history.get_metric_vals(history.tracked_metrics())
    finally:
        model = model.to("cpu")


def predict_module(
    model: nn.Module,
    dataset: Union[
        NumpyArrayTuple, torch.utils.data.Dataset, torch.utils.data.DataLoader
    ],
    device: torch.device,
    batch_size: int = 64,
    # for_regression: bool = False,
    logger: logging.Logger = None,
    seed=41,
) -> NumpyArrayTuple:
    """make predictions from array or dataset
    Internal function used by Trainer.predict() - please see Trainer class for details
    """
    torch.manual_seed(seed)
    try:
        model = model.to(device)

        # if dataset is a tuple of np.ndarrays, convert to torch Dataset
        if isinstance(dataset, tuple):
            X = torch.from_numpy(dataset[0]).type(torch.FloatTensor)
            y = torch.from_numpy(dataset[1]).type(
                torch.LongTensor
                if dataset[1].dtype in [int, np.int32, np.int64]  # , np.long]
                else torch.FloatTensor
            )
            dataset = torch.utils.data.TensorDataset(X, y)

        loader = (
            torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            if isinstance(dataset, torch.utils.data.Dataset)
            else dataset
        )
        # preds, actuals = [], []
        preds, actuals = [], []  # np.array([]), np.array([])

        model.eval()
        with torch.no_grad():
            for X, y in loader:
                X = X.to(device)
                y = y.to(device)
                # batch_preds = list(model(X).ravel().numpy())

                # NOTE: the np.argmax(...) for classification
                # should be done by caller
                batch_preds = list(model(X).cpu().numpy())
                batch_actuals = list(y.cpu())

                # if for_regression:
                #     batch_preds = list(model(X).numpy().ravel())
                # else:
                #     batch_preds = list(np.argmax(model(X).numpy(), axis=1))
                # batch_actuals = list(y.ravel().numpy())

                # batch_preds = list(model(X).to("cpu").numpy())
                # batch_actuals = list(y.to("cpu").numpy())
                # batch_preds = model(X).to("cpu").numpy()
                # batch_actuals = y.to("cpu").numpy()

                preds.extend(batch_preds)
                actuals.extend(batch_actuals)
                # preds = np.append(preds, batch_preds)
                # actuals = np.append(actuals, batch_actuals)
        if logger is not None:
            logger.debug("Preditions: ")
            logger.debug(f"  - Actuals     : {np.array(actuals)}")
            logger.debug(f"  - Predictions : {np.array(preds)}")
        return (np.array(preds), np.array(actuals))
        # return (preds, actuals)
    finally:
        model = model.to("cpu")


def predict_array(
    model: nn.Module,
    data: Union[np.ndarray, torch.Tensor],
    device: torch.device,
    batch_size: int = 64,
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
        assert isinstance(
            model, nn.Module
        ), "predict() works with instances of nn.Module only!"
        assert (isinstance(data, np.ndarray)) or (
            isinstance(data, torch.Tensor)
        ), "data must be an instance of Numpy ndarray or torch.tensor"
        # train on GPU if you can
        model = model.to(device)

        # run prediction
        with torch.no_grad():
            model.eval()
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float32)
            data = data.to(device)
            # forward pass
            logits = model(data)
            preds = np.array(logits.cpu().numpy())
        return preds
    finally:
        model = model.cpu()


def save_model(
    model: nn.Module,
    model_save_path: str,
    verbose: bool = True,
):
    """saves the state of the module (state_dict) to disk in model_save_path location

    Parameters
    ----------
    model: nn.Module
        instance of nn.Module whose state needs to be saved
    model_save_path: str
        absolute or relative path where model's state-dict should be saved
        **NOTE:** the model_save_path file is overwritten at destination without warning
            - if `model_save_path` is just a file name, then model saved to current dir
            - if `model_save_path` is a valid path, model's state_dict is saved to that path
              Alternatively, if the path does not exist, this function trys to create all intermedate
              directories to the full path before saving the model state_dict.
    verbose: bool (optional, default=True)
        displays a message showing path of file where state_dict is saved (True) or does not (False)

    Exceptions:
    OSError: is raised when function is unable to save state for any reason.
    """
    save_dir, _ = os.path.split(model_save_path)
    # create all folders leading upto path where applicable
    os.makedirs(save_dir, exist_ok=True)

    # if not os.path.exists(save_dir):
    #     # create directory from file_path, if it does not exist
    #     # e.g. if model_save_path = '/home/user_name/pytorch/model_states/model.pyt' and the
    #     # directory '/home/user_name/pytorch/model_states' does not exist, we will attempt to create it
    #     try:
    #         # os.mkdirs(save_dir)
    #         os.makedirs(save_dir)
    #     except OSError as err:
    #         print(f"Unable to create folder/directory {save_dir} to save model!")
    #         raise err

    # now save the model to file_path
    # NOTE: a good practice is to move the model to cpu before saving the state dict
    # as this will save tensors as CPU tensors, rather than CUDA tensors, which will
    # help load model on any machine that may or may not have GPUs
    torch.save(model.to("cpu").state_dict(), model_save_path)
    if verbose:
        print(f"Pytorch model saved to {model_save_path}")


def load_model(
    model: nn.Module,
    model_state_dict_path: str,
    verbose: bool = True,
) -> nn.Module:
    """loads model's state dict from file on disk

    Parameters
    ----------
    model: nn.Module
        instance of model derived from nn.Module
    model_state_dict_path: str
        the complete/relative path from where model's state dict should be loaded.
        This should be a valid path (i.e. should exist), else an IOError is raised.

    Returns
    -------
        an nn.Module with state dict (weights & structure) loaded from disk

    Example of use:
        model_save_path = "/valid/path/to/model.pth"
        model = MyModel(...)    # an instance of your model
        model = load_model(model, model_save_path)

        # use model
        model.predict(....)
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
    """
    class that manages the training, evaluation of Pytorch models & generate predictions

    Attributes
    ----------
    loss_fn : torch.nn.modules.loss._Loss
        the loss function to use during training (e.g. MSELoss(), CrossEntropyLoss())
    device  : torch.device
        the device to train the model on (e.g. torch.device("cuda"))
    metrics_map : map {str:torchmetrics.Metric} (optiona, default=None)
        a map of str mapped to a torchmetrics.Metric. These metrics will
        be tracked across each epoch of training
    epochs : int (optional, default=5)
        the number of epochs to train model on
    batch_size : int (optional, default=64)
        the size of batch to use during training
    reporting_interval : int (optional, default=1)
        the training progress reprting interval - progress is reported at end
        of each 'reporting_interval' epoch (e.g. if reporting_interval=10), progress
        will be reported on the first epoch and every 10th epoch thereafter.

    Methods
    -------
    fit(...) -> MetricsHistory
        cross-trains the model across epochs (i.e. the training & cross-validation loop)
    evaluate(...)
        evaluates model's performance on data
    predict(...)
        generates predictions on data
    """

    def __init__(
        self,
        loss_fn: torch.nn.modules.loss._Loss,
        device: torch.device,
        metrics_map: MetricsMapType = None,
        epochs: int = 5,
        batch_size: int = 64,
        # reporting_interval: int = 1,
    ):
        """constructs a Trainer object to be used for cross-training, evaluation of and
        getting predictions from an nn.Module instance.

        Parameters
        ----------
        loss_fn: torch.nn.modules.loss._Loss
            The loss function to be used during training & evaluation (pass an instance
            of one of PyTorch's loss functions (e.g. nn.BCELoss(), nn.CrossEntropyLoss() etc.)
        device: type torch.device
            The device on which training/evaluation etc. should happen (e.g., torch.device("cuda")
            for GPU or torch.device("cpu") for CPU). The model & data are automatically moved to
            device during the training process - the developer is NOT required to do so explicitly.
        metrics_map: dict {str : torchmetrics.Metric} (optional, default None)
            Optionally specifies the metrics ** IN ADDITION TO ** the loss metric that should be
            tracked across epochs during the cross-training loop. The `loss` metric will **ALWAYS**
            be tracked and you should not specify it in the metrics map - the training loop will use
            the loss function specified in the constructor to calculate the loss metric. Metrics are
            tracked across epochs for the training set and, if specified, the validation set.

            The `metrics_map` is of type `Dict[str, torchmetric.Metric]`. You can define several
            metrics to be monitored during the training/validation process. Each metric has a user-defined
            abbreviation (the `str` part of the Dict) and an associated torchmetric.Metric _formula_
            to calculate the metric.

            Examples of `metrics_map`(s):
               Example 1:
                metric_map = {
                    "acc": torchmetrics.classification.BinaryAccuracy()
                }
                Here 'acc' is the user chosen abbreviation for the `torchmetrics.classification.BinaryAccuracy()`
               Example 2:
                metrics_map = {
                    "acc": torchmetrics.classification.BinaryAccuracy(),
                    "f1": torchmetrics.classification.BinaryF1Score(),
                    "roc_auc": torchmetrics.classification.BinaryAUROC(thresholds = None)
                }
               Example 2:
                # you can also use your own function to calculate the metric
                def my_metric_calc(logits, labels):
                    metric = ....
                    return metric
                metrics_map = {
                    "my_metric" : my_metric_calc(),
                    "acc": torchmetrics.classification.BinaryAccuracy(),
                }
        epochs: int  (optional, default 5)
            Number of epochs for which the module should train (must be a +ve int)
        batch_size: int (optional,default 64)
            Size of batch used during training. Specify a negative value if you want to
            use the _entire_ dataset as the batch (not a good practice, unless your dataset
            itself is very small)
         - shuffle (bool, default=True) - should the data be shuffled during training?
             If True, it is shuffled else not. It is a good practice to always shuffle
             data between epochs. This setting should be left to the default value,
             unless you have a valid reason NOT to shuffle data (training and cross-validation)
         - num_workers: number of workers
        """
        if loss_fn is None:
            raise ValueError("FATAL ERROR: Trainer() -> 'loss_fn' cannot be None")
        if device is None:
            raise ValueError("FATAL ERROR: Trainer() -> 'device' cannot be None")
        if epochs < 1:
            raise ValueError("FATAL ERROR: Trainer() -> 'epochs' >= 1")
        # batch_size can be -ve
        batch_size = -1 if batch_size < 0 else batch_size

        self.loss_fn = loss_fn
        self.device = device
        self.metrics_map = metrics_map
        self.epochs = epochs
        self.batch_size = batch_size
        # self.reporting_interval = reporting_interval
        # self.shuffle = shuffle
        # self.num_workers = num_workers

    def fit(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataset: Union[
            NumpyArrayTuple, torch.utils.data.Dataset, torch.utils.data.DataLoader
        ],
        validation_dataset: Union[
            NumpyArrayTuple, torch.utils.data.Dataset, torch.utils.data.DataLoader
        ] = None,
        validation_split: float = 0.0,
        l1_reg: float = None,
        lr_scheduler: Union[LRSchedulerType, ReduceLROnPlateauType] = None,
        early_stopping: EarlyStopping = None,
        reporting_interval: int = 1,
        verbose: bool = True,
        logger: logging.Logger = None,
        shuffle: bool = True,
        num_workers: int = 4,
        seed=41,
    ) -> MetricsHistory:
        """
        fits (i.e cross-trains) the model using provided training data (and optionally
        cross-validation data).

        Parameters
        ----------
        model:  torch.nn.Module
            An instance of the model (derived from nn.Module) being trained.
        optimizer: torch.optim.Optimizer
            An instance of the optimizer used to adjust the weights during training.
            Use anyone of the optimizers defined in the torch.optim package, such as torch.optim.Adam
        train_dataset: `tuple(np.ndarray,np.ndarray)` OR `torch.utils.data.Dataset` or `torch.utils.data.Dataset`
            The training data to be used for training the model. This could be anyone of:
                - a tuple of Numpy arrays (e.g., (X_train, y_train))
                - an instance of torch.utils.data.Dataset class.
                - an instance of torch.utils.data.Dataloader class.
            It is easier to use the tuple of Numpy arrays when you are loading structured tabular data, the
            most common scikit datasets would fall into this category (e.g. Iris, Wisconsin etc.)
            For more complex data types, such as images, audio or text, you could use the Dataset or Dataloader
        validation_dataset: `tuple(np.ndarray,np.ndarray)` OR `torch.utils.data.Dataset` or `torch.utils.data.Dataset`
            (optional, default = None)
            The evaluation dataset to be used during cross-training. As with train_dataset, this could be one
            of `np.ndarray tuple` or `torch.utils.data.Dataset` or `torch.utils.data.DataLoader` class.
            **NOTE:** Be sure to use the same type for both train_dataset and validation_dataset
        validation_split: float (optional, default=0.2)
            Used as an alternative to validation_dataset, and when you want to randomly split a certain
            percentage of the train_dataset as your validation dataset.
            Should be specified a percentage (as a float >= 0.0 and < 1.0)
            For example, if validation_split = 0.2, then 20% of train dataset will be set aside
            for cross-validation.
        l1_reg: float (optional, default=None)
            The amount of L1 regularization to be applied to the weights during back-propogation
        lr_scheduler: torch.optim.lr_scheduler type or torch.optim.lr_scheduler.ReduceLROnPlateau
            (optional, default = None)
            The learning rate scheduler to be applied during training. Use either an ReduceLROnPlateau
            instance or any one of the LR schedulers defined by Pytorch (StepLR, ExponentialLR etc.)
        early_stopping: EarlyStopping (optional, default=None)
            Use to early stop training (before all epochs are done) if one of the metrics tracked is not
            improving for a set of X epochs. This should be an instance of EarlyStopping class defined
            as part of this toolkit.
        reporting_interval: int (optional, default 1)
            The interval (i.e. number of epochs) after which progress will be reported when training.
            By default, training progress is reported/printed after each epoch. Set this value to
            change the interval for example, if reporting_interval = 5, progress will be reported on
            the first, then every fifth epoch thereafter, and the last epoch (example: if epochs=22,
            and reporting_interval=5, then progress will be reported on 1, 5, 10, 15, 20 and 22 epoch).
        verbose: bool (optional, default=True)
            Specify if the training progress across training data is to be reported at end of each batch
            (True) or end of epoch (False).
        logger: logging,.Logger (optional, default=None)
            An instance of the logging.Logger class to be used to log progress as the model trains.
            **NOTE:** the torch training tooklit provides a `get_logger()` function with some default
            console & file logging configurations, which you can use, if needed.
        shuffle: bool (optional, default=True)
            To specify if the training (and validation data, if specified) should be shuffled before
            each epoch (True) or not (False). It is a good practice to shuffle the datasets, unless you
            are working with time-series datasets (or any other datasets) where data must be fed to the
            nn.Module in the sequence in which it is read.
        num_workers: int (optiona, default=4)
            The numbers of workers to be used when shuffling the train & cross-validation datasets.
        seed: int (optional, default=41)
            The seed to be used when intializing random number operations in the training loop.

        Returns
        --------
        An instance of MetricsHistory class defined in torch training toolkit.
        This object logs the epoch-wise history of the loss metric and all other metrics specified
        in the `metrics_map` parameter of the Trainer class instance.
            Example:
            # suppose you created a Trainer object like this
                metrics_map = {
                    "acc": torchmetrics.classification.BinaryAccuracy(),
                    "f1": torchmetrics.classification.BinaryF1Score(),
                }
                # for brevity all other Trainer() constructor parameters are excluded
                trainer = Trainer(..., metrics_map=metrics_map)
                module = nn.Module(...)  # your model
                # called fit() to kick of training passing in both training & cross-val datasets
                metrics = trainer.fit(...train_dataset=train_dataset, validation_dataset=val_dataset,...)
                # at the end of fit() call,, the return value would be like this
                    metrics_log = {
                        "loss" : [...],     # list of floats, len = # epochs, each value = average "loss" per epoch
                        "val_loss" : [...], # the corresponding "loss" valued for validation dataset
                        "acc" :  [...],     # list of floats, len = # epochs, each value = average "acc" per epoch
                        "val_acc" : [...],  # the corresponding values for validation dataset
                        "f1: : [...],
                        "val_f1" : [...],
                    }
                # NOTE: if validation_dataset (or validation_split) is not specified during the trainer.fit(...)
                # call, then all the "val_XXX" keys & values will be missing.
        """

        # check for some of the parameters & call cross_train_module(...) where all the action happens
        assert model is not None, "FATAL ERROR: Trainer.fit() -> 'model' cannot be None"
        assert (
            optimizer is not None
        ), "FATAL ERROR: Trainer.fit() -> 'optimizer' cannot be None"
        assert (
            train_dataset is not None
        ), "FATAL ERROR: Trainer.fit() -> 'train_dataset' cannot be None"
        assert (
            num_workers >= 0
        ), "FATAL ERROR: Trainer.fit() -> 'num_workers' must be >= 0"
        reporting_interval = 1 if reporting_interval < 1 else reporting_interval

        history = cross_train_module(
            model,
            train_dataset,
            self.loss_fn,
            optimizer,
            device=self.device,
            validation_split=validation_split,
            validation_dataset=validation_dataset,
            metrics_map=self.metrics_map,
            epochs=self.epochs,
            batch_size=self.batch_size,
            l1_reg=l1_reg,
            reporting_interval=reporting_interval,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping,
            shuffle=shuffle,
            num_workers=num_workers,
            verbose=verbose,
            seed=seed,
            logger=logger,
        )
        return history

    def evaluate(
        self,
        model: nn.Module,
        dataset: Union[
            NumpyArrayTuple, torch.utils.data.Dataset, torch.utils.data.DataLoader
        ],
        verbose: bool = True,
        logger: logging.Logger = None,
    ) -> MetricsValType:
        """
        Evaluates mode performance on a dataset. The model is put into an evaluation
        stage before running throughs the rows of the dataset. Should be called typically
        after training is completed.

        Parameters
        ----------
        model:  nn.Module
            an instance of the module being evaluated
        dataset: (`tuple(np.ndarray,np.ndarray)` OR `torch.utils.data.Dataset` or `torch.utils.data.Dataset`
            the dataset or dataloader on which the model should be evaluated
        verbose: bool (optional, default=True)
            display progress as model is evaluated (True) or not (False)
        logger: logging.Logger (optional, default=None)
            instance of logger class to be  used for logging

        Returns
        -------
        MetricsValType: Dict[str, float]
            a dictionary of loss and metrics defined in the metrics_map class + their respective iaverage
            values across the dataset
            Example: suppose the metrics map passed into the Trainer classe's constructor was
                metrics_map = {
                    "acc": torchmetrics.classification.BinaryAccuracy(),
                    "f1": torchmetrics.classification.BinaryF1Score(),
                }
            then this call will return a Dict with 3 values as follows:
                metric_value = {
                    "loss" : <<single value for loss - a float>>,
                    "acc" : <<single value for acc - a float>>,
                    "f1" : <<single value for f1 - a float>>,
                }
            The loss metric will ALWAYS be calculated.
            If you did not pass a metrics_map instance to the Trainer constructor, the you will
            get a dict with just the loss value, like so
                metric_value = {
                    "loss" : <<single value for loss, a float >>
                }

            Each metric can then be access as follows
                trainer = Trainer(...)
                model = nn.Module(...)  # your model
                ret = trainer.evaluate(model, train_dataset)
                print(f"Loss: {ret['loss']:.3f} - {ret['acc']:.3f}")
        """
        return evaluate_module(
            model,
            dataset,
            self.loss_fn,
            device=self.device,
            metrics_map=self.metrics_map,
            batch_size=self.batch_size,
            verbose=verbose,
            logger=logger,
        )

    def predict(
        self,
        model: nn.Module,
        dataset: Union[
            np.ndarray,
            torch.Tensor,
            NumpyArrayTuple,
            torch.utils.data.Dataset,
            torch.utils.data.DataLoader,
        ],
        logger: logging.Logger = None,
    ) -> Union[np.ndarray, NumpyArrayTuple]:
        """
        runs predictions on the model and returns prediction values

        Parameters
        ----------
        model: nn.Module
            the instance of your model from which to run predictions
            (NOTE: model should be trained first & all weights should be loaded into model)
        dataset: np.ndarray OR torch.Tensor OR (np.ndarray, np.ndarray) OR torch.utils.data.Dataset OR torch.utils.data.Dataloader
            An instance of any one of the following on which the predictions should be run
                - np.ndarray - a Numpy array (will be typicall used for structured datasets, such as scikit-datasets)
                - torch.Tensor - a tensor version of the above array (if you prefer, you can pass in a 1D tensor instead)
                - (np.ndarray, np.ndarray) - a tuple (X, y) of Numpy arrays
                - torch.utils.data.Dataset
                - torch.utils.data.Dataloader
        logger: logging.Logger (optional, default=None)
            instance of logger class to be used for logging

        Returns
        -------
        Either a single np.ndarray (when inputs are np.ndarray or torch.Tensor). This represents the array
        of predicted y values.
        OR
        A tuple of np.ndarray (np.ndarray, np.ndarray), when the inputs are a tuple of ndarrays or Dataset
        or Dataloader instances. This represents instances of y_predicted (first element) and y_actual (2nd
        element of tuple retuerned)
        NOTE: shape of return values is the same as the shape of values returned from the output layer
        of the module
        """
        if isinstance(dataset, np.ndarray) or isinstance(dataset, torch.Tensor):
            return predict_array(model, dataset, self.device, self.batch_size)
        else:
            return predict_module(
                model,
                dataset,
                self.device,
                self.batch_size,
                # for_regression=for_regression,
                logger=logger,
            )
