import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model,
    criterion,
    optimizer,
    trainset,
    valset=None,
    metric=None,
    device=torch.device("cpu"),
    num_epochs=10,
    batch_size=64,
    num_workers=0,
    seed=41,
):
    torch.manual_seed(seed)
    tot_samples = len(trainset)
    len_num_epochs, len_tot_samples = len(str(num_epochs)), len(str(tot_samples))
    model = model.to(device)

    hist = {"loss": []}
    if valset is not None:
        hist["val_loss"] = []

    if metric is not None:
        hist["metric"] = []
        if valset is not None:
            hist["val_metric"] = []

    if valset is None:
        print(f"Training on {device} with {len(trainset)} training records")
    else:
        print(
            f"Cross-training on {device} with {len(trainset)} training and {len(valset)} validation records"
        )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Training the model
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        running_loss = 0.0
        running_metric = 0.0
        num_batches, samples = 0, 0

        model.train()
        for _, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if metric is not None:
                metric_val = metric(outputs, labels)
            optimizer.step()

            running_loss += loss.item()
            if metric is not None:
                running_metric += metric_val.item()
            samples += len(inputs)
            num_batches += 1

            if metric is not None:
                print(
                    "\rEpoch (%*d/%*d): (%*d/%*d) -> loss: %.3f - metric: %.3f"
                    % (
                        len_num_epochs,
                        epoch + 1,
                        len_num_epochs,
                        num_epochs,
                        len_tot_samples,
                        samples,
                        len_tot_samples,
                        tot_samples,
                        loss.item(),
                        metric_val.item(),
                    ),
                    end="",
                    flush=True,
                )
            else:
                print(
                    "\rEpoch (%*d/%*d): (%*d/%*d) -> loss: %.3f"
                    % (
                        len_num_epochs,
                        epoch + 1,
                        len_num_epochs,
                        num_epochs,
                        len_tot_samples,
                        samples,
                        len_tot_samples,
                        tot_samples,
                        loss.item(),
                    ),
                    end="",
                    flush=True,
                )
        else:
            epoch_loss = running_loss / num_batches
            hist["loss"].append(epoch_loss)
            epoch_metric = None
            if metric is not None:
                epoch_metric = running_metric / num_batches
                hist["metric"].append(epoch_metric)

            contd = "..." if valset is not None else ""

            if metric is not None:
                print(
                    "\rEpoch (%*d/%*d): (%*d/%*d) -> loss: %.3f - metric: %.3f%s"
                    % (
                        len_num_epochs,
                        epoch + 1,
                        len_num_epochs,
                        num_epochs,
                        len_tot_samples,
                        samples,
                        len_tot_samples,
                        tot_samples,
                        epoch_loss,
                        epoch_metric,
                        contd,
                    ),
                    end="",
                    flush=True,
                )
            else:
                print(
                    "\rEpoch (%*d/%*d): (%*d/%*d) -> loss: %.3f%s"
                    % (
                        len_num_epochs,
                        epoch + 1,
                        len_num_epochs,
                        num_epochs,
                        len_tot_samples,
                        samples,
                        len_tot_samples,
                        tot_samples,
                        epoch_loss,
                        contd,
                    ),
                    end="",
                    flush=True,
                )

            val_metric = None
            if valset is not None:
                valloader = DataLoader(
                    valset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                )
                model.eval()
                val_loss, val_metric, num_batches = 0.0, 0.0, 0
                with torch.no_grad():
                    for _, (inputs, labels) in enumerate(valloader):
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        if metric is not None:
                            metric_val = metric(outputs, labels)
                        # optimizer.zero_grad()
                        # loss.backward()
                        # optimizer.step()

                        val_loss += loss.item()
                        if metric is not None:
                            val_metric += metric_val.item()
                        num_batches += 1
                hist["val_loss"].append(val_loss / num_batches)
                if metric is not None:
                    hist["val_metric"].append(val_metric / num_batches)

            # display final metrics for epoch
            if metric is not None:
                print(
                    "\rEpoch (%*d/%*d): (%*d/%*d) -> loss: %.3f - metric: %.3f - val_loss: %.3f - val_metric: %.3f"
                    % (
                        len_num_epochs,
                        epoch + 1,
                        len_num_epochs,
                        num_epochs,
                        len_tot_samples,
                        samples,
                        len_tot_samples,
                        tot_samples,
                        epoch_loss,
                        epoch_metric,
                        val_loss / num_batches,
                        val_metric / num_batches,
                    ),
                    flush=True,
                )
            else:
                print(
                    "\rEpoch (%*d/%*d): (%*d/%*d) -> loss: %.3f - val_loss: %.3f"
                    % (
                        len_num_epochs,
                        epoch + 1,
                        len_num_epochs,
                        num_epochs,
                        len_tot_samples,
                        samples,
                        len_tot_samples,
                        tot_samples,
                        epoch_loss,
                        val_loss / num_batches,
                    ),
                    flush=True,
                )
    return hist


def evaluate_model(
    model,
    dataset,
    metric,
    batch_size=64,
    device=torch.device("cpu"),
    num_workers=4,
    shuffle=True,
    seed=41,
) -> float:
    # Evaluate the model on the test set
    torch.manual_seed(seed)
    model = model.to(device)
    model.eval()
    running_metric, num_batches = 0.0, 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            metric_val = metric(outputs, labels)
            running_metric += metric_val.item()
            num_batches += 1
            # _, predicted = torch.max(outputs, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
    # acc = 100.0 * correct / total
    # return acc
    # return the average across batches
    return running_metric / num_batches


def predict_model(
    model,
    dataset,
    device=torch.device("cpu"),
    batch_size=64,
    shuffle=False,
    num_workers=0,
    seed=41,
):
    torch.manual_seed(seed)
    actuals, predictions = [], []
    model = model.to(device)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        # preds = preds.cpu().numpy()
        predictions.extend(list(preds.ravel().numpy()))
        actuals.extend(list(labels.ravel().numpy()))

    return np.array(actuals), np.array(predictions)


def save_model(model: nn.Module, model_save_path: str, verbose: bool = True):
    """saves Pytorch state (state_dict) to disk
    @params:
        - model: instance of model derived from nn.Module
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
            print(f"Unable to create folder/directory {save_dir} to save model!")
            raise err

    # now save the model to file_path
    # NOTE: a good practice is to move the model to cpu before saving the state dict
    # as this will save tensors as CPU tensors, rather than CUDA tensors, which will
    # help load model on any machine that may or may not have GPUs
    torch.save(model.to("cpu").state_dict(), model_save_path)
    if verbose:
        print(f"Pytorch model saved to {model_save_path}")


def load_model(
    model: nn.Module, model_state_dict_path: str, verbose: bool = True
) -> nn.Module:
    """loads model's state dict from file on disk
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
