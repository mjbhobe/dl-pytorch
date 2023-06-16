"""
utils.py - utility functions
"""
import os
import pathlib
import numpy as np
import torch
import torch.nn as nn

from typing import Tuple, Union


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
    # save only if save_dir is a directory
    if pathlib.Path(save_dir).is_dir():
        torch.save(model.to("cpu").state_dict(), model_save_path)
        if verbose:
            print(f"Pytorch model saved to {model_save_path}")
    else:
        raise ValueError(
            f"FATAL ERROR: {model_save_path} does not appear to be a path!\n"
            f"HINT: {pathlib.Path(save_dir)} exists but does not appear to be a directory."
        )


def load_model(model: nn.Module, model_state_dict_path: str, verbose: bool = True) -> nn.Module:
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
        raise IOError(f"ERROR: can't load model from {model_state_dict_path} - file does not exist!")

    # load state dict from path
    state_dict = torch.load(model_save_path)
    model.load_state_dict(state_dict)
    if verbose:
        print(f"Pytorch model loaded from {model_state_dict_path}")
    model.eval()
    return model


NumpyArrayTuple = Tuple[np.ndarray, np.ndarray]


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
        assert isinstance(model, nn.Module), "predict() works with instances of nn.Module only!"
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


def predict_module(
    model: nn.Module,
    dataset: Union[NumpyArrayTuple, torch.utils.data.Dataset, torch.utils.data.DataLoader],
    device: torch.device,
    batch_size: int = 64,
    num_iters=-1,
) -> NumpyArrayTuple:
    """make predictions from array or dataset or DataLoader
    @params:
       - model: instance of nn.Module (or LightningModule)
       - dataset: instance of Numpy array tuple (X, y) or torch Dataset or torch DataLoader
       - device: device on which to run predictions (GPU or CPU)
       - batch_size (optional, default=64): batch size to use when iterating over Dataset
         or DataLoader (applies only if dataset parameter is an instance of Dataset or
         DataLoader, ignored otherwise)
       - num_iters (optional, default=-1): number of iterations to run over dataset to generate
         predictions. -1 means run over all records. P\
    @returns:
       - Tuple T of numpy arrays: T[0] - predictions, T[1] actual values
         len(T[0]) == len(T[1]) == len(dataset)
         for a multi-class classification problem, don't forget to run a
         np.argmax(T[0], axis=1) after this call, so that predictions are reduces to class nos
    """
    try:
        model = model.to(device)
        num_iters = -1 if num_iters <= 0 else num_iters

        with torch.no_grad():
            model.eval()

            preds, actuals = [], []

            # if dataset is a tuple of np.ndarrays, convert to torch Dataset
            if isinstance(dataset, tuple):
                X = torch.from_numpy(dataset[0]).type(torch.FloatTensor)
                y = torch.from_numpy(dataset[1]).type(
                    torch.LongTensor if dataset[1].dtype in [np.dtype(int)] else torch.FloatTensor
                )
                dataset = torch.utils.data.TensorDataset(X, y)
                loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            elif isinstance(dataset, torch.utils.data.Dataset):
                loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            elif isinstance(dataset, torch.utils.data.DataLoader):
                loader = dataset
            else:
                raise ValueError(
                    f"Dataset is incorrect type - expecting one of [NumpyArrayTuple, torch.utils.data.Dataset, torch.utils.data.Dataloader]"
                )

            iter_count = 0
            for X, y in loader:
                X = X.to(device)
                y = y.to(device)
                with torch.no_grad():
                    model.eval()
                    batch_preds = list(model(X).to("cpu").numpy())
                    batch_actuals = list(y.to("cpu").numpy())
                    preds.extend(batch_preds)
                    actuals.extend(batch_actuals)
                iter_count += 1
                if (num_iters != -1) and (iter_count >= num_iters):
                    break

            return (np.array(preds), np.array(actuals))
    finally:
        model = model.to("cpu")
