# Introducing Pytorch Toolkit (PyTk) 
## A Keras-like API to train, evaluate and test Pytorch Models

Jump straight to the [step-by-step tutorial](Tutorial.md)

**Pytorch**, with it's _Pythonic_ interface and _dynamic graph evaluation_ makes it much easier to debug your deep learning models compared with Tensorflow 1.0. However, on the flip side, it's still a _low level_ library, which requires us to (often) write the same (or almost same) boilerplate code to train, evaluate & test our model. Some ML authors love to code this way as they feel _in control_ of what they create.

Me? I _hate_ writing the same boilerplate code repeatedly. I'd rathen have my library handle the mundane tasks while I focus on the tasks that _really matter_ viz. designing my model architecture, preparing data for my model & tuning the hyper-parameters to get the best performance. Keras does a remarkable job of providing a clean & simple API to train, evaluate & test models. With the `Pytorch Toolkit` I aim to bring the ease that Keras provides to Pytorch. Most of the API is similar to the Keras API, so Keras users should find it very easy to understand.

### Features of the `Pytorch Toolkit`
* Keras-like API to train model and evaluate model performance (e.g. `fit()` and `evaluate()`) and make predictions (e.g. `predict()`)
* Support for torchvision datasets - with `fit_dataset()`, `evaluate_dataset()` and `predict_dataset()` calls
* Convenience class `PytkModule` from which to extend your model - this class provides the Keras-like API
* Full support for using `nn.Sequential` API, via the `PytkModuleWrapper` class, which also provides the same functions listed above
* Support for saving and loading model states with `save()` and `load()` methods
* Support for a variety of metrics (like accuracy, f1-score, precision, recall), calculated for each training epoch
* Keras-like progress display while model trains
* Support for **Early Stopping** of training

I must confess, I'm not good at coming up with snazzy names for the libraries I create. I tried several acronyms and finally settled on a rather unfancy name `Pytorch Toolkit` (or `PyTk`). If you can come up with a really cool name, please let me know!

This Github repository includes the tooklit, along with several examples on how to use it. Also included is a [step-by-step tutorial](Tutorial.md) which gradually introduces you to the complete API included in the toolkit. All functions & classes are included in just 1 Python file (_ingeniously_ named `pytorch_toolkit.py` - I did warn you that I am not good at coming up with names, didn't I?). I have not yet created a module - maybe someday...

## Installing Pytorch Toolkit
Since all functions & classes are included in just 1 file - `pytorch_toolkit.py`, there are strictly no special installation steps required to use this toolkit. 
* Clone this repository - so you get _entire_ Pytorch Tooklit and several example files
* Copy the `pytorch_toolkit.py` file into your project's directory and you are done! (Alternately, if you don't like several copies scattered across your disk drive, copy this file to any one folder in your Python SYSPATH)
* At the top of your code file (or Jupyter Notebook) and after all your other imports, enter the following code to import the toolkit into your project - I use the `pytk` alias - you can use whatever you prefer.

    ```python
    import pytorch_tooklit as pytk
    ```

* This library depends on several other Python libraries, viz:
    * Numpy
    * Pandas
    * Matplotlib
    * scikit-learn
    * Pytorch (of course!)
    * torchsummary - if you want to see a Keras-like summary of your model (optional!)

I am assuming that you have these installed already - if you are an aspiring Data Scientist or ML enthusiast, you would have these (except perhaps Pytorch & torchsummary). Please refer to the respective module documentation on how to install these libraries.

### Testing the installation
* I assume you have followed the instructions above & have installed the pre-requisites, including Pytorch itself.
* Start with a new Python code file (or Jupyter notebook) - add the following line at the top (after all your other imports):

    ```python
    # ... your imports including Pytorch imports

    import pytorch_tooklit as pytk
    ```
* Just run this file/ code-cell - if you don't get any import errors, you are done - celebrate your success!! :)
    * Should you get any import errors, please correct them by installing the respective module/library (Pytorch Tooklit dependencies are mentioned above)
    * Repeat the above step (running file) until you get no errors

## Features of the Pytorch Toolkit
This tookkit was inspired by Keras' clean API to train, evaluate and test models. Much of the functions provided _mimick_ those from the Keras API. If you are already using Keras, you should notice the similarities immmediately. 

The toolkit provides:
* A custom class - `PytkModule` from which your custom models should derive
* Keras-like API to train your models - `fit(...)`, `fit_dataset(...)` functions and a `show_plots(...)` function to plot `loss` and `accuracy` metric across epochs, so you can see if your model is overfitting or underfitting.
* Several pre-defined metrics, like Accuracy, F1-Score, MSE, MAE etc., which can be tracked during training.
* Keras-like API to evaluate model's performance post training - `evaluate(...)` and `avaluate_dataset(...)` functions
* Functions to save & load model's state - `pytk.load()` and `save(...)` functions.
* Keras-like API to run predictions - `predict(...)` and `predict_dataset(...)` calls.
* Early stopping of training based on several criteria (e.g. validation loss not improving)

I am also **including several fully working examples** (as Python files or as Jupyter notebooks), where I have applied this API to solve several ML problems. The [step-by-step tutorial](Tutorial.md) will refer to one or more of these examples. I'll be adding more example to this Github repository, so please check back for changes.

If you are excited about starting with the Pytorch Toolkit, jump to the [step-by-step tutorial](Tutorial.md) right away!

### Classification Examples:
* `pyt_breast_cancer.py` - _binary classification_ on the `Wisconsin Breast Cancer dataset` using Pytorch ANN
* `pyt_iris.py` - _multiclass classification_ of `scikit-learn Iris dataset` using Pytorch ANN
* `pyt_wine.py` - _multiclass classification_ of `scikit-learn Wine` dataset
* `pyt_mnist_dnn.py` - `MNIST digits` _multiclass classification_ with Pytorch ANN
* `pyt_cifar10_cnn.py` - _multiclass classification_ with **CNN** on `CIFAR-10` dataset
* `Pytorch-Fruits360(Kaggle)_CNN.ipynb` - iPython Notebook for the [Kaggle Fruits360](https://www.kaggle.com/moltean/fruits) multiclass classification problem
* `Pytorch-Malaria Cell Detection(Kaggle)_CNN.ipynb` - iPython Notebook for the [Kaggle Malaric Cell Detection Dataset](https://www.kaggle.com/moltean/fruitshttps://www.kaggle.com/iarunava/cell-images-for-detecting-malaria) binary classification problem

### Regression Examples:
* `pyt_regression.py` - _univariate regression_ on synthesized data
* `pyt_salary_regression.py` - _multivariate regression_ on salary data (`@see csv_filed/salary_data.csv`)

I will be adding mode examples over the course of time. Keep watching this repository :).

Hope you enjoy using the Pytorch Toolkit - my small contribution to the Pytorch community. Feedback is welcome.


