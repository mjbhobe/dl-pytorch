# Deep Learning with Pytorch (with Keras like helper functions)

This project contains may illustrations of classifcation & regression using Pytorch

Highlight of the project is `pyt_helper_funcs.py` - utility functions for Pytorch,
which provide a Keras-like interface for training model, evaluating performance
and predicting results from model.

## pyt_helper_functions.py
Provides helper functions and utility class (pytModule) derived from nn.Module, which
provides a Keras like interface to train & evaluate model and generate predictions

## Examples:

### Classification Examples:
* `pyt_breast_cancer.py` - _binary classification_ on the `Wisconsin Breast Cancer dataset` using Pytorch ANN
* `pyt_iris.py` - _multiclass classification_ of `scikit-learn Iris dataset` using Pytorch ANN
* `pyt_wine.py` - _multiclass classification_ of `scikit-learn Wine` dataset
* `pyt_mnist_dnn.py` - `MNIST digits` _multiclass classification_ with Pytorch ANN
* `pyt_cifar10_cnn.py` - _multiclass classification_ with **CNN** on `CIFAR-10` dataset
* `Pytorch-Fruits360(Kaggle)_CNN.ipynb` - iPython Notebook for the [Kaggle Fruits360](https://www.kaggle.com/moltean/fruits) multiclass classification problem

### Regression Examples:
* `pyt_regression.py` - _univariate regression_ on synthesized data
* `pyt_salary_regression.py` - _multivariate regression_ on salary data (`@see csv_filed/salary_data.csv`)

