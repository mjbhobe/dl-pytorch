# Pytorch Toolkit - Tutorial
Thank you for your interest in the *Pytorch Toolkit* - an API that tucks away boilerplate code to _train_, _evaluate_ and _test_ Pytorch models, thus allowing you to focus on the core tasks of data preparation, defining the architecture of your model and tuning hyper-parameters for the same.

I am assuming that you have already installed the pre-requisites and have done the preliminary test as explained in the [Readme](Readme.md) file - if not, please do so now.

This tutorial will gradually expose you to the API provided by the **Pytorch Toolkit**, so it is best that you follow along from beginning to the end. I use a very informal style of writing, which I hope you'll like. The API is inspired by Keras, so if you have used Keras before you'll feel at home.

**One last thing**

In this tutorial, I **won't be covering tha rationale behind choosing a specific architecture for the model nor why I prepared the data in a specific way**. This document is all about how to use the Pytorch Toolkit's API and not on how to design ML models. You can refer to several books or online tutorials for guidance on model design. With that perspective, let's get started.

## Training with data available in Numpy arrays
Often data & labels are available in Numpy arrays, especially for structured datasets. For example: datasets available with the `scikit-learn` module (like Iris dataset, Boston Housing, the Wisconsin Breast Cancer dataset etc.) and in several repositories on Kaggle and UCI. 

We'll start with one such example, specifically a binary classification problem - classifying the Wisconsin Breast Cancer dataset. I'll be downloading the data from the [UCI Repository link](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)), but you can load it from the `scikit-learn datasets` module as well.

### Loading data
This section _does not strictly pertain to the Pytorch Toolkit API_, but I am showing the code nonetheless so you have some perspective on how our model can be easily trained on data & labels available as Numpy Arrays. Here is the code I used to load the Wisconsin Breast Cancer dataset - I am not providing detailed instructions here (as it does not pertain to the Pytorch toolkit)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# loading & preparing data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'

cols = [
    "id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean",
    "smoothness_mean","compactness_mean","concavity_mean","concave points_mean",
    "symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se",
    "area_se","smoothness_se","compactness_se","concavity_se","concave points_se",
    "symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst",
    "area_worst","smoothness_worst","compactness_worst","concavity_worst",
    "concave points_worst","symmetry_worst","fractal_dimension_worst"
]

df = pd.read_csv(url, header=None, names=cols, index_col=0)
# map 'M' (malignant) to 1 and 'B' (benign) to 0
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

X = wis_df.drop(['diagnosis'], axis=1).values
y = wis_df['diagnosis'].values

# split into train/test sets
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=test_split, random_state=seed)

# scale data using Standard scaling
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

```

### Creating our model
You would normally create your Pytorch model by deriving your class from `nn.Module` as follows:

```python
# import...
# Not showing other Python imports, including torch imports

class WBCNet(nn.Module):
    def __init__(self, inp_size, hidden1, hidden2, num_classes):
        super(WBCNet, self).__init__()
        self.fc1 = nn.Linear(inp_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        x = self.fc1(inp)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x
```

With the `Pytorch Toolkit` the **only change** is the base class from which you derive the module. I provide a class called `PytkModule`, which derives from `nn.Module` and provides additional functions to help with model training, evaluation and testing. Here is how you define your Module when using the Pytorch Toolkit:

```python
# import...
# Not showing other Python imports, including torch imports (same as before)

# import the Pytorch toolkit
import pytorch_toolkit as pytk

class WBCNet(pytk.PytkModule):
    def __init__(self, inp_size, hidden1, hidden2, num_classes):
        super(WBCNet, self).__init__()
        self.fc1 = nn.Linear(inp_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        x = self.fc1(inp)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x
```

Notice that **the only change is the base class from which your model is derived!**. You code the constructor `__init__(...)` and the `forward(...)` function as before!

Next, you'll need to specify the `loss` function, the `optimizer` and `metrics` to track during the training epochs (Yes, the `Pytorch Toolkit` provides several common metrics that you can use out-of the box - like Accuracy, F1-Score, MSE, MAE etc., which you can use). 

**NOTE:** At this time there is no support to add your own metrics - something that I plan on adding at a future date. I do include the most common metrics I normally use, so I believe this is not such a big shortcoming.

```python
# instantiate the loss function & optimizer (same as usual - nothing Pytorch toolkit specific here)
loss_fn = nn.BCELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
    nesterov=True, weight_decay=0.005, momentum=0.9, dampening=0) 
```

and here is how you _attach_ these to your model

```python
# NOTE: model is an instance of PytkModule class - it provides a compile() function
# which allows me to specify the loss function, optimizer and metrics to track
model.compile(loss=loss_fn, optimizer=optimizer, metrics=['acc'])
```

* Notice that the above line is very Keras-like (I've done this on purpose). Keras does a lot more with it's `compile(...)` function. I merely set attributes of the `PytkModule` derived class! 
* You can optionally specify one or more metrics that the training loop should track using the optional `metrics` parameter. Metrics are specified using a string abbreviation (just like in Keras) - refer to the table below for available metrics. **The current version of the Pytorch toolkit does not support user defined metrics :(**.
* You _need not specify any metrics_ (for example by omitting the optional`metrics` parameter in above call. The training loop will always track the `loss` metric though, _even if you leave out the metrics parameter in the `compile(). This is in addition to any other metrics you supply in the metrics parameter. You don't have to specifically mention the `loss` metric in your `metrics` parameter.

**Table of available metrics**

|String | Metric | 
|:---|:---|
|`'acc'`| Accuracy (works for binary and multiclass classification)
|`'prec'`| Precision
|`'rec'`| Recall
|`'f1'`| F1 Score
|`'roc_auc'`| ROC AUC Score
|`'mse'`| Mean Square Error
|`'rmse'`| Root Mean Square Error
|`'mae'`| Mean Absolute Error

### Training our model
Here is where the Pytorch Tooklit shines - **you don't have to write the code that loops through epochs, generates batches of data etc. etc.** - just call the model's `fit(...)` function as shown below:

```python
hist = model.fit(X_train, y_train, epochs=100, batch_size=16)
```
The `fit(...)` call takes many more parameters, which I will cover later. At the minumum, you need to specify the X (data), y (labels) [both Numpy arrays], the number of epochs to loop through and the batch size to use. You should see output like the one shown below (I have abbreviated the output for brevity)

```
Epoch (  1/100): (455/455) -> loss: 0.6900 - acc: 0.6296
Epoch (  2/100): (455/455) -> loss: 0.6696 - acc: 0.6373
...
... many more lines (truncated)
...
Epoch ( 99/100): (455/455) -> loss: 0.0512 - acc: 0.9871
Epoch (100/100): (455/455) -> loss: 0.0506 - acc: 0.9871
```
The output should be fairly easy to comprehend, especially if you have used Keras before. Basically each line shows the epoch wise progress of training followed by record count and then the metrics (here `loss` and `acc`) - `loss` will be tracked even if you don't specify any metrics.

#### Cross-training with validation data
It is always a good practice to cross-train your model on a `training` dataset and an `evaluation` dataset. You can easily accomplish this with the Pytorch Toolkit as shown below. Just modify the `fit(...)` call as follows:

```python
hist = model.fit(X_train, y_train, epochs=100, batch_size=16,
    validation_split=0.20)
```

Here I have used `validation_split` parameter, which takes a value between 0.0 and 1.0. This will internally randomly split the training data  & labels into `training` and `cross-validation` datasets using the `validation_split` proportion specified. Output will be slightly different, as shown below:

```
Epoch (  1/100): (364/364) -> loss: 0.6926 - acc: 0.6196 - val_loss: 0.6790 - val_acc: 0.6884
Epoch (  2/100): (364/364) -> loss: 0.6771 - acc: 0.6214 - val_loss: 0.6638 - val_acc: 0.6884
...
... many more (truncated)
...
Epoch ( 99/100): (364/364) -> loss: 0.0496 - acc: 0.9891 - val_loss: 0.0974 - val_acc: 0.9688
Epoch (100/100): (364/364) -> loss: 0.0489 - acc: 0.9891 - val_loss: 0.0971 - val_acc: 0.9688
```
Notice that metrics are now being tracked for the `training` data & labels as well as the `cross-validation` data and labels (values preceedes by `val_`)

#### Tracking multiple metrics
Suppose you want to track `accuracy` and `F1-Score` by epoch in the training loop. Here is what you do:

* Specify the metrics you want to track in your `model.compile(...)` call as follows (for metric abbreviations please refer to table above):
```python
model.compile(loss=loss_fn, optimizer=optimizer, metrics=['acc', 'f1']
```
* Call `fit(...)` as usual - no changes here!
```python
hist = model.fit(X_train, y_train, epochs=100, batch_size=16,
    validation_split=0.20)
```
You should see output like the following - notice the extra metric being tracked for the `training` and `cross-validation` datasets.

```
Epoch (  1/100): (364/364) -> loss: 0.6926 - acc: 0.6196 - f1: 0.0488 - val_loss: 0.6790 - val_acc: 0.6884 - val_f1: 0.0000
Epoch (  2/100): (364/364) -> loss: 0.6771 - acc: 0.6214 - f1: 0.0696 - val_loss: 0.6638 - val_acc: 0.6884 - val_f1: 0.0000
...
... many more
...
Epoch ( 99/100): (364/364) -> loss: 0.0496 - acc: 0.9891 - f1: 0.9868 - val_loss: 0.0974 - val_acc: 0.9688 - val_f1: 0.9681
Epoch (100/100): (364/364) -> loss: 0.0489 - acc: 0.9891 - f1: 0.9844 - val_loss: 0.0971 - val_acc: 0.9688 - val_f1: 0.9681
```

### Viewing model's performance
The `Pytorch Toolkit` provides a `show_plots(...)` function which plots the `loss` and `acc` (if specified) againsts epochs. This will quickly help you ascertain if your model if overfitting or underfitting the data it is trained on. 

The `fit(...)` call returns a `history` object, which is basically a map of all the metrics tracked across the various epochs (e.g. `hist['loss']` is a `list` of `loss` per epoch. So if you specified `epochs=100` in your `fit(...)` call, this would point to a list of 100 values and so on).

**After training is completed**, use the following call:

```python
pytk.show_plots(hist)
```

`show_plots(...)` is passed the value returned by `fit(...)` call. Since we specified the `acc` metric in our `model.compile(...)` call, the `fit(...)` call was tracking both `loss` and `acc`. The output if `show_plots(...)` will be something like below - on the left is the plot of loss vs epochs (for both the training & cross-validation datasets) and on the right is the plot for `acc` against epochs.

![](images/show_plots.png)

In this version of the `Pytorch Toolkit`, the `show_plot()` function plots only the `loss` and `acc` metric (if specified). I'll be adding support to plot other metrics in upcoming versions. For now, this suffices for most problems.

### Evaluating Model performance
Once you are done with training, you will want to verify model's performance on `testing` data & labels, which the model _has not seen_ during the entire cross-training process. This can be done as follows:

```python
# assuming you are tracking both accuracy & f1-score metrics
loss, acc, f1 = model.evaluate(X_test, y_test)
print(f'  Test dataset  -> loss: {loss:.4f} - acc: {acc:.4f} - f1: {f1:.4f}')
```

You'll see something like the following:

```
Evaluating (114/114) -> loss: 0.0622 - acc: 0.9922 - f1: 0.9906
  Test dataset  -> loss: 0.0622 - acc: 0.9922 - f1: 0.9906
```

**NOTE:** The `evaluate(...)` call returns _as many values_ as the metrics you are tracing (remember that the `fit(...)` call **always** tracks `loss` even if you do not specify it in the metrics list!). We specified `acc` and `f1`, hence we'll get back 3 values.

### Saving the model's state:
Once you are happy with the model's performance, you may want to save the model's weights and structure to disk, so you can just load this at a later date and use the model without having to re-train it. The `Pytorch Toolkit's PytModule` provides a `save()` function to do just that. Use it on an instance of the model as follows:

```python
# specify path of file where model state should be saved
model.save('./model_states/wbc.pt')
```

`save()` takes a path to file where the model state is saved - if the directory leading up to the path does not exist, it is created the very first time `save()` is called. You need not specify the `.pt` extension. It is tagged on if omitted. The `.pt` file is a binary file, which only Pytorch API can comprehend.

### Loading model's state from disk
The `Pytorch Toolkit` provides a **stand alone** `load()` function to load the model's state from disk - I could have coded this as a static function of the PytModule class, but I chose to code a stand-alone function instead. Here is how you use it.

```python
model = pytk.load_model('./model_states/wbc.pt')
```
This call will load the model's structure as well as the weights & biases of the various layers of the model. It is ready-to-go!

### Running Predictions
Once you have trained the model and are satisfied with the performace, you will run predictions on the _test_ data and labels (or even on the _training_ data and labels). Use the `predict(...)` call as follows:

```python
y_pred = model.predict(X_test)
```

**NOTE:**
* For a _binary classification_ problem, such as this one, `y_pred` is a Numpy array of shape `(X_train.shape[1], 1)`
* For a _multi-class classification_ problem, with `N` possible output classes, `y_pred` will be a Numpy array of shape `(X_train.shape[1], N)`

This completes our first example - please refer to the [complete code for this example here](pyt_breast_cancer.py).

