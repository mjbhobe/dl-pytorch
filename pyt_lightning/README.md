# Introducing Pytorch (En)lightening Module

Pytorch Lightning is a very capable Pytorch training library. Hard to believe, but it does have some limitations in my
opinion. For instance:

* It displays just 1 progress bar during training loop, so I cannot see how training is progressing across epochs.
  Keras displays a separate progress bar for each epoch, with metrics, so I can see how metrics are shaping up across
  my epochs.
* The 'current' training progress bar should display just the batch metrics, as they are updates, but the previous ones
  should show epoch level training & validation (if present) metrics.
* You usually land up writing (almost) the same code in the `training_step`, `validation_step` and `test_step`
  functions.
* Lacks a convenient function to track and plot metrics (both training & cross-validation) across epochs

To address the above _limitations_, I wrote a module, called `Pytorch (En)lightening` module that implements these
features:

* A new module `EnLitModule` that inherits `pl.LightningModule` and overloads ...
* A custom progress bar `EnLitProgressBar` that keeps displays a bar for each epoch of training _along with_ the epoch
  level metric values.
* A convenience class `MetricsLogger` that keeps track of metrics across epochs and which implements a `plot_metrics()`
  function, which will display `loss` and all metrics tracked across epochs, so you can _visually_ see if (and from
  which epoch) model is _overfitting_ or _underfitting_.

<< TODO: Finish this >>

