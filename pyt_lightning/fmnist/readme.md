# Multiclass Classification of Fashion MNIST Dataset with Pytorch &amp; Lightning
This code illustrates multi-class classification of the Fashion MNIST dataset using a **Pytorch  Convolutional Neural Network (CNN)**. Training is done using Pytorch Lightning.

Code is organized as follows:
* `fmnist_dataset.py` - (down-)loads dataset and returns `torch.util.data.Dataset` objects for `train`, `cross-val` and `test` datasets. The `test dataset` is randomly split into `cross-val` and `test` datasets in the process.
* `fmnist_model.py` - declares the model class (derived from lightning's `LightningModule` class) - the core network is a CNN.
* `pyt_lightning_fmnist.py` - the main driver module which cross-trains module & generates predictions.

## To Train/Run
Run the following command from within the IDE (setting up appropriate run configuration) or from the command line

```
$> python pyt_lightning_fmnist.py --train --pred [--epochs=50] [--batch_size=64] [--show_sample]

Where all options in [] are optional
```
Command line options:
```
  --train        - specify this flag to train model
  --pred         - specify this flag to generate predictions (must have trained model first!)
  --epochs=N     - specify the number N of epochs to train (optional, defaults to N = 25)
  --batch_size=B - specify the size of batch to use when training (optional, defaults to B = 64)
  --show_sample  - optional, if specified displays a random sample of images from test data

  **NOTE**: After training model's state is saved to `./model_state/pyt_fmnist_cnn.pth` file

  Example (NOTE: `$>` denotes the command line prompt)
  
  # train model & generate predictions. Model trained for 50 epochs with batch size = 128
  $> python pyt_lightning_fmnist.py --train --pred --epochs=50 --batch_size=128

  # train model & generate predictions. Also show sample images. 
  # Use (default) epochs=25 and batch_size=64
  $> python pyt_lightning_fmnist.py --train --pred --show_sample

  # generate predictions (assuming model trained before)
  $> python pyt_lightning_fmnist.py --pred
  ```