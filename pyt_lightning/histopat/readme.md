# Binary Classification of Medical Images dataset - Histopathologic Cancer Detection
This code illustrated *binary classification* using a CNN for the Histopathologic Cancer Detection dataset to identify metastatic cancer in small image patches taken from larger digital pathology scans.

The source image dataset has `96x96x3 px` images. Cancer detection is based on the *center `32x32 px` region* if a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 


Code is organized as follows:
* `histo_dataset.py` - (down-)loads dataset and returns `torch.util.data.Dataset` objects for `train`, `cross-val` and `test` datasets. The `test dataset` is randomly split into `cross-val` and `test` datasets in the process.
* `histo_model.py` - declares the model class (derived from lightning's `LightningModule` class) - the core network is a CNN.
* `pyt_lightning_histopatho.py` - the main driver module which cross-trains module & generates predictions.

## To Train/Run
Run the following command from within the IDE (setting up appropriate run configuration) or from the command line

```
$> python pyt_lightning_histopatho.py --train --eval --pred [--epochs=50] [--batch_size=64] [--show_sample]

Where all arguments in [] are optional
```
Command line options:
```
  --train        - specify this flag to train model
  --eval         - evaluate performance on datasets (must have trained the model first)
  --pred         - specify this flag to generate predictions (must have trained model first!)
  NOTE: You can specify just `--train` or `--eval` or `--test`, but `--eval` & `--test` require that model is trained before. So you should have at least one successful run with the `--train` flag, which saves model state. The model state is used by the `--eval` and `--test` flags.
  --epochs=N     - specify the number N of epochs to train (optional, defaults to N = 25)
  --batch_size=B - specify the size of batch to use when training (optional, defaults to B = 64)
  --show_sample  - optional, if specified displays a random sample of images from test data

  **NOTE**: After training model's state is saved to `./model_state/pyt_histo_cnn.pth` file

  Example (NOTE: `$>` denotes the command line prompt)
  
  # train model & generate predictions. Model trained for 50 epochs with batch size = 128
  $> python pyt_lightning_fmnist.py --train --pred --epochs=50 --batch_size=128

  # train model & generate predictions. Also show sample images. 
  # Use (default) epochs=25 and batch_size=64
  $> python pyt_lightning_fmnist.py --train --pred --show_sample

  # generate predictions (assuming model trained before)
  $> python pyt_lightning_fmnist.py --pred
  ```