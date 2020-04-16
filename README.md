# Improving accuracy and speeding up Document Image Classication through parallel systems

Paper: [Improving accuracy and speeding up Document Image Classification through parallel systems]()

## Datasets

SmallTobacco files can be downloaded [here](https://lampsrv02.umiacs.umd.edu/projdb/project.php?id=72). In _Data_ folder we provide the scripts for getting ocr .txt files (ocr_tobacco.py) and for creating .hdf5 files (ST_hdf5_dataset_creation.py) with images and ocr data.

BigTobacco files can be downloaded [here](http://www.cs.cmu.edu/~aharley/rvl-cdip/). _./Data/BT_hdf5_dataset_creation.py_ creates train, test and validation .hdf5 files based on the aforementioned link partition.


## Repository structure

## Image model

```
├── image_model
	├── eff_big_training.py # EfficientNet training in BigTobacco
	├── eff_small_training.py # EfficientNet training in SmallTobacco
	├── eff_utils.py # EfficientNet helper with common functions for Small and Big training
	├── H5Dataset.py # Dataset class reading hdf5 file
	├── tensorflow
		├── distr_effnet_shear.py # EfficientNet
```

## Text model

```
├── text_model
	├── main.py # BERT training in SmallTobacco
	├── bert_utils.py # BERT helpers
	├── training_modules
		├── data_utils.py # data cleaning and H5Dataset class
		├── finetuned_models.py # BERT model definition
		├── model_utils.py # train and test procedures
```

## Ensemble

```
├── text_model
	├── ensemble.py # ensemble image and text predictions
	├── bert_utils.py # BERT helpers
	├── ensemble_modules
		├── data_utils2.py # data cleaning and H5Dataset_ensemble class
		├── model_utils_ensemble.py # BERT and EfficientNet predictions and ensemble
```

_efficientnet_pytorch_ library downloads the models in .cache/torch/checkpoints.

_pytorch_transformers_ library does it in .cache/torch/pytorch_transformers. Make sure you previously download and store in those paths the models if your machine has no internet access.
