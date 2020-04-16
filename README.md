# Improving accuracy and speeding up Document Image Classication through parallel systems

Paper: [Improving accuracy and speeding up Document Image Classification through parallel systems]()

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
