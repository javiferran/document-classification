import logging
import random
import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

# !pip install pytorch_transformers
from pytorch_transformers import AdamW  # Adam's optimization w/ fixed weight decay

from training_modules.finetuned_models import FineTunedBert
from training_modules.data_utils import H5Dataset
from training_modules.model_utils import train, test

from ensemble_modules.data_utils2 import H5Dataset_ensemble
from ensemble_modules.model_utils_ensemble import test_ensemble


import copy
import os
import csv

import os.path
parent_path = os.path.abspath(os.path.join('./', os.pardir))
scratch_path = '/gpfs/scratch/bsc31/bsc31275/'

input_size = 384

data_transforms = {
'train': transforms.Compose([
    #transforms.RandomAffine((-5,5), translate=None, scale=None, shear=None, resample=False, fillcolor=0),
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
'val': transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
'test': transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
}

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.bert.parameters():
            param.requires_grad = False
