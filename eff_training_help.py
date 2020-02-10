import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
#import h5py
import copy
import torchtext.data as data
import torchtext.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from efficientnet_pytorch import EfficientNet

import H5Dataset as H5

from allen_scheduler.slanted_triangular import SlantedTriangular
import csv

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

scratch_path = '/gpfs/scratch/bsc31/bsc31275/'

#Freezes weights of a model
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

#Returns actual learning rate value
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

input_size = 384#N.Audebert et al ->384
#Image preprocessing
data_transforms = {
'train': transforms.Compose([
    #transforms.RandomResizedCrop(input_size,scale=(0.08, 1.0),interpolation=3),
    #transforms.CenterCrop(input_size),
    transforms.Resize(input_size),
    transforms.RandomAffine((-5,5), translate=None, scale=None, shear=None, resample=False, fillcolor=0),
    #transforms.Grayscale(num_output_channels=3),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
'val': transforms.Compose([
    transforms.Resize(input_size),
    #transforms.Grayscale(num_output_channels=3),
    #transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
'test': transforms.Compose([
    transforms.Resize(input_size),
    #transforms.Grayscale(num_output_channels=3),
    #transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
}
