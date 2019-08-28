
from __future__ import print_function, division
import os
import torch
torch.multiprocessing.set_start_method('spawn', force="True")
#import torch.multiprocessing as mp
#mp.set_start_method('spawn')
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import numpy
import h5py
import copy

from PIL import Image, ImageSequence

import cv2

import torchtext.data as data
import torchtext.datasets as datasets

import os
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, Sentence, DocumentPoolEmbeddings
from flair.embeddings import BytePairEmbeddings
from flair.embeddings import BertEmbeddings

#import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from typing import List, Union
from pathlib import Path

from models.imagenet import mobilenetv2

#from sklearn.metrics import confusion_matrix
import seaborn as sn
import nn_models as mod
import H5Dataset as H5


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# class H5Dataset(Dataset):
#
#     def __init__(self, path, data_transforms, embedding_model, embeddings_combination, type_model, phase):
#         print('dataset init')
#         self.model_embedding = embedding_model
#         self.combination_embeddings = embeddings_combination
#         self.file_path = path
#         self.dataset = None
#         self.data = None
#         self.target = None
#         self.ocr = None
#         self.phase = phase
#         self.type_model = type_model
#         with h5py.File(self.file_path, 'r') as file:
#             if phase == 'train':
#                 self.dataset_len = len(file["train_img"])
#             elif phase == 'val':
#                 self.dataset_len = len(file["val_img"])
#             elif phase == 'test':
#                 self.dataset_len = len(file["test_img"])
#
#         self.data_transforms = data_transforms
#
#     def __len__(self):
#         return self.dataset_len
#
#     def __getitem__(self, idx):
#         if self.dataset is None:
#             if self.phase == 'train':
#                 self.dataset = h5py.File(self.file_path, 'r')
#                 self.data = self.dataset.get('train_img')
#                 self.target = self.dataset.get('train_labels')
#                 self.ocr = self.dataset.get('train_ocrs')
#             elif self.phase == 'val':
#                 self.dataset = h5py.File(self.file_path, 'r')
#                 self.data = self.dataset.get('val_img')
#                 self.target = self.dataset.get('val_labels')
#                 self.ocr = self.dataset.get('val_ocrs')
#             elif self.phase == 'test':
#                 self.dataset = h5py.File(self.file_path, 'r')
#                 self.data = self.dataset.get('test_img')
#                 self.target = self.dataset.get('test_labels')
#                 self.ocr = self.dataset.get('test_ocrs')
#
#         img = self.data[idx,:,:,:],
#         img = Image.fromarray(img[0].astype('uint8'), 'RGB')
#         #doc_class = torch.from_numpy(self.target[idx,:,:,:]).float()
#         doc_class = self.target[idx]
#         doc_class = doc_class.astype(np.uint8)
#         doc_class = torch.tensor(doc_class)
#
#         ocr_text = self.ocr[idx]
#         embeddings_matrix = torch.zeros(1910,self.model_embedding.embedding_length)
#
#         if self.data_transforms is not None:
#             try:
#                 image = self.data_transforms(img)
#             except:
#                 print("Cannot transform image: {}")
#
#         if self.type_model != 'image':
#
#             ocr = ocr_text #ocr_text
#             # #print(ocr)
#             if ocr == '':
#                 ocr = 'empty'
#
#             sentence = Sentence(ocr, use_tokenizer = True)
#
#             #flair_embedding_fast = FlairEmbeddings('multi-forward-fast')
#             #flair_embedding_fast.embed(sentence)
#             self.model_embedding.embed(sentence)
#
#             if self.combination_embeddings == 'documentpool':
#                 embeddings_matrix = sentence.get_embedding() #extracts one single vector per document
#             else:
#                 #counter = 0
#                 embeddings_matrix = torch.zeros(1910,self.model_embedding.embedding_length) #max number of tokens in document = 1910
#                 for i, token in enumerate(sentence):
#                     #print(token)
#                     token_embedding = token.embedding
#                     token_embedding = token_embedding.unsqueeze(0)
#                     embeddings_matrix[i] = token_embedding
#
#
#         sample = {'image': image, 'class': doc_class, 'ocr': embeddings_matrix}#embeddings_matrix
#
#         return sample


def main():
    max_epochs = 70
    batch_size = 32
    num_classes = 10
    full_model = 'text'
    image_model = 'mobilenetv2'
    image_training_type = 'finetuning'
    text_model = 'cnn' #rnn to include
    combined_embeddings = 'stack'
    model_path = './resources/taggers/small_tobacco/'
    learning_rate = 0.001

    """
    :param batch_size: batch size
    :param num_classes: number of classes (10)
    :param full_model: model used: 'image', 'text' or 'combined'
    :param image_model: if image model used, specify which type: 'dense' (densenet121) or 'mobilenetv2'
    :param image_training_type: type of training of the image model: 'finetuning' (whole net is trained) or 'feature_extract' (only classifier is trained)
    :param text_model: type of text model: 'cnn', 'linear'
    :param combined_embeddings: process of combining different embeddings: 'individual', 'stack', 'documentpool', 'documentRNN'
    """

    if combined_embeddings == 'documentpool':
        text_model = 'linear'

    flair_embedding_forward = FlairEmbeddings('./Data/news-forward-0.4.1.pt')#2048
    fast_embedding = WordEmbeddings('en')#en-fasttext-news-300d-1M
    byte_embedding = BytePairEmbeddings(language= 'en')#Files stores in .flair/embeddings/en/...
    flair_embedding_backward = FlairEmbeddings('./Data/news-backward-0.4.1.pt')#1024
    glove_embedding = WordEmbeddings('glove')
    bert_embedding = BertEmbeddings()
    embeddings_models_list = [bert_embedding]


    if combined_embeddings == 'documentpool':
        embedding_model = DocumentPoolEmbeddings([fast_embedding,
                                              flair_embedding_backward,
                                              flair_embedding_forward]) #pooling argument: 'min', 'max' or 'mean' default
    elif combined_embeddings == 'stack':
        # Stacks n embeddings by concatenating them -> embedding_model_1 + embedding_model_1 + ... + embedding_model_n
        embedding_model = StackedEmbeddings(
        embeddings=embeddings_models_list)

    else:
        embedding_model = flair_embedding_forward #Change this


    total_embedding_length = embedding_model.embedding_length
    #print('Total length', total_embedding_length)

    #print('Embedding model', embedding_model)

    print('Model for training {}, {}'.format(full_model, combined_embeddings))

    # Class and Segmentation
    __all__ = ['MobileNetV2', 'mobilenetv2_19']

    input_size = 384#N.Audebert et al ->384

    #Image preprocessing
    # data_transforms = {
    # 'test': transforms.Compose([
    #     transforms.Resize(input_size),
    #     transforms.CenterCrop(input_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ]),
    # }
    data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.Grayscale(num_output_channels=3),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }


    #dataloader_train = DataLoader(documents_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)#=0 in inetractive job
    # for x in dataloader:
    #     x = x.to('cuda', non_blocking=True)

    feature_extracting = True
    output_image_model = 300

    if full_model == 'combined' or full_model == 'image':
        if image_model == 'mobilenetv2':
            #https://github.com/d-li14/mobilenetv2.pytorch
            net = mobilenetv2()
            net.load_state_dict(torch.load('pretrained/mobilenetv2_1.0-0c6065bc.pth'))
            if image_training_type == 'feature_extract':
                set_parameter_requires_grad(net, feature_extracting)
            num_ftrs = net.classifier.in_features
            net.classifier = nn.Linear(num_ftrs, output_image_model)

        elif image_model == 'dense':
             #Densenet
             use_pretrained = True
             net = models.densenet121(pretrained=use_pretrained)
             if image_training_type == 'feature_extract':
                 set_parameter_requires_grad(net, feature_extracting)
             num_ftrs = net.classifier.in_features
             net.classifier = nn.Linear(num_ftrs, output_image_model)
             #output_image_model = net.classifier.out_features

        if full_model == 'combined':
            if text_model == 'cnn':
                model = mod.CNN_Text_Image(net, total_embedding_length)
        elif full_model == 'image':
            net.classifier = nn.Linear(num_ftrs, num_classes) #Change classifier to 10 classes
            model = net

    #Add here RNN
    else:
        if text_model == 'cnn':
            model = mod.CNN_Text(total_embedding_length)
        else:
            model = mod.Fully_Connected(total_embedding_length)

    #Move model to gpu
    model = model.to(device)

    h5_dataset_test = H5.H5Dataset(path='./HDF5_files/hdf5_small_tobacco_audebert.hdf5', data_transforms=data_transforms['test'], embedding_model = embedding_model, embeddings_combination = combined_embeddings, type_model = full_model, phase = 'test')

    dataloader_test = DataLoader(h5_dataset_test, batch_size=1, shuffle=False, num_workers=0)#=0 in inetractive job

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.00004)

    PATH = model_path + 'text/700.005bert700.pt'#/image/2000.01_loading_best.pt
    #PATH = model_path + 'text700.01.pt'

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(loss)
    print(epoch)


    print('Testing...')
    model.eval()

    correct = 0
    max_size = 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        #print('Antes de batches')
        for i, local_batch in enumerate(dataloader_test):
            image, ocr_text, labels = Variable(local_batch['image']), Variable(local_batch['ocr']), Variable(local_batch['class'])
            # if max_size < ocr_text.shape[1]:
            #     max_size =  ocr_text.shape[1]
            #     print(max_size)
            if i%400==0:
                print(i)

            image, ocr_text, labels = image.to(device), ocr_text.to(device), labels.to(device)

            if full_model == 'combined':
                outputs = model(ocr_text, image)
            elif full_model == 'image':
                outputs = model(image)
            elif full_model == 'text':
                outputs = model(ocr_text)

            _, preds = torch.max(outputs.data, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                    if t.long() == p.long():
                        correct += 1

    print(confusion_matrix)

    print(confusion_matrix.diag()/confusion_matrix.sum(1))

    print('Accuracy: ', correct/2482)





if __name__ == '__main__':
    import flair, torch

    flair.device = torch.device('cuda')
    #print('GPU available', torch.cuda.is_available())
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    #print('Type of GPU', torch.cuda.get_device_name(0))
    #cuda = torch.device('cuda')
    main()
