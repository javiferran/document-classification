
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

from PIL import Image, ImageSequence

import cv2

import torchtext.data as data
import torchtext.datasets as datasets

import os
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, Sentence, DocumentPoolEmbeddings
from flair.embeddings import BytePairEmbeddings


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


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class CNN_Text(nn.Module):

    def __init__(self, image_model, embedding_dim):
        super(CNN_Text, self).__init__()
        #self.args = args
        self.image_dim = 128
        self.embedding_out_dim = 128

        #V = args.embed_num
        D = 2048 #embed_dim, 4196 for doc_embeddings
        C = 10 #class_num
        Ci = 1
        Co = 100 #kernel_num -> number of kernel with the same size
        Ks = [3,4,5] #kernel_sizes -> size = number of words

        #self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])


        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(0.5)
        #self.fc1 = nn.Linear(len(Ks)*Co, C)

        self.fc1 = nn.Linear(embedding_dim, self.embedding_out_dim)
        self.fc2 = nn.Linear(self.image_dim + self.embedding_out_dim, C)

        self.image_model = image_model


    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        #Output will be size (1,Ks*Co) -> Maxpool will get one ĉ value =  max(c_1,c_2...), where c_i is
        #the result of the convolution operation of the kernel over the input

        return x



    def forward(self, x, x2):

        x = self.fc1(x)#128

        x2 = self.image_model(x2)#128

        x2 = torch.cat((x,x2),1)#256

        #print('Concatenation shape', x2.shape)


        logit = self.fc2(x2)
        #logit = self.fc1(x)  # (N, C)
        return logit




class H5Dataset(Dataset):

    def __init__(self, path, data_transforms, embedding_model, phase):
        print('dataset init')
        self.model_embedding = embedding_model
        self.file_path = path
        self.dataset = None
        self.data = None
        self.target = None
        self.ocr = None
        self.phase = phase
        with h5py.File(self.file_path, 'r') as file:
            if phase == 'train':
                self.dataset_len = len(file["train_labels"])
            else:
                self.dataset_len = len(file["test_labels"])

        self.data_transforms = data_transforms

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.dataset is None:
            if self.phase == 'train':
                self.dataset = h5py.File(self.file_path, 'r')
                self.data = self.dataset.get('train_img')
                self.target = self.dataset.get('train_labels')
                self.ocr = self.dataset.get('train_ocrs')
            else:
                self.dataset = h5py.File(self.file_path, 'r')
                self.data = self.dataset.get('test_img')
                self.target = self.dataset.get('test_labels')
                self.ocr = self.dataset.get('test_ocrs')

        img = self.data[idx,:,:,:],
        img = Image.fromarray(img[0].astype('uint8'), 'RGB')
        #doc_class = torch.from_numpy(self.target[idx,:,:,:]).float()
        doc_class = self.target[idx]
        doc_class = doc_class.astype(np.uint8)
        doc_class = torch.tensor(doc_class)

        ocr_text = self.ocr[idx]

        if self.data_transforms is not None:
            try:
                image = self.data_transforms(img)
            except:
                print("Cannot transform image: {}")

        ocr = ocr_text #ocr_text
        # #print(ocr)
        if ocr == '':
            ocr = 'empty'

        sentence = Sentence(ocr)
        self.model_embedding.embed(sentence)

        # counter = 0
        # for token in sentence:
        #     #print(token)
        #     token_embedding = token.embedding
        #     token_embedding = token_embedding.unsqueeze(0)
        #     #print(token_embedding)
        #     #print(token_embedding.shape)
        #     if counter == 0:
        #         prev_token_embedding = token_embedding
        #     if counter != 0:
        #         prev_token_embedding = torch.cat((prev_token_embedding, token_embedding),0)
        #     counter += 1

        self.model_embedding.embed(sentence)

        #print('Document embedding', sentence.get_embedding().shape)

        prev_token_embedding = sentence.get_embedding()

        sample = {'image': image, 'class': doc_class, 'ocr': prev_token_embedding}

        return sample

# def save(self, model_file):
#     """
#     Saves the current model to the provided file.
#     :param model_file: the model file
#     """
#     model_state = self._get_state_dict()
#
#     torch.save(model_state, str(model_file), pickle_protocol=4)




def main():
    image_model = 'mobilenetv2'
    model_path = './resources/taggers/small_tobacco/'
    embeddings_length = []
    flair_embedding_forward = FlairEmbeddings('./Data/news-forward-0.4.1.pt')
    fast_embedding = WordEmbeddings('en')#en-fasttext-news-300d-1M
    byte_embedding = BytePairEmbeddings(language= 'en')#Files stores in .flair/embeddings/en/... dim=100
    flair_embedding_backward = FlairEmbeddings('./Data/lm-news-english-backward-1024-v0.2rc.pt')

    embeddings_length.append(fast_embedding)
    embeddings_length.append(byte_embedding)


    document_embeddings = DocumentPoolEmbeddings([byte_embedding,
                                                fast_embedding,
                                              #flair_embedding_backward,
                                              #flair_embedding_forward
                                              ])
    total_embedding_length = 0
    for embedding in embeddings_length:
        total_embedding_length += embedding.embedding_length


    # Dataset creation with image directory, image -> 'RGB' -> transformed to Mobilenetv2 input, Ocr,
    # Class and Segmentation
    __all__ = ['MobileNetV2', 'mobilenetv2_19']
    input_size = 224

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    #Independent train and test transformations can be done
    h5_dataset = H5Dataset(path='./HDF5_files/hdf5_small_tobacco_papers.hdf5', data_transforms=data_transforms['train'], embedding_model = document_embeddings, phase = 'train')
    batch_size = 40
    dataloader_train = DataLoader(h5_dataset, batch_size=batch_size, shuffle=True, num_workers=0)#=0 in inetractive job
    # for x in dataloader:
    #     x = x.to('cuda', non_blocking=True)

    feature_extracting = True
    output_image_model = 128

    if image_model == 'mobilenetv2':
        #https://github.com/d-li14/mobilenetv2.pytorch
        net = mobilenetv2()
        net.load_state_dict(torch.load('pretrained/mobilenetv2_1.0-0c6065bc.pth'))
        set_parameter_requires_grad(net, feature_extracting)
        #num_ftrs = net.classifier.in_features
        net.classifier = nn.Linear(1280, 128)

    elif image_model == 'dense':
         #Densenet
         use_pretrained = True
         net = models.densenet121(pretrained=use_pretrained)
         set_parameter_requires_grad(net, feature_extracting)
         num_ftrs = net.classifier.in_features
         net.classifier = nn.Linear(num_ftrs, output_image_model)
         #output_image_model = net.classifier.out_features


    combined_length = output_image_model + total_embedding_length
    # get model
    model = CNN_Text(net, total_embedding_length)
    model = model.to(device)
    #print(model)
    # define loss function
    criterion = nn.CrossEntropyLoss()
    # set optimize
    optimizer_ft = optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.00004)

    max_epochs = 5
    optimizer = optimizer_ft
    running_loss = 0.0
    loss_values = []
    epoch_values = []

    # Loop over epochs
    for epoch in range(max_epochs):
        running_loss = 0
        steps = 0
        # Training
        print('Antes del loop')
        batch_counter = 0
        batchs_number= 800/batch_size
        running_corrects = 0
        for local_batch in dataloader_train:
            batch_counter += 1
            image, ocr_text, labels = Variable(local_batch['image']), Variable(local_batch['ocr']), Variable(local_batch['class'])
            #print(ocr_text.shape)
            steps += 1
            if ocr_text.shape[1]< 5:
                #print(steps)
                pass
            else:
                image, ocr_text, labels = image.to(device), ocr_text.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                #print(local_batch['image_dir'])

                # forward
                outputs = model(ocr_text, image)
                _, preds = torch.max(outputs.data, 1)
                #print(preds, labels.long())
                labels = labels.long()
                loss = criterion(outputs, labels)
                #print(outputs)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()
                #running_loss += loss.data[0]

                running_corrects += torch.sum(preds == labels.data)

                running_loss += loss.item()
                if batch_counter % 10==0:
                    print('[Batch {}/{}]'.format(
                                    batch_counter, batchs_number))

        epoch_acc = running_corrects.double() / len(dataloader_train.dataset)
        print('[Epoch {}/{}], loss {}, Acc {}'.format(
                        epoch, max_epochs,running_loss/(800/batch_size), epoch_acc))


        loss_values.append(running_loss/(800/batch_size))
        epoch_values.append(epoch)


    h5_dataset_test = H5Dataset(path='./HDF5_files/hdf5_small_tobacco_papers.hdf5', data_transforms=data_transforms['test'], embedding_model = document_embeddings, phase = 'test')

    dataloader_test = DataLoader(h5_dataset_test, batch_size=1, shuffle=False, num_workers=0)#=0 in inetractive job


    torch.save(model.state_dict(), model_path  + 'combined_model.pt')

    nb_classes = 10

    print('Testing...')
    model.eval()
    test_counter = 0

    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, local_batch in enumerate(dataloader_test):
            image, ocr_text, labels = Variable(local_batch['image']), Variable(local_batch['ocr']), Variable(local_batch['class'])
            if ocr_text.shape[1]< 5:
                pass
            else:
                test_counter += 1
                image, ocr_text, labels = image.to(device), ocr_text.to(device), labels.to(device)
                outputs = model(ocr_text, image)
                _, preds = torch.max(outputs.data, 1)
                for t, p in zip(labels.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

    print(confusion_matrix)

    print(confusion_matrix.diag()/confusion_matrix.sum(1))
    print(test_counter)

    #save(base_path / "final-model.pt")




if __name__ == '__main__':
    import flair, torch

    flair.device = torch.device('cuda')
    #print('GPU available', torch.cuda.is_available())
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    #print('Type of GPU', torch.cuda.get_device_name(0))
    #cuda = torch.device('cuda')
    main()
