
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
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, Sentence

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

class CNN_Text_Image(nn.Module):

    def __init__(self, image_model):
        super(CNN_Text_Image, self).__init__()
        #self.args = args

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
        self.fc1 = nn.Linear(600, C)

        self.image_model = image_model

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        #Output will be size (1,Ks*Co) -> Maxpool will get one ĉ value =  max(c_1,c_2...), where c_i is
        #the result of the convolution operation of the kernel over the input

        return x



    def forward(self, x, x2):
        #x = self.embed(x)  # (N, W, D)

        #if self.args.static:
            #x = Variable(x)
        #print('CNN Text entry',x.shape)

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        #print('unsqueeze',x.shape)


        #print(x.shape)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)


        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)


        x = torch.cat(x, 1) #[1,100] + [1,100] + [1,100] = [1,300]

        #print('After cat', x.shape)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)

        x2 = self.image_model(x2)

        x2 = torch.cat((x,x2),1)


        logit = self.fc1(x2)
        #logit = self.fc1(x)  # (N, C)
        return logit




class H5Dataset(Dataset):

    def __init__(self, path, data_transforms, embedding_model,phase):
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
                self.dataset_len = len(file["train_img"])
            else:
                self.dataset_len = len(file["test_img"])

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

        sentence = Sentence(ocr, use_tokenizer = True)

        #flair_embedding_fast = FlairEmbeddings('multi-forward-fast')
        #flair_embedding_fast.embed(sentence)
        self.model_embedding.embed(sentence)
        #print('sentence embedded')

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




        counter = 0
        embeddings_matrix = torch.zeros(1910,2048) #max number of tokens
        for i, token in enumerate(sentence):
            #print(token)
            token_embedding = token.embedding
            token_embedding = token_embedding.unsqueeze(0)
            embeddings_matrix[i] = token_embedding


        sample = {'image': image, 'class': doc_class, 'ocr': embeddings_matrix}#embeddings_matrix

        return sample

def save(self, model_file):
    """
    Saves the current model to the provided file.
    :param model_file: the model file
    """
    model_state = self._get_state_dict()

    torch.save(model_state, str(model_file), pickle_protocol=4)




def main():
    batch_size = 32
    num_classes = 10
    full_model = 'image' #image, combined or text
    image_model = 'mobilenetv2'
    image_traing_type = 'finetuning'
    text_model = 'cnn' #rnn to include
    model_path = './resources/taggers/small_tobacco/'
    flair_embedding_forward = FlairEmbeddings('./Data/news-forward-0.4.1.pt')

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
    'val': transforms.Compose([
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
    h5_dataset = H5Dataset(path='./HDF5_files/hdf5_small_tobacco_papers.hdf5', data_transforms=data_transforms['train'], embedding_model = flair_embedding_forward, phase = 'train')

    dataloader_train = DataLoader(h5_dataset, batch_size=batch_size, shuffle=True, num_workers=0)#=0 in inetractive job
    # for x in dataloader:
    #     x = x.to('cuda', non_blocking=True)

    feature_extracting = True
    output_image_model = 300

    if full_model == 'combined' or 'image':
        if image_model == 'mobilenetv2':
            #https://github.com/d-li14/mobilenetv2.pytorch
            net = mobilenetv2()
            net.load_state_dict(torch.load('pretrained/mobilenetv2_1.0-0c6065bc.pth'))
            if image_traing_type == 'feature_extract':
                set_parameter_requires_grad(net, feature_extracting)
            num_ftrs = net.classifier.in_features
            net.classifier = nn.Linear(num_ftrs, output_image_model)

        elif image_model == 'dense':
             #Densenet
             use_pretrained = True
             net = models.densenet121(pretrained=use_pretrained)
             if image_traing_type == 'feature_extract':
                 set_parameter_requires_grad(net, feature_extracting)
             num_ftrs = net.classifier.in_features
             net.classifier = nn.Linear(num_ftrs, output_image_model)
             #output_image_model = net.classifier.out_features

        if full_model == 'combined':
            if text_model == 'cnn':
                model = CNN_Text_Image(net)
        elif full_model == 'image':
            net.classifier = nn.Linear(num_ftrs, num_classes) #Change classifier to 10 classes
            model = net

    else:
        print('text')


    model = model.to(device)

    print("Params to learn:")
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.00004)

    max_epochs = 100
    optimizer = optimizer_ft
    loss_values = []
    epoch_values = []

    for epoch in range(max_epochs):
        running_loss = 0
        steps = 0
        running_corrects = 0
        # Training
        print('Antes del loop')
        batch_counter = 0
        batchs_number= 800/batch_size
        for local_batch in dataloader_train:
            batch_counter += 1
            image, ocr_text, labels = Variable(local_batch['image']), Variable(local_batch['ocr']), Variable(local_batch['class'])
            steps += 1

            image, ocr_text, labels = image.to(device), ocr_text.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            if full_model == 'combined':
                outputs = model(ocr_text, image)
            elif full_model == 'image':
                outputs = model(image)
            elif full_model == 'text':
                outputs = model(ocr_text)
            _, preds = torch.max(outputs.data, 1)

            labels = labels.long()
            loss = criterion(outputs, labels)

            running_corrects += torch.sum(preds == labels.data)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
            #running_loss += loss.data[0]
            # if steps % 100 == 0:
            #     print(steps)
                #save(model,'./snapshot/', 'model', steps)
            running_loss += loss.item()
            if batch_counter % 10==0:
                print('[Batch {}/{}]'.format(
                                batch_counter, batchs_number))

        epoch_acc = running_corrects.double() / len(dataloader_train.dataset)
        print('[Epoch {}/{}], loss {}, Acc {}'.format(
                        epoch, max_epochs,running_loss/(800/batch_size), epoch_acc))

        loss_values.append(running_loss/800)
        epoch_values.append(epoch)


    h5_dataset_test = H5Dataset(path='./HDF5_files/hdf5_small_tobacco_papers.hdf5', data_transforms=data_transforms['test'], embedding_model = flair_embedding_forward, phase = 'test')

    dataloader_test = DataLoader(h5_dataset_test, batch_size=1, shuffle=False, num_workers=0)#=0 in inetractive job


    num_classes = 10

    torch.save(model.state_dict(), model_path  + 'image_model.pt')

    print('Testing...')
    model.eval()

    max_size = 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for i, local_batch in enumerate(dataloader_test):
            image, ocr_text, labels = Variable(local_batch['image']), Variable(local_batch['ocr']), Variable(local_batch['class'])
            # if max_size < ocr_text.shape[1]:
            #     max_size =  ocr_text.shape[1]
            #     print(max_size)

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

    print(confusion_matrix)

    print(confusion_matrix.diag()/confusion_matrix.sum(1))





if __name__ == '__main__':
    import flair, torch

    flair.device = torch.device('cuda')
    #print('GPU available', torch.cuda.is_available())
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    #print('Type of GPU', torch.cuda.get_device_name(0))
    #cuda = torch.device('cuda')
    main()
