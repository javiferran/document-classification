
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

class Fully_Connected(nn.Module):

    def __init__(self,embedding_dim):
        super(Fully_Connected, self).__init__()

        input_linear = embedding_dim
        C = 10

        self.fc1 = nn.Linear(input_linear, C)

    def forward(self, x):

        logit = self.fc1(x)

        return logit

class CNN_Text_Image(nn.Module):

    def __init__(self, image_model, embedding_dim):
        super(CNN_Text_Image, self).__init__()
        #self.args = args

        #V = args.embed_num
        D = embedding_dim #embed_dim, 4196 for doc_embeddings
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

        return logit

class CNN_Text(nn.Module):

    def __init__(self, embedding_dim):
        super(CNN_Text, self).__init__()
        #self.args = args

        #V = args.embed_num
        D = embedding_dim #embed_dim, 4196 for doc_embeddings
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
        self.fc1 = nn.Linear(300, C)


    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        #Output will be size (1,Ks*Co) -> Maxpool will get one ĉ value =  max(c_1,c_2...), where c_i is
        #the result of the convolution operation of the kernel over the input

        return x



    def forward(self, x):
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


        logit = self.fc1(x)

        return logit


class H5Dataset(Dataset):

    def __init__(self, path, data_transforms, embedding_model, embeddings_combination ,phase):
        print('dataset init')
        self.model_embedding = embedding_model
        self.combination_embeddings = embeddings_combination
        self.file_path = path
        self.dataset = None
        self.data = None
        self.target = None
        self.ocr = None
        self.phase = phase
        with h5py.File(self.file_path, 'r') as file:
            if phase == 'train':
                self.dataset_len = len(file["train_img"])
            elif phase == 'val':
                self.dataset_len = len(file["val_img"])
            elif phase == 'test':
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
            elif self.phase == 'val':
                self.dataset = h5py.File(self.file_path, 'r')
                self.data = self.dataset.get('val_img')
                self.target = self.dataset.get('val_labels')
                self.ocr = self.dataset.get('val_ocrs')
            elif self.phase == 'test':
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

        if self.combination_embeddings == 'documentpool':
            embeddings_matrix = sentence.get_embedding() #extracts one single vector per document
        else:
            #counter = 0
            embeddings_matrix = torch.zeros(1910,self.model_embedding.embedding_length) #max number of tokens in document = 1910
            for i, token in enumerate(sentence):
                #print(token)
                token_embedding = token.embedding
                token_embedding = token_embedding.unsqueeze(0)
                embeddings_matrix[i] = token_embedding


        sample = {'image': image, 'class': doc_class, 'ocr': embeddings_matrix}#embeddings_matrix

        return sample


def main():
    max_epochs = 50
    batch_size = 32
    num_classes = 10
    full_model = 'text'
    image_model = 'mobilenetv2'
    image_training_type = 'finetuning'
    text_model = 'cnn' #rnn to include
    combined_embeddings = 'stack'
    model_path = './resources/taggers/small_tobacco/'
    learning_rate = 0.015

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
    flair_embedding_backward = FlairEmbeddings('./Data/news-backward-0.4.1.pt')#2048
    glove_embedding = WordEmbeddings('glove')

    if combined_embeddings == 'documentpool':
        embedding_model = DocumentPoolEmbeddings([fast_embedding,
                                              flair_embedding_backward,
                                              flair_embedding_forward]) #pooling argument: 'min', 'max' or 'mean' default
    elif combined_embeddings == 'stack':
        # Stacks n embeddings by concatenating them -> embedding_model_1 + embedding_model_1 + ... + embedding_model_n
        #embedding_model = StackedEmbeddings(embeddings=[flair_embedding_forward, flair_embedding_backward, fast_embedding])
        embedding_model = StackedEmbeddings(embeddings=[fast_embedding, byte_embedding, glove_embedding])

    else:
        embedding_model = flair_embedding_forward #Change this


    total_embedding_length = embedding_model.embedding_length
    print('Total length', total_embedding_length)

    #print('Embedding model', embedding_model)

    print('Model for training {}, {}'.format(full_model, combined_embeddings))

    # Class and Segmentation
    __all__ = ['MobileNetV2', 'mobilenetv2_19']

    input_size = 224#N.Audebert et al ->384

    #Image preprocessing
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
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


    documents_datasets = {x: H5Dataset(path='./HDF5_files/hdf5_small_tobacco_papers.hdf5', data_transforms=data_transforms[x], embedding_model = embedding_model, embeddings_combination = combined_embeddings, phase = x) for x in ['train', 'val']}
    dataloaders_dict = {x: DataLoader(documents_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}


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
                model = CNN_Text_Image(net, total_embedding_length)
        elif full_model == 'image':
            net.classifier = nn.Linear(num_ftrs, num_classes) #Change classifier to 10 classes
            model = net

    #Add here RNN
    else:
        if text_model == 'cnn':
            model = CNN_Text(total_embedding_length)
        else:
            model = Fully_Connected(total_embedding_length)

    #Move model to gpu
    model = model.to(device)




    print("Params to learn:")
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.00004)

    optimizer = optimizer_ft
    loss_values = []
    epoch_values = []

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    for epoch in range(max_epochs):
        since = time.time()
        steps = 0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            batch_counter = 0

            for local_batch in dataloaders_dict[phase]:
                batchs_number = len(dataloaders_dict[phase].dataset)/batch_size
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
                if phase == 'train':
                    loss.backward()
                    optimizer.step()


                running_loss += loss.item()
                # if batch_counter % 10==0:
                #     print('[Batch {}/{}]'.format(
                #                     batch_counter, batchs_number))


            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('[Epoch {}/{}] {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, max_epochs,phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        print()

        # epoch_acc = running_corrects.double() / len(dataloader_train.dataset)
        # print('[Epoch {}/{}], loss {}, Acc {}'.format(
        #                 epoch, max_epochs,running_loss/(800/batch_size), epoch_acc))


        model.load_state_dict(best_model_wts)


    #torch.save(model.state_dict(), model_path  + full_model + str(max_epochs) + '.pt')


    h5_dataset_test = H5Dataset(path='./HDF5_files/hdf5_small_tobacco_papers.hdf5', data_transforms=data_transforms['test'], embedding_model = embedding_model, embeddings_combination = combined_embeddings, phase = 'test')

    dataloader_test = DataLoader(h5_dataset_test, batch_size=1, shuffle=False, num_workers=0)#=0 in inetractive job


    num_classes = 10



    print('Testing...')
    model.eval()

    max_size = 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        #print('Antes de batches')
        for i, local_batch in enumerate(dataloader_test):
            image, ocr_text, labels = Variable(local_batch['image']), Variable(local_batch['ocr']), Variable(local_batch['class'])
            # if max_size < ocr_text.shape[1]:
            #     max_size =  ocr_text.shape[1]
            #     print(max_size)
            #print('Antes de a gpu')

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
