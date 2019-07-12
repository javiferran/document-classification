#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class H5Dataset(Dataset):

    def __init__(self, hdf5_file, data_transforms):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.h5_file = h5py.File(hdf5_file, "r")
        self.data = self.h5_file.get('train_img')
        self.target = self.h5_file.get('train_labels')
        self.ocr = self.h5_file.get('train_ocrs')
        self.data_transforms = data_transforms

    def __len__(self):
        print(self.data.shape[0])
        return self.data.shape[0]

    def __getitem__(self, idx):
        
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
        
        
        
        
        method = 'word_embedding'
        #method = 'doc_embedding'
        
        if method == 'word_embedding':
            ocr = ocr_text #ocr_text

            sentence = Sentence(ocr)
            
            #flair_embedding_fast = FlairEmbeddings('multi-forward-fast')
            #flair_embedding_fast.embed(sentence)
            
            flair_embedding_forward = FlairEmbeddings('news-forward')
            flair_embedding_forward.embed(sentence)
            
            

            #flair_embedding_fast.embed(sentence)
            counter = 0
            for token in sentence:
                #print(token)
                token_embedding = token.embedding
                token_embedding = token_embedding.unsqueeze(0)
                #print(token_embedding)
                #print(token_embedding.shape)
                if counter == 0:
                    prev_token_embedding = token_embedding
                if counter != 0:
                    prev_token_embedding = torch.cat((prev_token_embedding, token_embedding),0)
                counter += 1
                
        else:
            ocr_processed = 'Hello World is awesome .'
            sentence_doc = Sentence(ocr_processed)
            document_embeddings.embed(sentence_doc)
        
            print('Document embedding', sentence_doc.get_embedding().shape)
        
            prev_token_embedding = sentence_doc.get_embedding()
        
        
        
        print('Sentence embedded size',prev_token_embedding.shape)
            
        #document_embeddings.embed(sentence_doc)
        
        #print('Document embedding', sentence_doc.get_embedding().shape)
        
        #prev_token_embedding = sentence_doc.get_embedding()
        
        sample = {'image': image, 'class': doc_class, 'ocr': prev_token_embedding}

        return sample



class CNN_Text(nn.Module):
    
    def __init__(self):
        super(CNN_Text, self).__init__()
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
        self.fc1 = nn.Linear(len(Ks)*Co, C)

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

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        
        
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
        logit = x
        #logit = self.fc1(x)  # (N, C)
        return logit


## Mega NN

class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, expansion=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes*expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes*expansion)
        self.conv2 = nn.Conv2d(inplanes*expansion, inplanes*expansion, kernel_size=3, stride=stride,
                            padding=1, bias=False, groups=inplanes*expansion)
        self.bn2 = nn.BatchNorm2d(inplanes*expansion)
        self.conv3 = nn.Conv2d(inplanes*expansion, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MobileNetV2(nn.Module):

    def __init__(self, block, layers, text_model, num_classes=16):
        self.inplanes = 32
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1, expansion = 1)
        self.layer2 = self._make_layer(block, 24, layers[1], stride=2, expansion = 6)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2, expansion = 6)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2, expansion = 6)
        self.layer5 = self._make_layer(block, 96, layers[4], stride=1, expansion = 6)
        self.layer6 = self._make_layer(block, 160, layers[5], stride=2, expansion = 6)
        self.layer7 = self._make_layer(block, 320, layers[6], stride=1, expansion = 6)
        self.conv8 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, bias=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.conv9 = nn.Conv2d(1280,num_classes, kernel_size=1, stride=1, bias=False)
        
        #Added
        #Fully connected
        self.fc1 = nn.Linear(310, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.text_model = text_model
                

    #def conv_and_pool(self, x, conv):
        #x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        #x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #return x

    def _make_layer(self, block, planes, blocks, stride, expansion):

        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes,
                    kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, expansion=expansion))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion))

        return nn.Sequential(*layers)

    def forward(self, x, x2): #NN input -> Image + ocr text
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.conv8(x)
        x = self.avgpool(x)
        x = self.conv9(x)
        
        #print('conv9 output', x.shape)
        
        x = x.view(x.size(0),-1)
        
        
        #print(x.size(0))#1Xnum_classes size
        
        #print('mobilenet output', x.shape)
        
        
        x2 = self.text_model(x2)
        
        #print('CNN Text output',x2.shape)
        
        #print('Text model output (without last layer)', x2.shape)
                
        x2 = torch.cat((x,x2),1)
        
        x2 = self.fc1(x2)
        
        #print('MegaNet output',x2)
        
        #print('Output shape', x2.shape)

        return x2


def mobilenetv2_19(text_model, **kwargs):
    """Constructs a MobileNetV2-19 model.
    """
    model = MobileNetV2(Bottleneck, [1, 2, 3, 4, 3, 3, 1], text_model, **kwargs)
    return model



def main():
    # Dataset creation with image directory, image -> 'RGB' -> transformed to Mobilenetv2 input, Ocr,
    # Class and Segmentation
    __all__ = ['MobileNetV2', 'mobilenetv2_19']

    data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}
    #Independent train and test transformations can be done
    h5_dataset = H5Dataset(hdf5_file='./HDF5_files/hdf5_small_tobacco.hdf5', data_transforms=data_transforms['train'])

    dataloader = DataLoader(h5_dataset, batch_size=1, shuffle=False, num_workers=0)

    # get model
    text_model = CNN_Text()
    model = mobilenetv2_19(text_model, num_classes = 10)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # set optimizer
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00004)

    max_epochs = 1
    optimizer = optimizer_ft
    batch_size=1
    running_loss = 0.0
    steps = 0

    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        for local_batch in dataloader:
            print('Hola')
            image, ocr_text, labels = Variable(local_batch['image']), Variable(local_batch['ocr']), Variable(local_batch['class'])
            # zero the parameter gradients
            optimizer.zero_grad()
            #print(local_batch['image_dir'])

            # forward
            outputs = model(image, ocr_text)
            _, preds = torch.max(outputs.data, 1)
            #print(preds, labels.long())
            loss = criterion(outputs, labels.long())
            #print(outputs)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
            #running_loss += loss.data[0]
            steps += 1
            if steps % 100 == 0:
                save(model,'./snapshot/', 'model', steps)
            
            #print(outputs)
            print('[Epoch {}/{}], loss {}'.format(
                            epoch, max_epochs,loss))
            
    save(model,'./snapshot/', 'model', steps)




if __name__ == '__main__':
    main()
