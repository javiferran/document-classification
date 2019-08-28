
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
#from torchvision.transforms import functional
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
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

from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, Sentence, DocumentPoolEmbeddings, DocumentRNNEmbeddings
from flair.embeddings import ELMoEmbeddings
from flair.embeddings import BytePairEmbeddings
from flair.embeddings import BertEmbeddings

#import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from typing import List, Union
from pathlib import Path

#Models definitions
from models.imagenet import mobilenetv2

#from sklearn.metrics import confusion_matrix
import seaborn as sn

import nn_models as mod
import H5Dataset as H5

#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False




def main():
    #description = '_10_11_12_flairs_fast_audebert'
    description = 'forward_bert'
    max_epochs = 70
    batch_size = 32
    num_classes = 10
    full_model = 'text'
    image_model = 'mobilenetv2'
    image_training_type = 'finetuning'
    text_model = 'stack' #rnn to include
    combined_embeddings = 'stack'
    model_path = './resources/taggers/small_tobacco/'
    learning_rate = 0.005

    """
    :param batch_size: batch size
    :param num_classes: number of classes (10)
    :param full_model: model used: 'image', 'text' or 'combined'
    :param image_model: if image model used, specify which type: 'dense' (densenet121) or 'mobilenetv2'
    :param image_training_type: type of training of the image model: 'finetuning' (whole net is trained) or 'feature_extract' (only classifier is trained)
    :param text_model: type of text model: 'cnn', 'linear', 'documentRNN'
    :param combined_embeddings: process of combining different embeddings: 'individual', 'stack', 'documentpool', 'documentRNN'
    """

    hdf5_file = './HDF5_files/hdf5_small_tobacco_audebert.hdf5'
    writer = SummaryWriter()

    if combined_embeddings == 'documentpool':
        text_model = 'linear'

    flair_embedding_forward = FlairEmbeddings('./Data/news-forward-0.4.1.pt')#2048
    fast_embedding = WordEmbeddings('en')#en-fasttext-news-300d-1M
    byte_embedding = BytePairEmbeddings(language= 'en')#Files stores in .flair/embeddings/en/...
    flair_embedding_backward = FlairEmbeddings('./Data/news-backward-0.4.1.pt')#2048
    glove_embedding = WordEmbeddings('glove')
    bert_embedding = BertEmbeddings()
    #elmo_embedding = ELMoEmbeddings(model = "original", options_file = './Data/elmo_2x2048_256_2048cnn_1xhighway_elmo_2x2048_256_2048cnn_1xhighway_options.json', weight_file = './Data/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5')

    embeddings_models_list = [flair_embedding_forward,bert_embedding]

    if combined_embeddings == 'documentpool':
        embedding_model = DocumentPoolEmbeddings([
                                                fast_embedding
                                              ]) #pooling argument: 'min', 'max' or 'mean' default
    elif combined_embeddings == 'stack':
        # Stacks n embeddings by concatenating them -> embedding_model_1 + embedding_model_1 + ... + embedding_model_n
        #embedding_model = StackedEmbeddings(embeddings=[flair_embedding_forward, flair_embedding_backward, fast_embedding])
        embedding_model = StackedEmbeddings(embeddings=embeddings_models_list)

    else:
        embedding_model = flair_embedding_forward #Change this

    if text_model == 'documentRNN':
        embedding_model = DocumentRNNEmbeddings(embeddings=embeddings_models_list, hidden_size=128)


    total_embedding_length = embedding_model.embedding_length
    print(embedding_model)
    print('Total length', total_embedding_length)

    #print('Embedding model', embedding_model)

    print('Model for training {}, {} {}  lr = {}, {}'.format(full_model, combined_embeddings, image_model , learning_rate, description))

    # Class and Segmentation
    __all__ = ['MobileNetV2', 'mobilenetv2_19']

    input_size = 384#N.Audebert et al ->384

    #Image preprocessing
    data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(input_size,scale=(0.08, 1.0),interpolation=3),
        #transforms.CenterCrop(input_size),
        #transforms.Resize(input_size),
        transforms.RandomAffine((-10,10), translate=None, scale=None, shear=None, resample=False, fillcolor=0),
        #transforms.Grayscale(num_output_channels=3),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Resize(input_size),
        #transforms.Grayscale(num_output_channels=3),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        #transforms.Resize(input_size),
        #transforms.Grayscale(num_output_channels=3),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    dataset_file = h5py.File(hdf5_file, 'r')#hdf5_small_tobacco_papers.hdf5
    target_test = dataset_file.get('train_labels')
    ## Class count
    class_sample_count = [0] * 10

    for element in target_test:
        #labels = Variable(local_batch['class'])
        labels = element
        class_sample_count[labels] += 1

    weights = 1 / torch.Tensor(class_sample_count)
    print(class_sample_count)
    # print(weights)
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)


    documents_datasets = {x: H5.H5Dataset(path=hdf5_file, data_transforms=data_transforms[x], embedding_model = embedding_model, embeddings_combination = combined_embeddings, type_model = full_model, phase = x) for x in ['train', 'val']}
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
            # print(net.conv[0])
            # net.conv = conv_1x1_bn(320, 46080)
            # print(net.conv[0])
            num_ftrs = net.classifier.in_features
            net.classifier = nn.Linear(num_ftrs, output_image_model)
            #print(net)

        elif image_model == 'dense':
             #Densenet
             use_pretrained = True
             net = models.densenet121(pretrained=use_pretrained)
             if image_training_type == 'feature_extract':
                 set_parameter_requires_grad(net, feature_extracting)
             num_ftrs = net.classifier.in_features
             net.classifier = nn.Linear(num_ftrs, output_image_model)

             #print(net)
             #output_image_model = net.classifier.out_features

        if full_model == 'combined':
            PATH_best = model_path  + 'combined/' + str(max_epochs) + str(learning_rate) + description + '.pt'
            if text_model == 'cnn':
                model = mod.CNN_Text_Image(net, total_embedding_length)
        elif full_model == 'image':
            PATH_best = model_path  + 'image/' + str(max_epochs) + str(learning_rate) + description + '.pt'
            net.classifier = nn.Linear(num_ftrs, num_classes) #Change classifier to 10 classes
            model = net

    #Text
    #Add here RNN
    else:
        PATH_best = model_path  + 'text/' + str(max_epochs) + str(learning_rate) + description + '.pt'
        if text_model == 'cnn':
            model = mod.CNN_Text(total_embedding_length)
        elif text_model == 'documentRNN':
            model = mod.TextClassifier(embedding_model)
        else:
            model = mod.Fully_Connected(total_embedding_length)

    #Move model to gpu
    model = model.to(device)

    #Printing parameters to learn in training
    # print("Params to learn:")
    # params_to_update = []
    # for name,param in model.named_parameters():
    #     if param.requires_grad == True:
    #         params_to_update.append(param)
    #         print("\t",name)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.00004)
    if full_model == 'image':
        #scheduler = StepLR(optimizer, 15, gamma=0.5, last_epoch=-1)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    else:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

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

                if text_model == 'documentRNN':
                    ocr_text, labels = local_batch['ocr'], Variable(local_batch['class'])
                    tokenized_sentences = []
                    for sentence in ocr_text:
                        sentence_tokenized = Sentence(sentence, use_tokenizer = True)
                        tokenized_sentences.append(sentence_tokenized)
                        ocr_text = tokenized_sentences
                    #print('ocr number of docs', len(ocr_text))
                    #ocr_text = Sentence(ocr_text, use_tokenizer = True)
                    #ocr_text = ocr_text.to(device)
                    labels = labels.to(device)
                else:

                    image, ocr_text, labels = Variable(local_batch['image']), Variable(local_batch['ocr']), Variable(local_batch['class'])


                    image, ocr_text, labels = image.to(device), ocr_text.to(device), labels.to(device)

                steps += 1
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

            if phase == 'train':
                train_loss = epoch_loss
            if phase == 'val':
                val_loss = epoch_loss

            if phase == 'val':
                #scheduler.step(epoch_loss) #use this for ReduceLROnPlateau scheduler
                if full_model == 'image':
                    #scheduler.step()
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step(epoch_loss)
                writer.add_scalars('Loss', {'train_loss': train_loss,
                                            'validation_loss': val_loss},epoch)

                # for i in range(10):
                #     writer.add_histogram('Fully connected weights',model.fc1.weight.data[i], i)

            actual_lr = get_lr(optimizer)

            print('[Epoch {}/{}] {} lr: {:.4f}, Loss: {:.4f} Acc: {:.4f}'.format(epoch, max_epochs,phase, actual_lr , epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    }, PATH_best)
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        PATH_final = model_path  + 'image/' + str(max_epochs) + str(learning_rate) + description + 'final.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, PATH_final)

        print()

        time_elapsed = time.time() - since
        print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        print()

        # epoch_acc = running_corrects.double() / len(dataloader_train.dataset)
        # print('[Epoch {}/{}], loss {}, Acc {}'.format(
        #                 epoch, max_epochs,running_loss/(800/batch_size), epoch_acc))

        # print('Loading best weights')
        # model.load_state_dict(best_model_wts)


    #torch.save(model.state_dict(), model_path  + full_model + str(max_epochs) + '.pt')


    h5_dataset_test = H5.H5Dataset(path=hdf5_file, data_transforms=data_transforms['test'], embedding_model = embedding_model, embeddings_combination = combined_embeddings, type_model = full_model, phase = 'test')

    dataloader_test = DataLoader(h5_dataset_test, batch_size=1, shuffle=False, num_workers=0)#=0 in inetractive job


    num_classes = 10



    print('Testing...')
    model.eval()

    correct = 0
    max_size = 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        #print('Antes de batches')
        for i, local_batch in enumerate(dataloader_test):

            if text_model == 'documentRNN':
                ocr_text = local_batch['ocr']
                ocr_text = Sentence(ocr_text, use_tokenizer = True)
                ocr_text = ocr_text.to(device)
            else:

                image, ocr_text, labels = Variable(local_batch['image']), Variable(local_batch['ocr']), Variable(local_batch['class'])
                # if max_size < ocr_text.shape[1]:
                #     max_size =  ocr_text.shape[1]
                #     print(max_size)
                #print('Antes de a gpu')


                image, ocr_text, labels = image.to(device), ocr_text.to(device), labels.to(device)

            if i%400==0:
                print(i)

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
    print('GPU available', torch.cuda.is_available())
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    #print('Type of GPU', torch.cuda.get_device_name(0))
    #cuda = torch.device('cuda')
    main()
