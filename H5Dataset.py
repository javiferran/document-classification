import h5py
from PIL import Image, ImageSequence
import cv2

from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, Sentence, DocumentPoolEmbeddings

import torchtext.data as data
import torchtext.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class H5Dataset(Dataset):

    def __init__(self, path, data_transforms, embedding_model, embeddings_combination ,type_model, phase):
        print('dataset init')
        self.model_embedding = embedding_model
        self.combination_embeddings = embeddings_combination
        self.file_path = path
        self.dataset = None
        self.data = None
        self.target = None
        self.ocr = None
        self.phase = phase
        self.type_model = type_model
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
        embeddings_matrix = torch.zeros(1910,self.model_embedding.embedding_length)


        if self.type_model != 'image':
            ocr = ocr_text #ocr_text
            # #print(ocr)
            if ocr == '':
                ocr = 'empty'

            ocr = ocr[:512]
            #print(ocr)
            sentence = Sentence(ocr, use_tokenizer = True)


            #flair_embedding_fast = FlairEmbeddings('multi-forward-fast')
            #flair_embedding_fast.embed(sentence)
            self.model_embedding.embed(sentence)

            if self.combination_embeddings == 'documentpool':
                embeddings_matrix = sentence.get_embedding() #extracts one single vector per document
            elif self.combination_embeddings == 'documentRNN':
                embeddings_matrix = ocr
            else:
                #counter = 0
                embeddings_matrix = torch.zeros(512,self.model_embedding.embedding_length)#.to(flair.device) #max number of tokens in document = 1910
                for i, token in enumerate(sentence):
                    #print(token)
                        token_embedding = token.embedding
                        token_embedding = token_embedding.unsqueeze(0)
                        embeddings_matrix[i] = token_embedding


        if self.type_model == 'image' or 'combined':
            if self.data_transforms is not None:
                try:
                    image = self.data_transforms(img)
                except:
                    print("Cannot transform image: {}")


        sample = {'image': image, 'class': doc_class, 'ocr': embeddings_matrix}#embeddings_matrix

        return sample
