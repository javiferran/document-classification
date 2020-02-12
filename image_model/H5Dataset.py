import h5py
from PIL import Image, ImageSequence
import cv2

import torch
import torchtext.data as data
import torchtext.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np


class H5Dataset(Dataset):

    def __init__(self, path, data_transforms, phase):
        print('dataset init', phase)
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
            elif self.phase == 'val':
                self.dataset = h5py.File(self.file_path, 'r')
                self.data = self.dataset.get('val_img')
                self.target = self.dataset.get('val_labels')
            elif self.phase == 'test':
                self.dataset = h5py.File(self.file_path, 'r')
                self.data = self.dataset.get('test_img')
                self.target = self.dataset.get('test_labels')

        img = self.data[idx,:,:,:],
        img = Image.fromarray(img[0].astype('uint8'), 'RGB')
        #doc_class = torch.from_numpy(self.target[idx,:,:,:]).float()
        doc_class = self.target[idx]
        doc_class = doc_class.astype(np.uint8)
        doc_class = torch.tensor(doc_class)

        if self.data_transforms is not None:
            try:
                image = self.data_transforms(img)
            except:
                print("Cannot transform image: {}")


        sample = {'image': image, 'class': doc_class}

        return sample
