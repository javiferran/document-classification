import numpy as np
import h5py
from random import shuffle
import glob
import csv
import cv2
import pandas as pd



#http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html

hdf5_path = './HDF5_files/hdf5_small_tobacco_papers.hdf5'
file_read = open('./Data/Small_Tobacco_cover.csv', "rU")
reader = csv.reader(file_read, delimiter=',')

df = []

for element in reader:
    new_row = [element[0],element[1], element[2],element[3]]
    df.append(new_row)

file_read.close()

columns = ['img_dir','class', 'label', 'ocr_dir']
df = pd.DataFrame(df, columns = columns)
df = df.sample(frac=1).reset_index(drop=True) #shuffle

values = df.values
train_sample = []
test_sample = []
classes = ['0','1','2','3','4','5','6','7','8','9']
for clas in classes:
    counter = 0
    for row in values:
        if row[1] == clas and counter < 100:
            counter += 1
            new_row = [row[0], row[1], row[2], row[3]]
            train_sample.append(new_row)

for row in values:
    counter = 0
    for element in train_sample:
        if row[0] != element[0]:
            counter += 1
        if counter == len(train_sample):
            new_row = [row[0], row[1], row[2], row[3]]
            test_sample.append(new_row)
import random
random.shuffle(test_sample)
random.shuffle(train_sample)



#Original label csv reading into list
addrs = []
labels = []
segmentation = []
ocr_dirs = []
for row in train_sample:
      adress = row[0]
      lab = int(row[1])
      seg = int(row[2])
      ocr = row[3]
      #label = np.array(label).astype(int)
      addrs.append(adress)
      labels.append(lab)
      segmentation.append(seg)
      ocr_dirs.append(ocr)

for row in test_sample:
      adress = row[0]
      lab = int(row[1])
      seg = int(row[2])
      ocr = row[3]
      #label = np.array(label).astype(int)
      addrs.append(adress)
      labels.append(lab)
      segmentation.append(seg)
      ocr_dirs.append(ocr)



#Split
# train_addrs = addrs[0:int(0.6*len(addrs))]
# train_labels = labels[0:int(0.6*len(labels))]
# train_segmentations = segmentation[0:int(0.6*len(segmentation))]
# train_ocrs = ocr_dirs[0:int(0.6*len(ocr_dirs))]
# val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
# val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
# val_segmentations = segmentation[int(0.6*len(segmentation)):int(0.8*len(segmentation))]
# val_ocrs = ocr_dirs[int(0.6*len(ocr_dirs)):int(0.8*len(ocr_dirs))]
# test_addrs = addrs[int(0.8*len(addrs)):]
# test_labels = labels[int(0.8*len(ocr_dirs)):]
# test_segmentations = segmentation[int(0.8*len(segmentation)):]
# test_ocrs = ocr_dirs[int(0.8*len(ocr_dirs)):]

#Papers split 800/200/rest
train_addrs = addrs[0:800]
train_labels = labels[0:800]
train_segmentations = segmentation[0:800]
train_ocrs = ocr_dirs[0:800]
val_addrs = addrs[800:1000]
val_labels = labels[800:1000]
val_segmentations = segmentation[800:1000]
val_ocrs = ocr_dirs[800:1000]
test_addrs = addrs[1000:]
test_labels = labels[1000:]
test_segmentations = segmentation[1000:]
test_ocrs = ocr_dirs[1000:]


#Preprocess
data_order = 'tf'  #'tf' for Tensorflow order
dt = h5py.special_dtype(vlen=str)     # PY3

# check the order of data and chose proper data shape to save images
train_shape = (len(train_addrs), 224, 224, 3)
val_shape = (len(val_addrs), 224, 224, 3)
test_shape = (len(test_addrs), 224, 224, 3)

# open a hdf5 file and create arrays
hdf5_file = h5py.File(hdf5_path, mode='w')

hdf5_file.create_dataset("train_img", train_shape, np.int8)
hdf5_file.create_dataset("val_img", val_shape, np.int8)
hdf5_file.create_dataset("test_img", test_shape, np.int8)

hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)

#Train
hdf5_file.create_dataset("train_labels", (len(train_labels),), np.int8)
hdf5_file["train_labels"][...] = train_labels
hdf5_file.create_dataset("train_segmentations", (len(train_segmentations),), np.int8)
hdf5_file["train_segmentations"][...] = train_segmentations
hdf5_file.create_dataset("train_ocrs", (len(train_ocrs),), dtype=dt)


#Validation
hdf5_file.create_dataset("val_labels", (len(val_labels),), np.int8)
hdf5_file["val_labels"][...] = val_labels
hdf5_file.create_dataset("val_segmentations", (len(val_segmentations),), np.int8)
hdf5_file["val_segmentations"][...] = val_segmentations
hdf5_file.create_dataset("val_ocrs", (len(val_ocrs),), dtype=dt)

#Test
hdf5_file.create_dataset("test_labels", (len(test_labels),), np.int8)
hdf5_file["test_labels"][...] = test_labels
hdf5_file.create_dataset("test_segmentations", (len(test_segmentations),), np.int8)
hdf5_file["test_segmentations"][...] = test_segmentations
hdf5_file.create_dataset("test_ocrs", (len(test_ocrs),), dtype=dt)

def extract_ocrs(ocrs):
    for i in range(len(ocrs)):
        # print how many images are saved every 1000 images
        if i % 100 == 0 and i > 1:
            print('File: {}/{}'.format(i, len(ocrs)))
        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB

        addr = ocrs[i]
        #print(addr)
        f = open(addr, "r")
        text = f.read()
        #print(text)

        if ocrs == train_ocrs:
            hdf5_file["train_ocrs"][i,...] = text
        elif ocrs ==val_ocrs:
            hdf5_file["val_ocrs"][i,...] = text
        else:
            hdf5_file["test_ocrs"][i,...] = text

        #mean += img / float(len(train_labels))


extract_ocrs(train_ocrs)
extract_ocrs(val_ocrs)
extract_ocrs(test_ocrs)




# a numpy array to save the mean of the images
mean = np.zeros(train_shape[1:], np.float32)
# loop over train addresses
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if i % 100 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, len(train_addrs)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB

    addr = train_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # # if the data order is Theano, axis orders should change
    # if data_order == 'th':
    #     img = np.rollaxis(img, 2)
    # save the image and calculate the mean so far
    hdf5_file["train_img"][i, ...] = img[None]
    mean += img / float(len(train_labels))


    # loop over validation addresses
for i in range(len(val_addrs)):
    # print how many images are saved every 1000 images
    if i % 100 == 0 and i > 1:
        print('Validation data: {}/{}'.format(i, len(val_addrs)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = val_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # # if the data order is Theano, axis orders should change
    # if data_order == 'th':
    #     img = np.rollaxis(img, 2)
    # save the image
    hdf5_file["val_img"][i, ...] = img[None]
# loop over test addresses
for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images
    if i % 100 == 0 and i > 1:
        print('Test data: {}/{}'.format(i, len(test_addrs)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = test_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # # if the data order is Theano, axis orders should change
    # if data_order == 'th':
    #     img = np.rollaxis(img, 2)
    # save the image
    hdf5_file["test_img"][i, ...] = img[None]
# save the mean and close the hdf5 file
hdf5_file["train_mean"][...] = mean
hdf5_file.close()
