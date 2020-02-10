import numpy as np
import h5py
from random import shuffle
import glob
import csv
import cv2
import pandas as pd




#http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html

hdf5_path = '/media/bscuser/bsc/BigTobacco_ocrs_train_intersection_sample_300000.hdf5'
file_read = open('/media/bscuser/bsc/labels/BigTobacco_cover_train_intersection.csv', "rU")
reader = csv.reader(file_read, delimiter=',')

df = []

for element in reader:
    new_row = [element[0],element[1],element[2]]
    df.append(new_row)

file_read.close()

# columns = ['img_dir','class', 'ocr_dir']
# df = pd.DataFrame(df, columns = columns)
# df = df.sample(frac=1).reset_index(drop=True) #shuffle
#
# values = df.values
# train_sample = []
# test_sample = []
# classes = ['0','1','2','3','4','5','6','7','8','9']
# for clas in classes:
#     counter = 0
#     for row in values:
#         if row[1] == clas and counter < max_samples_class:
#             counter += 1
#             new_row = [row[0], row[1], row[2], row[3]]
#             train_sample.append(new_row)

# for row in values:
#     counter = 0
#     for element in train_sample:
#         if row[0] != element[0]:
#             counter += 1
#         if counter == len(train_sample):
#             new_row = [row[0], row[1], row[2], row[3]]
#             test_sample.append(new_row)
# import random
# random.shuffle(test_sample)
# random.shuffle(train_sample)



#Original label csv reading into list
addrs = []
labels = []
segmentation = []
ocr_dirs = []
for row in df:
      adress = row[2]
      lab = int(row[1])
      #seg = int(row[2])
      #ocr = row[3]
      addrs.append(adress)
      labels.append(lab)
      # segmentation.append(seg)
      # ocr_dirs.append(ocr)

# for row in test_sample:
#       adress = row[0]
#       lab = int(row[1])
#       seg = int(row[2])
#       ocr = row[3]
#       #label = np.array(label).astype(int)
#       addrs.append(adress)
#       labels.append(lab)
#       segmentation.append(seg)
#       ocr_dirs.append(ocr)



#Split
# train_addrs = addrs[0:int(0.6*len(addrs))]
# train_labels = labels[0:int(0.6*len(labels))]
# train_segmentations = segmentation[0:int(0.6*len(segmentation))]
train_ocrs = addrs[0:315000]
# val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
# val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
# val_segmentations = segmentation[int(0.6*len(segmentation)):int(0.8*len(segmentation))]
val_ocrs = addrs[315001:317800]
# test_addrs = addrs[int(0.8*len(addrs)):]
# test_labels = labels[int(0.8*len(ocr_dirs)):]
# test_segmentations = segmentation[int(0.8*len(segmentation)):]
# test_ocrs = ocr_dirs[int(0.8*len(ocr_dirs)):]

#Papers split 800/200/rest
# num_samples_train = 800#800
# num_samples_val = 1000#1000
# train_addrs = addrs[0:num_samples_train]
train_labels = labels[0:315000]
# train_segmentations = segmentation[0:num_samples_train]
# train_ocrs = ocr_dirs[0:num_samples_train]
# val_addrs = addrs[num_samples_train:num_samples_val]
val_labels = labels[315001:317800]
# val_segmentations = segmentation[num_samples_train:num_samples_val]
# val_ocrs = ocr_dirs[num_samples_train:num_samples_val]
# test_addrs = addrs[num_samples_val:]
# test_labels = labels[num_samples_val:]
# test_segmentations = segmentation[num_samples_val:]
# test_ocrs = ocr_dirs[num_samples_val:]

#Preprocess
data_order = 'tf'  #'tf' for Tensorflow order
dt = h5py.special_dtype(vlen=str)     # PY3

# check the order of data and chose proper data shape to save images
#train_shape = (len(addrs), img_size, img_size, 3)
#val_shape = (len(addrs), img_size, img_size, 3)
#test_shape = (len(addrs), img_size, img_size, 3)

# open a hdf5 file and create arrays
hdf5_file = h5py.File(hdf5_path, mode='w')
#
#hdf5_file.create_dataset("train_img", train_shape, np.int8)
#hdf5_file.create_dataset("val_img", val_shape, np.int8)
#hdf5_file.create_dataset("test_img", test_shape, np.int8)
#
# hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)

#Train
hdf5_file.create_dataset("train_labels", (len(train_labels),), np.int8)
hdf5_file["train_labels"][...] = train_labels
# hdf5_file.create_dataset("train_segmentations", (len(train_segmentations),), np.int8)
# hdf5_file["train_segmentations"][...] = train_segmentations
hdf5_file.create_dataset("train_ocrs", (len(train_ocrs),), dtype=dt)
#
#
# #Validation
hdf5_file.create_dataset("val_labels", (len(val_labels),), np.int8)
hdf5_file["val_labels"][...] = val_labels
# hdf5_file.create_dataset("val_segmentations", (len(val_segmentations),), np.int8)
# hdf5_file["val_segmentations"][...] = val_segmentations
hdf5_file.create_dataset("val_ocrs", (len(val_ocrs),), dtype=dt)
#
# #Test
# hdf5_file.create_dataset("test_labels", (len(labels),), np.int8)
# hdf5_file["test_labels"][...] = labels
# hdf5_file.create_dataset("test_segmentations", (len(test_segmentations),), np.int8)
# hdf5_file["test_segmentations"][...] = test_segmentations
# hdf5_file.create_dataset("test_ocrs", (len(test_ocrs),), dtype=dt)

def extract_ocrs(ocrs):
    for i in range(len(ocrs)):
        # print how many images are saved every 1000 images
        if i % 100 == 0 and i > 1:
            print('File: {}/{}'.format(i, len(ocrs)))
        # read an image and resize to (img_size, img_size)
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
# extract_ocrs(test_ocrs)
hdf5_file.close()
