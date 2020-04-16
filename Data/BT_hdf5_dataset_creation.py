import numpy as np
import h5py
from random import shuffle
import glob
import csv
import cv2
import pandas as pd

# Creation of random splits of SmallTobacco .hdf5 files ensuring max_samples_class
# samples per class in training set

sets = ['train','val', 'test']
save_path = '/media/bscuser/bsc/'
for set in sets:
    hdf5_path = save_path + 'BigTobacco_images_' + str(set) + '.hdf5'
    file_read = open('./labels/' + str(set) + '.txt', "rU")
    reader = csv.reader(file_read, delimiter=' ')

    partition = []

    for element in reader:
        new_row = [element[0],element[1]]
        partition.append(new_row)

    file_read.close()

    print(len(partition))

    #Original label csv reading into list
    addrs_list = []
    labels = []
    for row in partition:
          address = row[0]
          lab = int(row[1])

          addrs_list.append(address)
          labels.append(lab)

    #Split
    train_addrs = addrs_list[0:len(addrs_list)]
    train_labels = labels[0:len(labels)]


    img_size = 384

    #Preprocess
    data_order = 'tf'  #'tf' for Tensorflow order
    dt = h5py.special_dtype(vlen=str)     # PY3

    # check the order of data and chose proper data shape to save images
    train_shape = (len(train_addrs), img_size, img_size, 3)

    # open a hdf5 file and create arrays
    hdf5_file = h5py.File(hdf5_path, mode='w')

    hdf5_file.create_dataset(str(set) + "_img", train_shape, np.int8)


    #Train
    hdf5_file.create_dataset(str(set) + "_labels", (len(train_labels),), np.int8)
    hdf5_file[str(set) + "_labels"][...] = train_labels


    # loop over train addresses
    for i in range(len(train_addrs)):
        if i % 100 == 0 and i > 1:
            print(str(set) + 'data: {}/{}'.format(i, len(train_addrs)))

        addr = '/media/bscuser/bsc/Tobacco_extracted/' + train_addrs[i]
        #addr = addr.replace('.tif', '.png', 1)
        img = cv2.imread(addr)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # save the image
        hdf5_file[str(set) + "_img"][i, ...] = img[None]

        # loop over validation addresses

    # close the hdf5 file
    hdf5_file.close()
