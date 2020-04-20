import tensorflow as tf
import os
import glob
import shutil
import sys
import numpy as np
import math
import cv2
from tqdm.autonotebook import tqdm
from PIL import Image
import io
import base64
from keras.utils import HDF5Matrix


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def to_rgb(x):
    return x * 2

img_size = 384

sets = ['train','val', 'test']
for set in sets:
    path_in = '/media/bscuser/bsc/BigTobacco_images_' + set + '.hdf5'
    path_out = '/media/bscuser/bsc/BigTobacco_images_' + set + '.tfrecord'
    #path_out = '/gpfs/scratch/bsc31/bsc31961/BigTobacco_images_train_sheared.tfrecord'


    images = HDF5Matrix(path_in, set + '_img')
    labels = HDF5Matrix(path_in, set + '_labels')

    n_elems = len(images)

    def image_example(image, label):

        img = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        img = tf.convert_to_tensor(img, tf.uint8)
        image = tf.image.encode_png(img)
        image_shape = tf.image.decode_png(image).shape

        feature = {
          'height': _int64_feature(image_shape[0]),
          'width': _int64_feature(image_shape[1]),
          'depth': _int64_feature(image_shape[2]),
          'label': _int64_feature(label),
          'image_raw': _bytes_feature(image),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))


    with tf.io.TFRecordWriter(path_out) as writer:
        for idx in tqdm(range(n_elems)):
            array = images[idx]
            array_2 = np.vectorize(to_rgb)(array).astype(np.float32)
            example = image_example(array_2, labels[idx])
            writer.write(example.SerializeToString())
