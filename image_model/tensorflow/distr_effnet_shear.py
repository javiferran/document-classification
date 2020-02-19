import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model, Model
import os
import sys
import argparse
import random
import numpy as np
from tensorflow.keras.layers import Dense, Input
from efficientnet.tfkeras import EfficientNetB0, EfficientNetB2, EfficientNetB4
from math import ceil
from time import time
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--img_size', type=int, default=384)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--image_model', type=int, default=0)

args = parser.parse_args()

print("----- ARGUMENTS: ------")
print("EPOCHS:", str(args.epochs))
print("IMG_SIZE:", str(args.img_size))
print("BATCH_SIZE", str(args.batch_size))
print("OPTIMIZER", str(args.optimizer))

IMG_SIZE = args.img_size

# efficientnet activation function
def swish(x):
    return K.sigmoid(x) * x

# takes tf_record dataset, decodes it and applies transformations
@tf.function
def read_tfrecord(serialized_example):
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    input_2 = tf.image.decode_png(example['image_raw'], channels=3, dtype=tf.dtypes.uint8)
    input_2 = tf.image.resize(input_2, [IMG_SIZE, IMG_SIZE])
    images  = tf.expand_dims(input_2, 0)
    # random shear
    shear   = random.choice([-1, 1]) * 0.05
    images_t    = tfa.image.transform(images, [1., shear, 0., 0., 1., 0., 0., 0.])   # X direction
    images_t    = tfa.image.transform(images_t, [1., 0., 0., shear, 1., 0., 0., 0.]) # Y direction
    final_image = tf.squeeze(images_t)

    final_image = input_2
    return final_image, example['label']

# parameters
epochs        = args.epochs
image_model   = args.image_model
train_size    = 320000
val_size      = 40000
test_size     = 40000
num_classes   = 16
max_lr        = 0.05
input_size    = (IMG_SIZE, IMG_SIZE, 3)
opt_name      = args.optimizer.lower()
model_path    = str('saved_model_' + opt_name + '/')

strategy = tf.distribute.MirroredStrategy()
N_GPUS   = strategy.num_replicas_in_sync
print ('Number of devices: {}'.format(N_GPUS))
BUFFER_SIZE = train_size
BATCH_SIZE_PER_REPLICA = args.batch_size
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * N_GPUS

# number of steps definition
train_steps = ceil(train_size/GLOBAL_BATCH_SIZE)
val_steps   = ceil(val_size/GLOBAL_BATCH_SIZE)
test_steps  = ceil(test_size/GLOBAL_BATCH_SIZE)

# load tfrecord
train_tfrecord = tf.data.TFRecordDataset(filenames = ['/gpfs/scratch/bsc31/bsc31961/BigTobacco_images_train_full.tfrecord'])
val_tfrecord   = tf.data.TFRecordDataset(filenames = ['/gpfs/scratch/bsc31/bsc31961/BigTobacco_images_val.tfrecord'])
test_tfrecord  = tf.data.TFRecordDataset(filenames = ['/gpfs/scratch/bsc31/bsc31961/BigTobacco_images_test.tfrecord'])

# apply read_tfrecord function to every image
train_dataset  = train_tfrecord.map(read_tfrecord).batch(GLOBAL_BATCH_SIZE).repeat()
val_dataset    = val_tfrecord.map(read_tfrecord).batch(GLOBAL_BATCH_SIZE).repeat()
test_dataset   = test_tfrecord.map(read_tfrecord).batch(GLOBAL_BATCH_SIZE).repeat()

with strategy.scope():

    def create_efficientNet():
        if image_model == 4:
            baseModel = EfficientNetB4(weights='imagenet', include_top=False, input_shape=input_size)
        elif image_model == 2:
            baseModel = EfficientNetB2(weights='imagenet', include_top=False, input_shape=input_size)
        elif image_model == 0:
            baseModel = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_size)
        # reconstruction of last EfficientNet layers
        probs     = baseModel.layers.pop()
        top_conv  = probs.input
        headModel = layers.Activation(swish, name='top_activation')(top_conv)
        headModel = layers.GlobalAveragePooling2D(name='avg_pool')(headModel)
        headModel = layers.Dropout(0.2, name='top_dropout')(headModel)
        headModel = layers.Dense(num_classes, activation = 'softmax')(headModel)
        model     = Model(inputs=baseModel.input, outputs=headModel)
        return model

        model = create_efficientNet()

    # optimizer definition
    if opt_name == 'sgd':
        print("optimizer: SGD")
        opt = optimizers.SGD(lr=max_lr, decay=0.00004, momentum=0.9)
    else:
        print("optimizer: Adam")
        # default Adam parameters: (learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        opt = optimizers.Adam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,
                    metrics = ['accuracy'])


def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5


# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_begin(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr.numpy()))

# Callback for STLR
class MyLearningRateScheduler(tf.keras.callbacks.Callback):

    #def on_train_batch_begin(self, batch, logs=None):
    def on_epoch_begin(self, epoch, logs=None):
        cut_frac  = 0.1           # 0.1 => lr increases till epoch 2, then decreases
        cut       = max(int(epochs * cut_frac), 1)
        max_lr    = 0.05          # max_lr = 0.05
        ratio     = 31            # lr = [0.0016, 0.05]
        min_lr    = max_lr/ratio  # min_lr = 0.0016
        lr_change = max_lr-min_lr # difference between max_lr & min_lr
        current_epoch = epoch     # current epoch
        jump_up   = lr_change/cut
        jump_down = lr_change/(epochs-cut)

        if current_epoch < cut:
            lr = min_lr + current_epoch * jump_up
        elif current_epoch == cut:
            lr = max_lr
        else:
            lr = max_lr - (current_epoch - cut) * jump_down

        K.set_value(self.model.optimizer.lr, lr)

    #def on_train_batch_end(self, batch, logs=None):
    #    logs = logs or {}
    #    logs['lr'] = K.get_value(self.model.optimizer.lr)

#callback_save_model = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
callbacks = []

# callbacks depending on optimizer
if opt_name == 'sgd':
    print("LOADING CALLBACKS FOR SGD")
    callbacks = [
        MyLearningRateScheduler(),
        PrintLR()
    ]
elif opt_name == 'adam':
    print("LOADING CALLBACKS FOR ADAM")
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(decay),
        PrintLR()
    ]

#callbacks.append(callback_save_model)

# TRAIN MODEL

time_start = time()
model.fit(train_dataset, epochs=epochs, steps_per_epoch=train_steps, callbacks=callbacks, validation_data = val_dataset, validation_steps=val_steps)
#model.fit_generator(train_iterator, epochs=epochs, callbacks=callbacks, validation_data=val_iterator)
time_end = time()
print("Training time: " + str(time_end - time_start))

# SAVE TRAINED MODEL

try:
    model.save("model_shear.h5")
    print("Model saved to: model_shear.h5")
except Exception:
    pass

# TEST MODEL

print("Testing model...")
test_loss, test_acc = model.evaluate(test_dataset, steps=test_steps)
#test_loss, test_acc = model.evaluate_generator(test_iterator)
print('Test loss: {}, Test Accuracy: {}'.format(test_loss, test_acc))
