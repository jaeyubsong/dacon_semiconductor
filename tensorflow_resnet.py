#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import warnings
import os
import glob
import pickle
import argparse
import math
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

# Dataset
data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
data = data.to_numpy()


def splitData(data, valid=150000, np_seed=1):
    myData = np.array(data)
    np.random.seed(np_seed)
    np.random.shuffle(myData)

    train_set = myData[valid:,:]
    valid_set = myData[:valid,:]
    return train_set, valid_set



def create_model():
    n_feature_maps = 64
    input_layer = Input((226, 1))
    # BLOCK 1

    conv_x = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_1 = tf.keras.layers.add([shortcut_y, conv_z])
    output_block_1 = tf.keras.layers.Activation('relu')(output_block_1)

    # BLOCK 2

    conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_2 = tf.keras.layers.add([shortcut_y, conv_z])
    output_block_2 = tf.keras.layers.Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = BatchNormalization()(output_block_2)

    output_block_3 = tf.keras.layers.add([shortcut_y, conv_z])
    output_block_3 = tf.keras.layers.Activation('relu')(output_block_3)


    # Final
    gap_layer = GlobalAveragePooling1D()(output_block_3)
    output_layer = Dense(units=4, activation='linear')(gap_layer)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.0005
    epochs_drop = 10.0
    lrate = min(initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop)), 0.0001)
    return lrate



folder_name = 'resnet'
checkpoint_path = folder_name + "/cp.{epoch:02d}-{val_loss:.2f}.ckpt"
cp_callback = ModelCheckpoint(checkpoint_path,
                                  monitor='val_loss',
                                  mode='min',
                                  save_weights_only=True,
                                  save_best_only=False,
                                  verbose=1)
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(checkpoint_path,
                              monitor='val_loss',
                              mode='min',
                              save_weights_only=True,
                              save_best_only=False,
                              verbose=1)
tensorboard_callback = TensorBoard(log_dir=logdir)

data_numpy = data.to_numpy()
train, valid = splitData(data, valid=150000, np_seed=1)
train_X = train[:, 4:]
train_Y = train[:, 0:4]
valid_X = valid[:, 4:]
valid_Y = valid[:, 0:4]


model = create_model()
model.compile(loss='mae', optimizer='adam', metrics=['mae'])

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                              min_lr=0.0001)
callbacks_list = [cp_callback, tensorboard_callback, reduce_lr]

# Fit model
model.fit(train_X, train_Y, epochs=100000, batch_size=64,
          validation_data=(valid_X, valid_Y), callbacks=callbacks_list)
