#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, LeakyReLU, ELU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
warnings.filterwarnings('ignore')

# Train
data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

best_val_loss = 99999999
best_epoch = -1

def splitData(data, valid=200000, np_seed=1):
    myData = np.array(data)
    np.random.seed(np_seed)
    np.random.shuffle(myData)

    train_set = myData[valid:,:]
    valid_set = myData[:valid,:]
    return train_set, valid_set


def createModel(model_params):
    model = Sequential()
    if ',' in model_params:
        model_params = model_params.split(',')
    elif '_' in model_params:
        model_params = model_params.split('_')
    else:
        exit(-1)
    i = 0
    while i < len(model_params):
        if i == 0:
            model.add(Dense(units=int(model_params[i]), input_dim=226, use_bias=False))
            model.add(BatchNormalization())
            if (model_params[i+1] == "leakyrelu"):
                model.add(LeakyReLU(alpha=0.1))
            elif (model_params[i+1] == "elu"):
                model.add(ELU(alpha=1.0))
            else:
                model.add(Activation(model_params[i+1]))
        elif len(model_params) == i+1:
            model.add(Dense(units=int(model_params[i])))
        elif len(model_params) == i+2:
            model.add(Dense(units=int(model_params[i])))
            if (model_params[i+1] == "leakyrelu"):
                model.add(LeakyReLU(alpha=0.1))
            elif (model_params[i+1] == "elu"):
                model.add(ELU(alpha=1.0))
            else:
                model.add(Activation(model_params[i+1]))
        else:
            model.add(Dense(units=int(model_params[i]), use_bias=False))
            model.add(BatchNormalization())
            if (model_params[i+1] == "leakyrelu"):
                model.add(LeakyReLU(alpha=0.1))
            elif (model_params[i+1] == "elu"):
                model.add(ELU(alpha=1.0))
            else:
                model.add(Activation(model_params[i+1]))
        i += 2
    return model


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global best_val_loss
        global best_epoch
        if logs['val_loss'] < best_val_loss:
            best_val_loss = logs['val_loss']
            best_epoch = epoch + 1
        print(logs)
        print("Best epoch: %d, best val_loss: %.2f" % (best_epoch, best_val_loss))


data_numpy = data.to_numpy()
train, valid = splitData(data, valid=200000, np_seed=1)
train_X = train[:, 4:]
train_Y = train[:, 0:4]
valid_X = valid[:, 4:]
valid_Y = valid[:, 0:4]

test_numpy = test.to_numpy()
test_X = test_numpy[:,1:]


for k in range(1000):
    hyperparams_result = open('hyperparams_result.txt', 'a+')
    hyperparams = open('hyperparams.txt', 'r')
    for j in range(k):
        hyperparams.readline()
    cur_hyperparam = hyperparams.readline().strip()
    hyperparams.close()
    if cur_hyperparam == "":
        break
    best_val_loss = 99999999
    best_epoch = -1
    folder_name = '_'.join(cur_hyperparam.split(','))
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = folder_name + "/cp.{epoch:02d}-{val_loss:.2f}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = ModelCheckpoint(checkpoint_path,
                                  monitor='val_loss',
                                  mode='min',
                                  save_weights_only=True,
                                  save_best_only=False,
                                  verbose=1)
    es_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    bestModel = LossAndErrorPrintingCallback()
    tensorboard_callback = TensorBoard(log_dir=logdir)

    model = createModel(cur_hyperparam)
    model.save_weights(checkpoint_path.format(epoch=0, val_loss=0))

    # Compile model
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])

    # Fit model
    model.fit(train_X, train_Y, epochs=1000000, batch_size=100,
              validation_data = (valid_X, valid_Y),
              callbacks = [cp_callback, bestModel, es_callback, tensorboard_callback])
    print("Best epoch: %d, best val_loss: %.2f" % (best_epoch, best_val_loss))
    hyperparams_result.write(cur_hyperparam)
    hyperparams_result.write("Best epoch: %d, best val_loss: %.2f\n\n" % (best_epoch, best_val_loss))
    hyperparams_result.close()

print("Done")
