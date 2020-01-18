#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import warnings
import os
import glob
import pickle
import argparse
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, LeakyReLU, ELU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument("-es", "--earlystop", required=False, type=int, default=0, help="Set to have earlystop")
parser.add_argument("-gpu", "--gpu", required=False, default="", help="Set to have earlystop")
args = parser.parse_args()
if args.gpu == "":
    gpu_nums = 0
    os.environ["CUDA_VISIBLE_DEVICES"]=""
else:
    gpu_nums = len(args.gpu.split(','))
    print("Using %d gpus: %s" % (gpu_nums,args.gpu))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

tf.config.list_physical_devices('GPU')
print("Finished parsing")
print(args.earlystop)
# Train
data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

best_val_loss = 99999999
best_epoch = -1
cur_hyperparam = ""
pickle_path = ""

def splitData(data, valid=200000, np_seed=1):
    myData = np.array(data)
    np.random.seed(np_seed)
    np.random.shuffle(myData)

    train_set = myData[valid:,:]
    valid_set = myData[:valid,:]
    return train_set, valid_set


def build_model(hp):
    model = Sequential()

    model.add(Dense(units=hp.Int("input_units", min_value=256, max_value=512, step=32), input_dim=226, use_bias=False))
    model.add(BatchNormalization())
    model.add(ELU(alpha=1.0))

    for i in range(hp.Int("n_layers", 4, 12)):
        model.add(
            Dense(units=hp.Int(f"dense_{i}_units", min_value=256, max_value=512, step=32), use_bias=False))
        model.add(BatchNormalization())
        model.add(ELU(alpha=1.0))

    model.add(Dense(units=4))
    model.add(Activation('linear'))

    model.compile(loss='mae',
                  optimizer=Adam(
                      hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
                  ),
                  metrics=['mae'])
    return model


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global best_val_loss
        global best_epoch
        global cur_hyperparam
        global pickle_path
        if logs['val_loss'] < best_val_loss:
            best_val_loss = logs['val_loss']
            best_epoch = epoch + 1
            # Save pickle file
            if os.path.exists(pickle_path):
                os.remove(pickle_path)
            else:
                print("Can not delete the pickle")
            fileObject = open(pickle_path, 'wb')
            pickle.dump([best_epoch, best_val_loss], fileObject)
            fileObject.close()
        print(logs)
        print("Best epoch: %d, best val_loss: %.2f" % (best_epoch, best_val_loss))


data_numpy = data.to_numpy()
train, valid = splitData(data, valid=150000, np_seed=1)
train_X = train[:, 4:]
train_Y = train[:, 0:4]
valid_X = valid[:, 4:]
valid_Y = valid[:, 0:4]

test_numpy = test.to_numpy()
test_X = test_numpy[:,1:]

LOG_DIR = 'autotuner_results'
tuner = RandomSearch(
    build_model,
    objective = "val_loss",
    max_trials=1,
    executions_per_trial=1,
    directory=LOG_DIR
)

tuner.search(x=train_X,
             y=train_Y,
             epochs=100000,
             batch_size=1000,
             callbacks=[EarlyStopping('val_loss', patience=30)],
             validation_data=(valid_X, valid_Y))


print("Done")
