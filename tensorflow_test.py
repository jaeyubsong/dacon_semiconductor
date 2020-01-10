#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import warnings
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np


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
            model.add(Dense(units=int(model_params[i]), activation=model_params[i+1], input_dim=226))
        elif len(model_params) == i+1:
            model.add(Dense(units=int(model_params[i])))
        else:
            model.add(Dense(units=int(model_params[i]), activation=model_params[i+1]))
        i += 2
    return model


warnings.filterwarnings('ignore')

submissionPath = "submission.csv"
# As file at filePath is deleted now, so we should check if file exists or not not before deleting them
if os.path.exists(submissionPath):
    os.remove(submissionPath)
else:
    print("Can not delete the file as it doesn't exists")

# Test
data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

data_numpy = data.to_numpy()
test_numpy = test.to_numpy()
data_X = data_numpy[:, 4:]
data_Y = data_numpy[:, 0:4]
test_X = test_numpy[:,1:]

checkpoint_path = "320_relu_320_relu_320_relu_320_relu_320_relu_320_relu_320_relu_4/cp.380-2.45.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model = createModel("320_relu_320_relu_320_relu_320_relu_320_relu_320_relu_320_relu_4")
model.load_weights(checkpoint_path)
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
model.evaluate(data_X, data_Y)

test_Y = model.predict(test_X)


result = pd.DataFrame(test_Y, columns=['layer_1', 'layer_2', 'layer_3', 'layer_4'])
id_arr = [i for i in range(test_Y.shape[0])]
result['id'] = id_arr 
result = result[['id','layer_1', 'layer_2', 'layer_3', 'layer_4']]
print(result)
export_csv = result.to_csv (submissionPath, index = None, header=True) #Don't forget to add '.csv' at the end of the path


# In[ ]:




