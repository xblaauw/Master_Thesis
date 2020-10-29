#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from time import time as time
from pathlib import Path
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import ParameterGrid

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


# In[28]:


results_path = 'conv_gridsearch_results/dwt_features/'
Path(results_path).mkdir(parents=True, exist_ok=True)

sun_up, sun_down = '04:00:00', '19:45:00'

csv_path = '../pv_data/ML_input_15T_dwt.csv'
df = pd.read_csv(csv_path,
                 index_col=0,
                 parse_dates=True,
                 infer_datetime_format = True)
df = df.between_time(sun_up, sun_down)

TIME_STEPS_PER_DAY = len(df.loc['1-1-2016'])
TRAIN_TEST_SPLIT = len(df.loc['1-1-2015':'31-12-2016'])

cols = df.columns
index = df.index

df = df.values
scaler_target = MinMaxScaler()
target = scaler_target.fit_transform(df[:,:1])
scaler_features = MinMaxScaler()
features = scaler_features.fit_transform(df[:,1:])
df = np.concatenate((target, features), axis=1)

df = pd.DataFrame(df, index = index, columns = cols)


# In[29]:


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index, step):
    indices = range(i-history_size, i)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

def remove_nan(X,Y):
    x = []
    y = []
    for sample in range(X.shape[0]):
        if np.isnan(X[sample,:,:]).any() | np.isnan(Y[sample,:]).any():
            None
        else:
            x.append(X[sample,:,:])
            y.append(Y[sample,:])
    x = np.array(x)
    y = np.array(y)
    return x, y


# In[30]:


def make_model(x_train, y_train, FILTERS1, KERNEL_SIZE1, PADDING1, DILATION_RATE1, 
               ACTIVATION1, CONV2, POOL_SIZE1, CONV_POOL2, 
               DENSE_UNITS, DROPOUT, ACTIVATION2, OPTIMIZER):
    initializer = keras.initializers.TruncatedNormal(mean=0., stddev=0.05)
    bias = keras.initializers.Constant(0.1)        
   
    model = Sequential()
    model.add(Input(shape = (x_train.shape[1], x_train.shape[2])))
    
    model.add(Conv1D(filters            = FILTERS1, 
                     kernel_size        = KERNEL_SIZE1, 
                     padding            = PADDING1,
                     dilation_rate      = DILATION_RATE1,
                     activation         = ACTIVATION1,
                     kernel_initializer = initializer,
                     bias_initializer   = bias))
    if CONV2:
        model.add(Conv1D(filters            = FILTERS1, 
                         kernel_size        = KERNEL_SIZE1, 
                         padding            = PADDING1,
                         dilation_rate      = DILATION_RATE1,
                         activation         = ACTIVATION1,
                         kernel_initializer = initializer,
                         bias_initializer   = bias))
    model.add(MaxPooling1D(pool_size        = POOL_SIZE1))
    model.add(Dropout(     rate             = DROPOUT))
    
    if CONV_POOL2:
        model.add(Conv1D(filters            = FILTERS1, 
                         kernel_size        = KERNEL_SIZE1, 
                         padding            = PADDING1,
                         dilation_rate      = DILATION_RATE1,
                         activation         = ACTIVATION1,
                         kernel_initializer = initializer,
                         bias_initializer   = bias))
        if CONV2:
            model.add(Conv1D(filters            = FILTERS1, 
                             kernel_size        = KERNEL_SIZE1, 
                             padding            = PADDING1,
                             dilation_rate      = DILATION_RATE1,
                             activation         = ACTIVATION1,
                             kernel_initializer = initializer,
                             bias_initializer   = bias))
        model.add(MaxPooling1D(pool_size        = POOL_SIZE1))
        model.add(Dropout(     rate             = DROPOUT))
    
    model.add(Flatten())
    
    model.add(Dense(units               = DENSE_UNITS, 
                    activation          = ACTIVATION2,
                    kernel_initializer  = initializer,
                    bias_initializer    = bias))

    model.add(Dropout(rate              = DROPOUT))

    model.add(Dense(units               = y_train.shape[1], 
                    activation          = ACTIVATION2,
                    kernel_initializer  = initializer,
                    bias_initializer    = bias))

    model.compile(loss = 'mse', optimizer = OPTIMIZER)
    return model


# In[31]:


def grid_search(counter, df, col_selection, HISTORY, 
                FILTERS1, KERNEL_SIZE1, PADDING1, DILATION_RATE1, ACTIVATION1, 
                CONV2, POOL_SIZE1, CONV_POOL2, DENSE_UNITS, DROPOUT, ACTIVATION2, OPTIMIZER,
                TIME_STEPS_PER_DAY = TIME_STEPS_PER_DAY, 
                TRAIN_TEST_SPLIT = TRAIN_TEST_SPLIT):
    df = df[['151'] + col_selection].values
    
    TARGET_COL = df[:,0]
    HISTORY_SIZE = TIME_STEPS_PER_DAY * HISTORY
    TARGET_SIZE = TIME_STEPS_PER_DAY
    STEP = TIME_STEPS_PER_DAY

    x_train, y_train = multivariate_data(df, TARGET_COL, 0, TRAIN_TEST_SPLIT, HISTORY_SIZE, TARGET_SIZE, STEP)
    x_train, y_train = remove_nan(x_train, y_train)
    train_shape = x_train.shape
    
    x_test, y_test = multivariate_data(df, TARGET_COL, TRAIN_TEST_SPLIT, None, HISTORY_SIZE, TARGET_SIZE, STEP)
    x_test, y_test = remove_nan(x_test, y_test)
    test_shape = x_test.shape

    model = make_model(x_train = x_train,
                       y_train = y_train,
                       FILTERS1       = FILTERS1, 
                       KERNEL_SIZE1   = KERNEL_SIZE1,
                       PADDING1       = PADDING1,
                       DILATION_RATE1 = DILATION_RATE1, 
                       ACTIVATION1    = ACTIVATION1, 
                       CONV2          = CONV2,
                       POOL_SIZE1     = POOL_SIZE1, 
                       CONV_POOL2     = CONV_POOL2, 
                       DENSE_UNITS    = DENSE_UNITS, 
                       DROPOUT        = DROPOUT, 
                       ACTIVATION2    = ACTIVATION2, 
                       OPTIMIZER      = OPTIMIZER)

    checkpoint_path = results_path + 'checkpoints/'
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath       = checkpoint_path + str(counter) + '_cp.h5',
                                                 save_best_only = True,
                                                 monitor        = 'val_loss')

    history = model.fit(x_train, y_train,
                        epochs          = 100, 
                        batch_size      = 24,
                        validation_data = (x_test, y_test),
                        callbacks       = [checkpoint],
                        verbose         = 0,
                        shuffle         = False)

    model = load_model(checkpoint_path + str(counter) + '_cp.h5')

    conv_predictions = model.predict(x_test)
    conv_predictions[conv_predictions<0] = 0
    conv_mse = mean_squared_error(y_test, conv_predictions, squared=True)
    mae = mean_absolute_error(y_test, conv_predictions)

    return conv_mse, mae, train_shape, test_shape


# In[33]:


dt_only = ['year_2015', 'year_2016', 'year_2017', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']
irradiance = ['Q_GLOB_10', 'QN_GLOB_10', 'QX_GLOB_10', 'SQ_10']
wind = ['FF_10M_10', 'DD_10_sin', 'DD_10_cos', 'DDN_10_sin', 'DDN_10_cos', 'DD_STD_10_sin', 'DD_STD_10_cos', 'DDX_10_sin', 'DDX_10_cos', 'FF_SENSOR_10', 'FF_10M_STD_10', 'FX_SENSOR_10']
dwt1 = list(df[df.columns[df.columns.str.startswith('dwt1', na=False)]].columns)
dwt2 = list(df[df.columns[df.columns.str.startswith('dwt2', na=False)]].columns)
dwt3 = list(df[df.columns[df.columns.str.startswith('dwt3', na=False)]].columns)
dwt4 = list(df[df.columns[df.columns.str.startswith('dwt4', na=False)]].columns)


parameters = {'col_selection'  : [dt_only + wind + irradiance,
                                  dt_only + wind + irradiance + dwt1,
                                  dt_only + wind + irradiance + dwt2,
                                  dt_only + wind + irradiance + dwt3,
                                  dt_only + wind + irradiance + dwt4],
              'HISTORY'        : [1,3,5],
              'FILTERS1'       : [16, 32, 64],
              'KERNEL_SIZE1'   : [2,3,5],
              'PADDING1'       : ['same'],
              'DILATION_RATE1' : [1, 3],
              'ACTIVATION1'    : ['tanh'],
              'CONV2'          : [True, False],
              'POOL_SIZE1'     : [2, 3, 5],
              'CONV_POOL2'     : [True, False],
              'DROPOUT'        : [0.2, 0.4], 
              'DENSE_UNITS'    : [100],
              'ACTIVATION2'    : ['relu'],
              'OPTIMIZER'      : ['Adam']
             }

grid_results = pd.DataFrame(columns = ['counter', 'col_selection', 'HISTORY',  'FILTERS1', 'KERNEL_SIZE1', 'PADDING1', 'DILATION_RATE1', 'ACTIVATION1', 'CONV2', 
                             'POOL_SIZE1', 'CONV_POOL2', 'DROPOUT', 'DENSE_UNITS', 'ACTIVATION2', 'OPTIMIZER',
                             'mse', 'mae', 'train_samples', 'test_samples', 'train_time'])

param_grid = ParameterGrid(parameters)
output_path = results_path + 'gridsearch_results.csv'

counter = 0
for dict_ in param_grid:
    start = time()
    conv_mse, mae, train_shape, test_shape = grid_search(counter, df, dict_['col_selection'], dict_['HISTORY'],  
                                                                            dict_['FILTERS1'], 
                                                                            dict_['KERNEL_SIZE1'],
                                                                            dict_['PADDING1'], 
                                                                            dict_['DILATION_RATE1'], 
                                                                            dict_['ACTIVATION1'], 
                                                                            dict_['CONV2'], 
                                                                            dict_['POOL_SIZE1'], 
                                                                            dict_['CONV_POOL2'], 
                                                                            dict_['DENSE_UNITS'],
                                                                            dict_['DROPOUT'], 
                                                                            dict_['ACTIVATION2'], 
                                                                            dict_['OPTIMIZER'])
    train_time = round(time() - start, 2)

    grid_results = grid_results.append({'counter'       : counter, 
                                        'col_selection' : dict_['col_selection'], 
                                        'HISTORY'       : dict_['HISTORY'], 
                                        'FILTERS1'      : dict_['FILTERS1'], 
                                        'KERNEL_SIZE1'  : dict_['KERNEL_SIZE1'], 
                                        'PADDING1'      : dict_['PADDING1'], 
                                        'DILATION_RATE1': dict_['DILATION_RATE1'], 
                                        'ACTIVATION1'   : dict_['ACTIVATION1'], 
                                        'CONV2'         : dict_['CONV2'], 
                                        'POOL_SIZE1'    : dict_['POOL_SIZE1'], 
                                        'CONV_POOL2'    : dict_['CONV_POOL2'], 
                                        'DENSE_UNITS'   : dict_['DENSE_UNITS'], 
                                        'DROPOUT'       : dict_['DROPOUT'], 
                                        'ACTIVATION2'   : dict_['ACTIVATION2'], 
                                        'OPTIMIZER'     : dict_['OPTIMIZER'], 
                                        'mse'           : conv_mse, 
                                        'mae'           : mae, 
                                        'train_samples' : train_shape[0], 
                                        'test_samples'  : test_shape[0], 
                                        'train_time'    : str(train_time)
                                       }, ignore_index = True
                                      )
    grid_results.to_csv(output_path, sep=',')
    
    print('Done!: ' + str(counter) + ' train_time: ' + str(train_time))
    counter += 1

