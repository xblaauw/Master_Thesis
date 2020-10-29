#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
from sklearn.model_selection import ParameterGrid

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


# In[4]:
results_path = 'lstm/learning_rate/'
Path(results_path).mkdir(parents=True, exist_ok=True)

sun_up, sun_down = '04:00:00', '19:45:00'

csv_path = '../pv_data/ML_input_15T.csv'
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

# In[5]:


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


# In[6]:


def make_model(x_train, y_train, 
               FILTERS1, KERNEL_SIZE1, PADDING1, DILATION_RATE1, 
               CONV_ACTIVATION, CONV2, POOL_SIZE1, CONV_POOL2, 
               OPEN_END, DEPTH, LSTM1, LSTM_ACTIVATION,
               DENSE_UNITS, DROPOUT, ACTIVATION2, 
               OPTIMIZER):
    initializer = keras.initializers.TruncatedNormal(mean=0., stddev=0.05)
    bias = keras.initializers.Constant(0.1)        
    model = Sequential()
    model.add(Input(shape = (x_train.shape[1], x_train.shape[2])))
    
    if OPEN_END:
        if DEPTH == 1:
            model.add(LSTM(units = LSTM1, return_sequences = True, bias_initializer = keras.initializers.Constant(value=0.1)))
        if DEPTH == 2:
            model.add(LSTM(units = LSTM1, return_sequences = True, bias_initializer = keras.initializers.Constant(value=0.1)))
            model.add(LSTM(units = LSTM1, return_sequences = True, bias_initializer = keras.initializers.Constant(value=0.1)))
        if DEPTH == 3:
            model.add(LSTM(units = LSTM1, return_sequences = True, bias_initializer = keras.initializers.Constant(value=0.1)))
            model.add(LSTM(units = LSTM1, return_sequences = True, bias_initializer = keras.initializers.Constant(value=0.1)))
            model.add(LSTM(units = LSTM1, return_sequences = True, bias_initializer = keras.initializers.Constant(value=0.1)))
        model.add(Flatten())
    else:
        if DEPTH == 1:
            model.add(LSTM(units = LSTM1, return_sequences = False, bias_initializer = keras.initializers.Constant(value=0.1)))
        if DEPTH == 2:
            model.add(LSTM(units = LSTM1, return_sequences = True, bias_initializer = keras.initializers.Constant(value=0.1)))
            model.add(LSTM(units = LSTM1, return_sequences = False, bias_initializer = keras.initializers.Constant(value=0.1)))
        if DEPTH == 3:
            model.add(LSTM(units = LSTM1, return_sequences = True, bias_initializer = keras.initializers.Constant(value=0.1)))
            model.add(LSTM(units = LSTM1, return_sequences = True, bias_initializer = keras.initializers.Constant(value=0.1)))
            model.add(LSTM(units = LSTM1, return_sequences = False, bias_initializer = keras.initializers.Constant(value=0.1)))
    
    model.add(Activation(LSTM_ACTIVATION))    
    model.add(Dropout(rate = DROPOUT))
    
    model.add(Dense(units               = DENSE_UNITS, 
                    activation          = ACTIVATION2,
                    kernel_initializer  = initializer,
                    bias_initializer    = bias))

    model.add(Dropout(rate              = DROPOUT))

    model.add(Dense(units               = y_train.shape[1], 
                    activation          = ACTIVATION2,
                    kernel_initializer  = initializer,
                    bias_initializer    = bias))
    
    model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(learning_rate=OPTIMIZER))
    return model


# In[7]:


def grid_search(df, col_selection, HISTORY, STEP_FACTOR,
                FILTERS1, KERNEL_SIZE1, PADDING1, DILATION_RATE1,
                CONV_ACTIVATION, CONV2, POOL_SIZE1, CONV_POOL2, 
                OPEN_END, DEPTH, LSTM1, LSTM_ACTIVATION, 
                DENSE_UNITS, DROPOUT, ACTIVATION2, 
                OPTIMIZER, counter,
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
                       CONV_ACTIVATION= CONV_ACTIVATION,
                       CONV2          = CONV2,
                       POOL_SIZE1     = POOL_SIZE1,
                       CONV_POOL2     = CONV_POOL2,
                       OPEN_END       = OPEN_END,
                       DEPTH          = DEPTH,
                       LSTM1          = LSTM1,
                       LSTM_ACTIVATION= LSTM_ACTIVATION,
                       DENSE_UNITS    = DENSE_UNITS, 
                       DROPOUT        = DROPOUT, 
                       ACTIVATION2    = ACTIVATION2, 
                       OPTIMIZER      = OPTIMIZER)

    checkpoint_path = results_path + 'checkpoints/'
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath       = checkpoint_path + str(counter) + '_cp.h5',
                                                 save_best_only = True,
                                                 monitor        = 'val_loss')

    model.fit(x_train, y_train,
                        epochs          = 400, 
                        batch_size      = 24,
                        validation_data = (x_test, y_test),
                        callbacks       = [checkpoint],
                        verbose         = 0,
                        shuffle         = False)

    model = load_model(checkpoint_path + str(counter) + '_cp.h5')
    
    x_test1, y_test1 = multivariate_data(df, TARGET_COL, TRAIN_TEST_SPLIT, None, HISTORY_SIZE, TARGET_SIZE, STEP)
    x_test1, y_test1 = remove_nan(x_test1, y_test1)
    
    predictions = model.predict(x_test1)
    predictions[predictions<0] = 0
    mse = mean_squared_error(y_test1, predictions, squared=True)
    mae = mean_absolute_error(y_test1, predictions)

    return mse, mae, train_shape, test_shape


# In[11]:


dt_only = ['year_2015', 'year_2016', 'year_2017', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']
sensors = ['181', '192', '226', '262', '288', '317', '373', '380', '532']
rain = ['DR_REGENM_10', 'RI_REGENM_10']
temp = ['U_BOOL_10', 'T_DRYB_10', 'TN_10CM_PAST_6H_10', 'T_DEWP_10', 'TN_DRYB_10', 'T_WETB_10', 'TX_DRYB_10', 'U_10']
pres_hum = ['P_NAP_MSL_10', 'VV_10', 'AH_10', 'MOR_10']
wind_cols = ['FF_10M_10', 'DD_10_sin', 'DD_10_cos', 'DDN_10_sin', 'DDN_10_cos', 'DD_STD_10_sin', 'DD_STD_10_cos', 'DDX_10_sin', 'DDX_10_cos', 'FF_SENSOR_10', 'FF_10M_STD_10', 'FX_SENSOR_10']
irr_cols = ['Q_GLOB_10', 'QN_GLOB_10', 'QX_GLOB_10', 'SQ_10']

parameters = {'col_selection'  : [[], dt_only, sensors, dt_only + sensors, dt_only + wind_cols + sensors, wind_cols + sensors, rain, temp, pres_hum, wind_cols, irr_cols, wind_cols + irr_cols, dt_only + wind_cols + irr_cols],
              'HISTORY'        : [1, 3, 5],
              'STEP_FACTOR'    : [1],
              'FILTERS1'       : [16],
              'KERNEL_SIZE1'   : [3],
              'PADDING1'       : ['same'],
              'DILATION_RATE1' : [1],
              'CONV_ACTIVATION': ['relu'],
              'CONV2'          : [False],
              'POOL_SIZE1'     : [3],
              'CONV_POOL2'     : [False],
              'OPEN_END'       : [False],
              'DEPTH'          : [1,2,3],
              'LSTM1'          : [80,100,120],
              'LSTM_ACTIVATION': ['tanh','relu'],
              'DROPOUT'        : [0.2], 
              'DENSE_UNITS'    : [100],
              'ACTIVATION2'    : ['relu'],
              'OPTIMIZER'      : [0.001, 0.0005, 0.0001]
             }


param_grid = ParameterGrid(parameters)
print(len(param_grid))

#%%


grid_results = pd.DataFrame(columns = ['counter', 'col_selection', 'HISTORY',  
                                       
                                       'FILTERS1', 'KERNEL_SIZE1', 'PADDING1', 'DILATION_RATE1', 
                                       'CONV_ACTIVATION', 'CONV2', 'POOL_SIZE1', 'CONV_POOL2',
                                       
                                       'STEP_FACTOR', 'OPEN_END', 'DEPTH', 'LSTM1', 'LSTM_ACTIVATION',
                                       
                                       'DROPOUT', 'DENSE_UNITS', 'ACTIVATION2', 
                                       
                                       'OPTIMIZER', 'mse', 'mae', 'train_samples', 'test_samples', 'train_time'])

output_path = results_path + 'gridsearch_results.csv'
counter = 0
for dict_ in param_grid:
    start = time()
    conv_mse, mae, train_shape, test_shape = grid_search(df, dict_['col_selection'], dict_['HISTORY'], dict_['STEP_FACTOR'], 
                                                        dict_['FILTERS1'], 
                                                        dict_['KERNEL_SIZE1'],
                                                        dict_['PADDING1'], 
                                                        dict_['DILATION_RATE1'], 
                                                        dict_['CONV_ACTIVATION'], 
                                                        dict_['CONV2'], 
                                                        dict_['POOL_SIZE1'], 
                                                        dict_['CONV_POOL2'],
                                                        dict_['OPEN_END'],
                                                        dict_['DEPTH'],
                                                        dict_['LSTM1'],
                                                        dict_['LSTM_ACTIVATION'],
                                                        dict_['DENSE_UNITS'],
                                                        dict_['DROPOUT'], 
                                                        dict_['ACTIVATION2'], 
                                                        dict_['OPTIMIZER'],
                                                        counter)
    train_time = round(time() - start, 2)
    grid_results = grid_results.append({'counter'    : counter, 
                                    'col_selection'  : dict_['col_selection'], 
                                    'HISTORY'        : dict_['HISTORY'], 
                                    'FILTERS1'       : dict_['FILTERS1'],
                                    'KERNEL_SIZE1'   : dict_['KERNEL_SIZE1'],
                                    'PADDING1'       : dict_['PADDING1'],
                                    'DILATION_RATE1' : dict_['DILATION_RATE1'],
                                    'CONV_ACTIVATION': dict_['CONV_ACTIVATION'],
                                    'CONV2'          : dict_['CONV2'],
                                    'POOL_SIZE1'     : dict_['POOL_SIZE1'],
                                    'CONV_POOL2'     : dict_['CONV_POOL2'],
                                    'STEP_FACTOR'    : dict_['STEP_FACTOR'], 
                                    'OPEN_END'       : dict_['OPEN_END'], 
                                    'DEPTH'          : dict_['DEPTH'], 
                                    'LSTM1'          : dict_['LSTM1'], 
                                    'LSTM_ACTIVATION': dict_['LSTM_ACTIVATION'],
                                    'DROPOUT'        : dict_['DROPOUT'], 
                                    'DENSE_UNITS'    : dict_['DENSE_UNITS'], 
                                    'ACTIVATION2'    : dict_['ACTIVATION2'], 
                                    'OPTIMIZER'      : dict_['OPTIMIZER'], 
                                    'mse'            : conv_mse, 
                                    'mae'            : mae, 
                                    'train_samples'  : train_shape[0], 
                                    'test_samples'   : test_shape[0], 
                                    'train_time'     : str(train_time)
                                   }, ignore_index = True
                                  )
    grid_results.to_csv(output_path, sep=',')
    counter += 1
    print('Done!: ' + str(counter) + ' train_time: ' + str(train_time))

