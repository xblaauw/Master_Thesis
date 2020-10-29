#!/usr/bin/env python
# coding: utf-8

# In[20]:

from sys import platform
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

# library obtained from: https://github.com/nii3lsh?tab=repositories needs to be in same directory as this script
import ramp_score as rs

# In[21]:

results_path = 'dwt_conv_30/params_1/'
Path(results_path).mkdir(parents=True, exist_ok=True)

sun_up, sun_down = '04:00:00', '19:45:00'

# makes life easier when switching between personal and cluster environment
if platform == "win32":
    csv_path = '../../../pv_data/ML_input_1T.csv'
else:
    csv_path = '../pv_data/ML_input_1T.csv'
    
df = pd.read_csv(csv_path,
                 index_col=0,
                 parse_dates=True,
                 infer_datetime_format = True)
df = df.between_time(sun_up, sun_down)

columns = df.columns
index = df.index
scale_all = MinMaxScaler()
df = scale_all.fit_transform(df)
df = pd.DataFrame(df, index = index, columns = columns)

TIME_STEPS_PER_DAY = len(df.loc['1-1-2016'])
TRAIN_TEST_SPLIT = len(df.loc['1-1-2015':'31-12-2016'])


# In[22]:


def scale_df(df, cols):
    
    scaler_target = MinMaxScaler()
    target = scaler_target.fit_transform(df[:,:1])
    if len(cols) > 1:
        scaler_features = MinMaxScaler()
        features = scaler_features.fit_transform(df[:,1:])
        df = np.concatenate((target, features), axis=1)
        df = pd.DataFrame(df, columns = cols)
    else:
        df = pd.DataFrame(target, columns = cols)
    return scaler_target, df

def dwt(df, WAVELET, LEVEL):
    cols = df.columns
    coeff = pywt.wavedec(df, WAVELET, level = LEVEL, axis = 0)
    dwt = []
    scalers = []
    for I in range(LEVEL + 1):
        scaler_0, coeff_0 = scale_df(coeff[I], cols)
        dwt.append(coeff_0)
        scalers.append(scaler_0)
    return scalers, dwt

def ts_per_day(coeff, TIME_STEPS_PER_DAY):
    ts = []
    ts.append(int(TIME_STEPS_PER_DAY / 2**(len(coeff)-1)))
    for i in range(1, len(coeff)):
        ts.append(int(TIME_STEPS_PER_DAY / 2**((len(coeff))-i)))
    return ts


def mean_ramp_error(y_true, y_pred):
    ramp_results_3 = []
    for i in range(y_true.shape[0]):
        ramp_score_3 = rs.get_ramp_score(y_true[i,:], y_pred[i,:], avg_mins = 3, plot=False)
        ramp_results_3.append(ramp_score_3)
    return np.array(ramp_results_3).mean()

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


# In[24]:


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


# In[25]:


def grid_search(df, col_selection, HISTORY, STEP_FACTOR,
                LEVEL, WAVELET,
                FILTERS1, KERNEL_SIZE1, PADDING1, DILATION_RATE1, ACTIVATION1, 
                CONV2, POOL_SIZE1, CONV_POOL2, DENSE_UNITS, DROPOUT, ACTIVATION2, 
                OPTIMIZER, counter,
                TIME_STEPS_PER_DAY = 64, 
                TRAIN_TEST_SPLIT = TRAIN_TEST_SPLIT):
    df = df[['151'] + col_selection]
    
    df_eval = df.values
    x_eval, y_eval = multivariate_data(df_eval, df_eval[:,0], TRAIN_TEST_SPLIT, None, TIME_STEPS_PER_DAY * HISTORY, int(TIME_STEPS_PER_DAY / 2), TIME_STEPS_PER_DAY)
    x_eval, y_eval = remove_nan(x_eval, y_eval)
    
    scalers, wavelets = dwt(df, WAVELET, LEVEL)
    
    TS = ts_per_day(wavelets, TIME_STEPS_PER_DAY)
    TTsplit = ts_per_day(wavelets, TRAIN_TEST_SPLIT)
    
    PREDICTIONS = []
    
    coeff = 0
    for TIME_STEPS_PER_DAY in TS:

        df = wavelets[coeff].values
        
        TARGET_COL = df[:,0]
        HISTORY_SIZE = TIME_STEPS_PER_DAY * HISTORY
        TARGET_SIZE = int(TIME_STEPS_PER_DAY / 2)
        STEP = TIME_STEPS_PER_DAY
        
        x_train, y_train = multivariate_data(df, TARGET_COL, 0, TTsplit[coeff], HISTORY_SIZE, TARGET_SIZE, STEP)
        x_train, y_train = remove_nan(x_train, y_train)
        train_shape = x_train.shape

        x_test, y_test = multivariate_data(df, TARGET_COL, TTsplit[coeff], None, HISTORY_SIZE, TARGET_SIZE, STEP)
        x_test, y_test = remove_nan(x_test, y_test)
        test_shape = x_test.shape
        

        
        model = make_model(x_train        = x_train,
                           y_train        = y_train,
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
        checkpoint = keras.callbacks.ModelCheckpoint(filepath       = checkpoint_path + str(counter) + '_' + str(coeff) + '_cp.h5',
                                                     save_best_only = True,
                                                     monitor        = 'val_loss')

        history = model.fit(x_train, y_train,
                            epochs          = 200, 
                            batch_size      = 24,
                            validation_data = (x_test, y_test),
                            callbacks       = [checkpoint],
                            verbose         = 0,
                            shuffle         = False)

        model = load_model(checkpoint_path + str(counter) + '_' + str(coeff) + '_cp.h5')

        x_test1, y_test1 = multivariate_data(df, TARGET_COL, TTsplit[coeff], None, HISTORY_SIZE, TARGET_SIZE, TIME_STEPS_PER_DAY)
        x_test1, y_test1 = remove_nan(x_test1, y_test1)

        predictions = model.predict(x_test1)
        predictions = scalers[coeff].inverse_transform(predictions)
        PREDICTIONS.append(predictions[0:5000])
        
        coeff += 1
    
    predictions = pywt.waverec(PREDICTIONS, WAVELET, axis = 1)
    predictions[predictions<0] = 0
    
    mse = mean_squared_error(y_eval[0:5000, :], predictions, squared=True)
    mae = mean_absolute_error(y_eval[0:5000, :], predictions)
    mre_3 = mean_ramp_error(y_eval[0:5000, :], predictions)

    return mre_3, mse, mae, train_shape, test_shape, PREDICTIONS


    # In[29]:


wind_cols = ['FF_10M_10', 'DD_10_sin', 'DD_10_cos', 'DDN_10_sin', 'DDN_10_cos', 'DD_STD_10_sin', 'DD_STD_10_cos', 'DDX_10_sin', 'DDX_10_cos', 
             'FF_SENSOR_10', 'FF_10M_STD_10', 'FX_SENSOR_10']
irr_cols = ['Q_GLOB_10', 'QN_GLOB_10', 'QX_GLOB_10', 'SQ_10']
dt_only = ['year_2015', 'year_2016', 'year_2017', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 
             'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']

parameters = {'col_selection'   : [[], wind_cols, irr_cols, wind_cols + irr_cols],
              'HISTORY'         : [3, 5],
              'LEVEL'           : [0,1,2,3,4],
              'WAVELET'         : ['haar'],
              'STEP_FACTOR'     : [1],
              'FILTERS1'        : [64],
              'KERNEL_SIZE1'    : [5, 7],
              'PADDING1'        : ['same'],
              'DILATION_RATE1'  : [1],
              'ACTIVATION1'     : ['relu'],
              'CONV2'           : [True],
              'POOL_SIZE1'      : [5, 7],
              'CONV_POOL2'      : [False],
              'DENSE_UNITS'     : [100],
              'DROPOUT'         : [0.2],
              'ACTIVATION2'     : ['relu'],
              'OPTIMIZER'       : ['Adam']
              }

# parameters = {'col_selection'   : [wind_cols + irr_cols],
#               'HISTORY'         : [3, 5],
#               'LEVEL'           : [0,1,2,3,4],
#               'WAVELET'         : ['haar'],
#               'STEP_FACTOR'     : [1],
#               'FILTERS1'        : [64],
#               'KERNEL_SIZE1'    : [7],
#               'PADDING1'        : ['same'],
#               'DILATION_RATE1'  : [1],
#               'ACTIVATION1'     : ['relu'],
#               'CONV2'           : [True],
#               'POOL_SIZE1'      : [7],
#               'CONV_POOL2'      : [False],
#               'DENSE_UNITS'     : [100],
#               'DROPOUT'         : [0.2],
#               'ACTIVATION2'     : ['relu'],
#               'OPTIMIZER'       : ['Adam']
#              }

param_grid = ParameterGrid(parameters)
print(len(param_grid))

#%%
counter = 0
grid_results = pd.DataFrame(columns = ['counter', 'col_selection', 'HISTORY', 'STEP_FACTOR', 
                                       'LEVEL', 'WAVELET',
                                       'FILTERS1', 'KERNEL_SIZE1', 'PADDING1', 'DILATION_RATE1', 'ACTIVATION1',
                                       'CONV2', 'POOL_SIZE1', 'CONV_POOL2', 'DENSE_UNITS', 'DROPOUT', 'ACTIVATION2',
                                       'OPTIMIZER', 'mse', 'mae', 'mre_3', 'train_samples', 'test_samples', 'train_time'])
output_path = results_path + 'gridsearch_results.csv'


for dict_ in param_grid:
    start = time()
    mre_3, conv_mse, mae, train_shape, test_shape, predictions = grid_search(df, dict_['col_selection'], dict_['HISTORY'], dict_['STEP_FACTOR'], 
                                                                                 dict_['LEVEL'], dict_['WAVELET'],
                                                                                 dict_['FILTERS1'], dict_['KERNEL_SIZE1'], dict_['PADDING1'], dict_['DILATION_RATE1'], dict_['ACTIVATION1'], 
                                                                                 dict_['CONV2'], dict_['POOL_SIZE1'], dict_['CONV_POOL2'], dict_['DENSE_UNITS'], dict_['DROPOUT'], dict_['ACTIVATION2'], 
                                                                                 dict_['OPTIMIZER'],
                                                                                 counter)
    train_time = round(time() - start, 2)
    grid_results = grid_results.append({'counter'       : counter, 
                                        'col_selection' : dict_['col_selection'], 
                                        'HISTORY'       : dict_['HISTORY'], 
                                        'STEP_FACTOR'   : dict_['STEP_FACTOR'], 
                                        'LEVEL'         : dict_['LEVEL'],
                                        'WAVELET'       : dict_['WAVELET'],
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
                                        'mre_3'         : mre_3,
                                        'train_samples' : train_shape[0], 
                                        'test_samples'  : test_shape[0], 
                                        'train_time'    : str(train_time)
                                       }, ignore_index = True
                                      )
    
    grid_results.to_csv(output_path, sep=',')
    counter += 1
    print('Done!: ' + str(counter) + ' train_time: ' + str(train_time))


