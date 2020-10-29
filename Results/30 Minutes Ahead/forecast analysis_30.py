# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 09:12:38 2020

@author: Xander
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import functions_temp as func



data = pd.read_csv('../../../pv_data/ML_input_1T_dwt.csv',
                 index_col=0,
                 parse_dates=True,
                 infer_datetime_format = True)
#%%
df = data.copy()
sun_up, sun_down = '04:00:00', '19:45:00'
df = df.between_time(sun_up, sun_down)
TIME_STEPS_PER_DAY = 64
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

df1 = df.copy()
#%%

df = df1['7-1-2017' : '8-1-2017'].copy()
dt_only = ['year_2015', 'year_2016', 'year_2017', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']
irradiance = ['Q_GLOB_10', 'QN_GLOB_10', 'QX_GLOB_10', 'SQ_10']
wind = ['FF_10M_10', 'DD_10_sin', 'DD_10_cos', 'DDN_10_sin', 'DDN_10_cos', 'DD_STD_10_sin', 'DD_STD_10_cos', 'DDX_10_sin', 'DDX_10_cos', 'FF_SENSOR_10', 'FF_10M_STD_10', 'FX_SENSOR_10']
dwt1 = list(df[df.columns[df.columns.str.startswith('dwt1', na=False)]].columns)
dwt2 = list(df[df.columns[df.columns.str.startswith('dwt2', na=False)]].columns)
dwt3 = list(df[df.columns[df.columns.str.startswith('dwt3', na=False)]].columns)
dwt4 = list(df[df.columns[df.columns.str.startswith('dwt4', na=False)]].columns)
df = df[['151'] + irradiance + dwt4]
cols = df.columns
df = df.values
HISTORY = 5
TARGET_COL = df[:,0]
HISTORY_SIZE = TIME_STEPS_PER_DAY * HISTORY
TARGET_SIZE = 32
STEP = TIME_STEPS_PER_DAY
x_test1, y_test1 = func.multivariate_data(df, TARGET_COL, 0, None, HISTORY_SIZE, TARGET_SIZE, 15)
x_test1, y_test1 = func.remove_nan(x_test1, y_test1)

#%%
DFL = load_model('../best_models/34_cp.h5')    
predictions = DFL.predict(x_test1)
predictions[predictions < 0] = 0

#%%

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test1, predictions))

#%%
# Plot measured v predicted
fontsize = 13

plt.figure(figsize=(7,5), dpi = 300)
plt.scatter(y_test1,predictions, s = 0.05, label = 'Forecast V True')
plt.scatter(y_test1,y_test1, s = 0.1, label = 'True V True')
plt.ylabel('Forecasted values \n[% maximum PV power output in test set]', fontsize = fontsize)
plt.xlabel('True values \n[% maximum PV power output in test set]', fontsize = fontsize)
plt.tight_layout()
plt.legend(markerscale = 10, fontsize = fontsize-1)
plt.savefig('../figures/TruevPred_plot_30.png')

# Plot prediction dataframe
fig, ax = plt.subplots(2,1, figsize = (11,4.5), constrained_layout = True, sharex = True)
ax1 = ax[0].imshow(y_test1[30:331,:].T, vmin = 0, vmax = 0.9)
ax[0].set_title('Measurements', fontsize = fontsize+2)
ax[0].set_ylabel('Time [minutes]', fontsize = fontsize)
ax0 = ax[1].imshow(predictions[30:331,:].T, vmin = 0, vmax = 0.9)
ax[1].set_title('Forecasted values', fontsize = fontsize+2)
ax[1].set_xlabel('Samples', fontsize = fontsize)
ax[1].set_ylabel('Time [minutes]', fontsize = fontsize)
fig.colorbar(ax1, orientation = 'horizontal').set_label(label = 'Percentage of maximum PV power output', size = fontsize)
fig.suptitle('')
plt.show()
plt.savefig('../figures/TruevPred_Matrix_30.eps')

# Plot all input & corresponding output
import functions_temp as func
sample = np.array([0, 100, 1000, 150])
func.plot_train_test(x_test1[sample], 
                     y_test1[sample], 
                     fig_title = '../figures/input_output_plot_30.eps', 
                     predictions = predictions[sample], 
                     end_range = 4, 
                     fontsize = fontsize+8)

# Plot DWT components and source
def plot_DWT_components(day, source_name, select_feature, var_name):
    temp = []
    for i in range(x_test1.shape[0]):
        test = pd.DataFrame(x_test1[i,:,:], columns = cols)
        temp.append(test[[source_name, dwt4[0+select_feature], dwt4[9+select_feature], dwt4[18+select_feature], dwt4[27+select_feature], dwt4[36+select_feature]]].values)
    temp = np.array(temp)
    test2 = pd.DataFrame(temp[day,:,:])
    colors = ['r','g','b','c','m', 'y']
    dwt_level = len(test2.columns)
    columns = [var_name, 'DWT level 4 approximation function', 'DWT level 4 detailed function', 'DWT level 3 detailed function', 'DWT level 2 detailed function', 'DWT level 1 detailed function',]
    fig, axs = plt.subplots(dwt_level,1, sharex = True, figsize=(12,7), tight_layout=True)
    for i in range(dwt_level):
        axs[i].plot(test2.iloc[:300,i], label=columns[i], color=colors[i])
        axs[i].set_title(columns[i], fontsize = fontsize)
    plt.xlabel('Time in minutes (5 hours total)', fontsize = fontsize+1)
    fig.savefig('../figures/' + var_name  + '_dwt_30.eps')
plot_DWT_components(174, '151', 0, 'PV power')

plt.figure()
sns.distplot(y_test1, hist=False, label='True').set_xlim(left = 0)
sns.distplot(predictions.reshape(predictions.size), hist=False, label = 'Predictions')
plt.ylabel('% values', fontsize = fontsize+1)
plt.xlabel('% maximum PV power output in test set', fontsize = fontsize+1)
plt.legend(fontsize = fontsize-1)
plt.savefig('../figures/frequency_histogram_30.eps')
