# Much inspiration came from this blog post: 
    # https://blog.edugrad.com/forecasting-and-modeling-with-a-multivariate-time-series-in-python/

# Hyper parameters are listed at the top of the cell they belong too.

from pathlib import Path
import time as t
import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

import math
import tensorflow.keras
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from scipy import stats
from statsmodels.tsa.stattools import pacf

#%% data import



csv_path = '../Datasets/all_data/all_data.csv'
df = pd.read_csv(csv_path,
                 index_col='DateTime',
                 parse_dates=True)

index = pd.date_range(start = '2015-01-01 00:00:10+00:00', end = '2017-12-31 23:59:10+00:00', freq = '10S')
df1 = pd.DataFrame(index = index)
df1 = df1.join(df, how='left')

# generating features from the date-time-index
df1['date_time'] = df1.index

df1['year'] = df1['date_time'].apply(lambda x: x.year)
df1['quarter'] = df1['date_time'].apply(lambda x: x.quarter)
df1['month'] = df1['date_time'].apply(lambda x: x.month)
df1['day_of_month'] = df1['date_time'].apply(lambda x: x.day)
df1['hour'] = df1.index.hour
df1['minute']=df1.index.minute

df1 = df1.drop('date_time', axis=1)


#%% Data evaluation

df1.info()
df1.describe()

stat, p = stats.normaltest(df1['151'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
   print('Data looks Gaussian (fail to reject H0)')
else:
   print('Data does not look Gaussian (reject H0)')
   
plt.figure()
sns.distplot(df1['151'], hist=False, label = 'PV system power output data').set_xlim(left=0)
plt.xlabel('power output')
plt.ylabel('percentage of measurements')
plt.legend()
plt.savefig('distplot.eps')

print('Kurtosis of normal distribution: {}'.format(stats.kurtosis
                                                      (df1['151'])))
print('Skewness of normal distribution: {}'.format(stats.skew
                                                      (df1['151'])))

plt.figure(figsize=(14,5), dpi=200)
plt.subplot(1,3,1)
plt.subplots_adjust(wspace=0.2)
sns.boxplot(x='month', y='151', data=df1)
plt.xlabel('month')
plt.title('Box plot of monthly PV power')
sns.despine(left=True)
plt.tight_layout()

plt.subplot(1,3,2)
sns.boxplot(x="hour", y='151', data=df1)
plt.xlabel('hour')
plt.title('Box plot of hourly PV power')
sns.despine(left=True)
plt.tight_layout();

plt.subplot(1,3,3)
sns.boxplot(x="quarter", y='151', data=df1)
plt.xlabel('quarter')
plt.title('Box plot of quarterly PV power')
sns.despine(left=True)
plt.tight_layout();

plt.savefig('quarterly__monthly_daily_power.png')



data = df1.copy()
data = data['151']
data = data.to_frame()
onemin = data['151'].resample('1T').mean().interpolate(method='linear', limit = 60)
onemin = onemin.to_frame()
onemin.columns = ['Frequency: 1 min']
fifteenmin = data['151'].resample('15T').mean().interpolate(method='linear', limit = 4)
fifteenmin = fifteenmin.to_frame()
fifteenmin.columns = ['Frequency: 15 min']
data.columns = ['Measurements']

fig, ax = plt.subplots(1,3, dpi = 300)
sns.heatmap(data.isnull(), cbar=False, ax = ax[0])
sns.heatmap(onemin.isnull(), cbar=False, ax = ax[1])
sns.heatmap(fifteenmin.isnull(), cbar=False, ax = ax[2])
#ax[0].get_yaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)
ax[2].get_yaxis().set_visible(False)
plt.sca(ax[0])
plt.yticks(range(0,9469435,3156478), ['2015', '2016', '2017'])

plt.savefig('missing_values_heatmap.png')

#%%

csv_path = '../ML_input_15T.csv'
df = pd.read_csv(csv_path,
                 index_col=0,
                 parse_dates=True,
                 infer_datetime_format = True)
# df = df.between_time(sun_up, sun_down)

TIME_STEPS_PER_DAY = len(df.loc['1-1-2016'])

times = df.index.time
times = pd.DataFrame(times)
times = times[0].unique()

min_mean_max_power_at_time = pd.DataFrame()

for i in range(len(times)):
    time, min, mean, max = times[i], df.at_time(times[i])['151'].min(), df.at_time(times[i])['151'].mean(), df.at_time(times[i])['151'].max()
    min_mean_max_power_at_time = min_mean_max_power_at_time.append([[time, min, mean, max]])
min_mean_max_power_at_time.columns = ['time', 'min', 'mean', 'max']
min_mean_max_power_at_time = min_mean_max_power_at_time.set_index('time')
min_mean_max_power_at_time = round(min_mean_max_power_at_time, 2)
min_mean_max_power_at_time.plot(figsize = (8,4))
plt.ylabel('measured PV power [W]')

plt.axvline(x = '04:00:00')
plt.axvline(x = '19:45:00')
plt.savefig('min_mean_max_pv_power.eps')

