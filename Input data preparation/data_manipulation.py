# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:36:06 2020

@author: Xander
"""
#%% append 2015, 2016 and 2017 from different csv files into a single csv file that contains all the data for that sensorid
import pandas as pd
import numpy as np
import seaborn as sns

# =============================================================================
# sensorid = ['151', '181', '192', '226', '262', '288', '317', '373', '380', '532']
# 
# 
# df = pd.read_csv('sensorid_' + sensorid[0] + '_all_years.csv', index_col = 0, parse_dates = True)
# df1 = pd.read_csv('sensorid_' + sensorid[1] + '_all_years.csv', index_col = 0, parse_dates = True)
# df2 = pd.read_csv('sensorid_' + sensorid[2] + '_all_years.csv', index_col = 0, parse_dates = True)
# df3 = pd.read_csv('sensorid_' + sensorid[3] + '_all_years.csv', index_col = 0, parse_dates = True)
# df4 = pd.read_csv('sensorid_' + sensorid[4] + '_all_years.csv', index_col = 0, parse_dates = True)
# df5 = pd.read_csv('sensorid_' + sensorid[5] + '_all_years.csv', index_col = 0, parse_dates = True)
# df6 = pd.read_csv('sensorid_' + sensorid[6] + '_all_years.csv', index_col = 0, parse_dates = True)
# df7 = pd.read_csv('sensorid_' + sensorid[7] + '_all_years.csv', index_col = 0, parse_dates = True)
# df8 = pd.read_csv('sensorid_' + sensorid[8] + '_all_years.csv', index_col = 0, parse_dates = True)
# df9 = pd.read_csv('sensorid_' + sensorid[9] + '_all_years.csv', index_col = 0, parse_dates = True)
# 
# data = pd.concat([df, df1, df2, df3, df4, df5, df6, df7, df8, df9], axis=1)
# data.to_csv('all_data.csv')
# =============================================================================

total_na_all_data = data.isna().sum()

# =============================================================================
# data = pd.read_csv('all_data.csv', index_col=0, parse_dates=True)
# =============================================================================

# =============================================================================
# sns.heatmap(data.isnull(), cbar=False)
# describe = data.describe()
# =============================================================================

# =============================================================================
# data15 = data.resample('15T').median()
# data15 = data15.interpolate(limit = 4)
# data15.to_csv('all_data_resample-15T_interpolate-4.csv')
# =============================================================================

# =============================================================================
# sns.heatmap(data15.isnull(), cbar=False)
# describe = data15.describe()
# =============================================================================

data1 = data.resample('1T').median()
data1_resample_1T_median_total_na = data1.isna().sum()
data1 = data1.interpolate(limit = 4)
data1_resample_1T_median_interpolate_4_total_na = data1.isna().sum()
data1.to_csv('all_data_resample-1T_interpolate-4.csv')

sns.heatmap(data1.isnull(), cbar=False)
describe = data1.describe()

# =============================================================================
# numpy_data1 = data1.values
# np.save('all_data_resample-1T_interpolate-4.npy', numpy_data1)
# 
# numpy_data15 = data15.values
# np.save('all_data_resample-15T_interpolate-4.npy', numpy_data15)
# =============================================================================

# =============================================================================
# from time import time
# begin = time()
# data_numpy = np.load('all_data_resample-1T_interpolate-4.npy', allow_pickle='TRUE')
# data_numpy = pd.DataFrame(data_numpy)
# end = time()
# total_numpy = end - begin
# print(total_numpy)
# 
# begin = time()
# data_pandas = pd.read_csv('all_data_resample-1T_interpolate-4.csv')
# end = time()
# total_pandas = end - begin
# print(total_pandas)
# 
# difference = total_pandas / total_numpy
# print(difference)
# =============================================================================

# =============================================================================
# def make_df(sensorid):
#     csv_2015 = '2015/sensorid_' + sensorid + '_2015.csv'
#     csv_2016 = '2016/sensorid_' + sensorid + '_2016.csv'
#     csv_2017 = '2017/sensorid_' + sensorid + '_2017.csv'
#     fifteen = pd.read_csv(csv_2015, index_col = 'DateTime', parse_dates = True)
#     sixteen = pd.read_csv(csv_2016, index_col = 'DateTime', parse_dates = True)
#     seventeen = pd.read_csv(csv_2017, index_col = 'DateTime', parse_dates = True)
#     all_years = fifteen.append(sixteen)
#     all_years = all_years.append(seventeen)
#     file_name = 'sensorid_' + sensorid + '_all_years.csv'
#     all_years.to_csv(file_name)
# 
# for sensor in sensorid:
#     make_df(sensor)
# =============================================================================
