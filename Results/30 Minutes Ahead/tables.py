# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:54:37 2020

@author: Xander
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from os import listdir
from scipy.stats import pearsonr

#%%

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

def replace_values(df):
    dt_only = ['year_2015', 'year_2016', 'year_2017', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']
    sensors = ['181', '192', '226', '262', '288', '317', '373', '380', '532']
    rain = ['DR_REGENM_10', 'RI_REGENM_10']
    temp = ['U_BOOL_10', 'T_DRYB_10', 'TN_10CM_PAST_6H_10', 'T_DEWP_10', 'TN_DRYB_10', 'T_WETB_10', 'TX_DRYB_10', 'U_10']
    pres_hum = ['P_NAP_MSL_10', 'VV_10', 'AH_10', 'MOR_10']
    wind_cols = ['FF_10M_10', 'DD_10_sin', 'DD_10_cos', 'DDN_10_sin', 'DDN_10_cos', 'DD_STD_10_sin', 'DD_STD_10_cos', 'DDX_10_sin', 'DDX_10_cos', 'FF_SENSOR_10', 'FF_10M_STD_10', 'FX_SENSOR_10']
    irr_cols = ['Q_GLOB_10', 'QN_GLOB_10', 'QX_GLOB_10', 'SQ_10']
    
    csv_path = '../../pv_data/ML_input_15T_dwt.csv'
    dwt = pd.read_csv(csv_path,
                     index_col=0,
                     parse_dates=True,
                     infer_datetime_format = True,
                     nrows = 1)
    
    dwt1 = list(dwt[dwt.columns[dwt.columns.str.startswith('dwt1', na=False)]].columns)
    dwt2 = list(dwt[dwt.columns[dwt.columns.str.startswith('dwt2', na=False)]].columns)
    dwt3 = list(dwt[dwt.columns[dwt.columns.str.startswith('dwt3', na=False)]].columns)
    dwt4 = list(dwt[dwt.columns[dwt.columns.str.startswith('dwt4', na=False)]].columns)
    
    d = {'[]' : 'univariate' ,
         str(['year_2015', 'year_2016', 'year_2017', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']) : 'daytime',
         str(['181', '192', '226', '262', '288', '317', '373', '380', '532']) : 'sensors',
         str(dt_only + sensors) : 'daytime + sensors',
         str(dt_only + wind_cols + sensors) : 'daytime + wind + sensors',
         str(wind_cols + sensors) : 'wind + sensors',
         str(rain) : 'rain',
         str(temp) : 'temperature',
         str(pres_hum) : 'pressure and humidity',
         str(wind_cols) : 'wind',
         str(irr_cols) : 'irradiance',
         str(wind_cols + irr_cols) : 'wind + irradiance',
         str(irr_cols + dwt1) : 'daytime + wind + irradiance\n + DWT level 1',
         str(irr_cols + dwt2) : 'daytime + wind + irradiance\n + DWT level 2',
         str(irr_cols + dwt3) : 'daytime + wind + irradiance\n + DWT level 3',
         str(irr_cols + dwt4) : 'daytime + wind + irradiance\n + DWT level 4'
         } 
    df.col_selection = df.col_selection.map(d)
    df.OPTIMIZER = df.OPTIMIZER.map({'Adam' : 0.001,
                                     0.001  : 0.001,
                                     0.0005 : 0.0005,
                                     0.0001 : 0.0001})

    df = df.rename(columns = {'LSTM_ACTIVATION' : 'ACTIVATION1',
                              'CONV_ACTIVATION' : 'ACTIVATION1',
                              'col_selection'   : 'Feature collections',
                              'HISTORY'         : 'History (in days)',
                              'WAVELET'         : 'Wavelet function used',
                              'LEVEL'           : 'DWT level',
                              'FILTERS1'        : 'CNN filter layers',
                              'KERNEL_SIZE1'    : 'Kernel size',
                              'CONV2'           : 'Second CNN kernel',
                              'POOL_SIZE1'      : 'MaxPooling layer size',
                              'CONV_POOL2'      : 'Second set of cnn+pooling layers',
                              'DILATION_RATE1'  : 'Dilation rate',
                              'DEPTH'           : 'Number of LSTM layers',
                              'LSTM1'           : 'LSTM history vector length',
                              'DROPOUT'         : 'Dropout rate',
                              'ACTIVATION2'     : 'Dense layer activation function',
                              'DENSE_UNITS'     : 'Number of neurons in Dense layer',
                              'OPTIMIZER'       : 'Learning rate',
                              'mse'             : 'MSE',
                              'mae'             : 'MAE',
                              'train_time'      : 'Time required to train in seconds',
                              'mre_3'           : 'Mean ramp error window size 3',
                              'mre_5'           : 'Mean ramp error window size 5',
                              'mre_10'          : 'Mean ramp error window size 10',
                              'train_samples'   : '# samples available for training',
                              'test_samples'    : '# samples available for testing',
                              })
    df = df.rename(columns = {'ACTIVATION1'     : 'LSTM & CNN activation functions'})
    df = df.loc[:,~df.columns.duplicated()]
    df = round(df, 5)
    return df

def import_results(file_name):
    df = pd.read_csv(file_name, index_col = 'counter')
    
    drop_list = ['Unnamed: 0', 'STEP_FACTOR', 'OPEN_END', 'PADDING1', 'DILATION_RATE1']
    for column in drop_list:
        if column in df.columns:
            df = df.drop(column, axis = 1)
    df = replace_values(df)
    df = df.sort_values('MSE')
    best_model = df.iloc[0,:]
    return df, best_model

def columns(df_dict):
    cols = []
    for i in df_dict:
        cols.append(df_dict[i].columns)
    return pd.DataFrame(cols)

def make_results():
    results = {}
    best_models = {}
    for name in find_csv_filenames(None):
        model = name.strip('.csv').strip('_30')
        df, best = import_results(name)
        df['model archetype'] = model
        df = df.set_index('model archetype')
        results[model] = df
        best_models[model] = best
    
    results['no_prepro_conv'] = results['no_prepro_conv'].drop(['Number of LSTM layers', 'LSTM history vector length'], axis = 1)
    results['no_prepro_lstm'] = results['no_prepro_lstm'].drop(['CNN filter layers', 'Kernel size', 'Second CNN kernel', 
                                                                'MaxPooling layer size', 'Second set of cnn+pooling layers',], axis = 1)
    return results, best_models

results, best_models = make_results()
columns = columns(results)

#%%

def make_best_model_table(df_dict, prefixes):
    best = {}
    for prefix in prefixes:
        best_model = pd.DataFrame(best_models[prefix + '_lstm']).transpose().append(
            [pd.DataFrame(best_models[prefix + '_conv']).transpose(), 
             pd.DataFrame(best_models[prefix + '_conv_lstm']).transpose()], 
             sort = False
             ).transpose()
        best_model.columns = ['lstm', 'conv', 'conv-lstm']
        best[prefix] = best_model
    return best

def group_models(df_dict, prefixes):
    sorted_models = {}
    for prefix in prefixes:
        models = df_dict[prefix + '_lstm'].append([df_dict[prefix + '_conv'], df_dict[prefix + '_conv_lstm']], sort = False)
        models = models.sort_values('MSE')
        sorted_models[prefix] = models
    return sorted_models


prefixes = ['no_prepro', 'dwt', 'dwt_AF']
best_model_tables = make_best_model_table(best_models, prefixes)
grouped_models = group_models(results, prefixes)

#%% Export

best_models_table = pd.concat(best_model_tables, axis=1)
for i in range(0,len(best_models_table.columns),3):
    best_models_table.iloc[:,i][[2,3,5,6,7]] = np.nan
    best_models_table.iloc[:,i+1][[8,9]] = np.nan
best_models_table.columns = best_models_table.columns.set_levels(['No Preprocessing', 'DWT separate models','DWT as features'], level=0)

temporal_horizon = '30_min'
best_models_table.to_latex('tables/' + temporal_horizon + '_best_models.tex',
                           multicolumn = True,
                           bold_rows = True,
                           na_rep = '-',
                           label = 'tab: ' + temporal_horizon + 'best_models',
                           caption = 'The best trained model and its hyper parameters for each model archetype and DWT implementation')

#%%

pivot_for = ['Feature collections',
             'History (in days)',
             'DWT level',
             'Learning rate',
             'LSTM & CNN activation functions',
             'Second CNN kernel',
             'Second set of cnn+pooling layers',
             'Kernel size',
             'CNN filter layers',
             'MaxPooling layer size',
             'Number of LSTM layers',
             'LSTM history vector length']

def pivot_results(pivot_for = pivot_for, grouped_models = grouped_models):
    pivot_tables = {}
    for model in grouped_models.keys():
        tables = {}
        for i in pivot_for:
            if i in grouped_models[model].columns:
                pivot_table = pd.pivot_table(grouped_models[model],
                                             values = 'MSE',
                                             index = [i],
                                             columns = grouped_models[model].index)
                tables.update({i: pivot_table})
        table = pd.concat(tables)
        table.columns = ['conv', 'conv lstm','lstm']
        table = table[['lstm', 'conv', 'conv lstm']]
        pivot_tables.update({model : table})
    return pivot_tables

Hparam_tuning_results = pivot_results()



"""
These tables are not done yet, but i can't be bothered to change the labels of some row labels in python from boolean and float to integer, 
so i will do that manually.
"""
Hparam_tuning_results['no_prepro'].to_latex('tables/' + temporal_horizon + '_Hparam_pivot_no_preprocessing.tex',
                                            multirow = True,
                                            bold_rows = False,
                                            na_rep = '-',
                                            label = 'tab: Hparam_pivot_no_preprocessing',
                                            caption = 'Pivot table of MSE per hyper parameter and model archetype without DWT')

Hparam_tuning_results['dwt'].to_latex('tables/' + temporal_horizon + '_Hparam_pivot_dwt.tex',
                                            multirow = True,
                                            bold_rows = False,
                                            na_rep = '-',
                                            label = 'tab: Hparam_pivot_dwt',
                                            caption = 'Pivot table of MSE per hyper parameter and model archetype using a separate model for each level of DWT')

Hparam_tuning_results['dwt_AF'].to_latex('tables/' + temporal_horizon + '_Hparam_pivot_dwt_AF.tex',
                                            multirow = True,
                                            bold_rows = False,
                                            na_rep = '-',
                                            label = 'tab: Hparam_pivot_dwt_AF',
                                            caption = 'Pivot table of MSE per hyper parameter and model archetype using l;inearly interpolated DWT levels (of continuous features) as additional features')

#%% 

plt.figure(figsize = (10,7))
plt.scatter(x = grouped_models['no_prepro']['MSE'], y=grouped_models['no_prepro']['Mean ramp error window size 3'], s=0.3, 
            label = 'Mean Ramp Error 3, pearson R = ' + str(round(pearsonr(grouped_models['no_prepro']['MSE'], 
                                                              grouped_models['no_prepro']['Mean ramp error window size 3'])[0], 3)))

plt.scatter(x = grouped_models['no_prepro']['MSE'], y=grouped_models['no_prepro']['Mean ramp error window size 5'], s=0.3, 
            label = 'Mean Ramp Error 5, pearson R = ' + str(round(pearsonr(grouped_models['no_prepro']['MSE'], 
                                                              grouped_models['no_prepro']['Mean ramp error window size 5'])[0], 3)))

plt.scatter(x = grouped_models['no_prepro']['MSE'], y=grouped_models['no_prepro']['Mean ramp error window size 10'], s=0.3, 
            label = 'Mean Ramp Error 10, pearson R = ' + str(round(pearsonr(grouped_models['no_prepro']['MSE'], 
                                                               grouped_models['no_prepro']['Mean ramp error window size 10'])[0], 3)))

plt.scatter(x = grouped_models['no_prepro']['MSE'], y = grouped_models['no_prepro']['MAE'], s=0.3, 
            label = 'MAE, pearson R = ' + str(round(pearsonr(grouped_models['no_prepro']['MSE'], 
                                                            grouped_models['no_prepro']['MAE'])[0], 3)))
plt.legend(markerscale = 9)
plt.xlabel('MSE')
plt.ylabel('Error metric value')
plt.savefig('figures/mseVmre.eps')

