# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:19:31 2020

@author: Xander
"""
import numpy as np
import matplotlib.pyplot as plt

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

def plot_train_test(model_input, expected_output, fig_title, fontsize, predictions = None, end_range = None):
    if end_range == None:
        end_range = model_input.shape[0]
    if hasattr(predictions, "__len__"):
        cols = 3
        FIGSIZE = (15,4)
    else:
        cols = 2
        FIGSIZE = (10,4)
        
    fig, axs = plt.subplots(1,cols, sharey = True, figsize = FIGSIZE, constrained_layout = True)
    for i in range(0, end_range):
        axs[0].plot(model_input[i,:,0])
        axs[0].set_title('PV power input', fontsize = fontsize+2)
        axs[0].set_xlabel('Timesteps, Δt = 1 minute', fontsize = fontsize)
        axs[0].set_ylabel('% maximum \nPV power output', fontsize = fontsize)
        axs[1].plot(expected_output[i,:])
        axs[1].set_title('expected output', fontsize = fontsize+2)
        axs[1].set_xlabel('Timesteps, Δt = 1 minute', fontsize = fontsize)
        if hasattr(predictions, "__len__"):
            axs[2].plot(predictions[i,:])
            axs[2].set_title('model output', fontsize = fontsize+2)
            axs[2].set_xlabel('Timesteps, Δt = 1 minute', fontsize = fontsize)
    plt.legend()
    plt.savefig(fig_title)
    
