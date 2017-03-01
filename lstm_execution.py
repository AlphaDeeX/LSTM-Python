# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 12:32:03 2017

@author: tismi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lstm_class_object import LSTMPopulation
import sys

def normalise(signal):
    mu = np.mean(signal)
    variance = np.mean((signal - mu)**2)
    signal_normalised = (signal - mu)/(np.sqrt(variance + 1e-8))
    return signal_normalised
    
t_range = np.linspace(0,100,1000)
train_df_roc_signal_unnormalised = np.sin(2*np.pi*300*t_range) + 0.5*np.sin(2*np.pi*t_range)

temp = train_df_roc_signal_unnormalised - min(train_df_roc_signal_unnormalised)
train_df_roc_signal = (temp)/max(temp)


plt.figure(1)
plt.plot(train_df_roc_signal[0:100])

seq_len = 24*3
input_size = 1
hidden_size_a = 200

output_size = 1
learning_rate = 1e-3
n, p = 0, 0

W_out = np.random.randn(output_size, hidden_size_a) / np.sqrt(output_size)

lstm_a = LSTMPopulation(input_size, hidden_size_a)

signal = np.zeros((seq_len,1))
target = np.zeros((seq_len,output_size))

mW_out = np.zeros_like(W_out)

j=0
k=0
for i in xrange(1000):
    if j+seq_len+output_size >= len(train_df_roc_signal):
        j=0
        lstm_a.reset_states()

    signal[:,0] = train_df_roc_signal[j:j+seq_len]
    target[:,0] = train_df_roc_signal[j+1:j+1+seq_len]

    lstm_a.forward(signal)
    lstm_a_hidden_out = lstm_a.get_hidden_output()

    
    output = lstm_a_hidden_out.dot(W_out.T)   

    error = output - target
    dW_out = (error).T.dot(lstm_a_hidden_out)
    
    loss = np.mean(np.square(output - target))
    
    dh_out = (error).dot(W_out)
  
    lstm_a.backward(dh_out)
    lstm_a.train_network(learning_rate)    
    
    for param, dparam, mem in zip([W_out],
                              [dW_out],
                              [mW_out]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
    
    print (k, loss)

    k += 1
    j += 1


# Testing phase
for ll in range(1):
    index = 400+ll*100
    plot_len = 24*10
    next_vals = 12
    
    sample_signal = np.zeros((plot_len,1))
    sample_signal[:,0] = train_df_roc_signal[index:index+plot_len] 
    sample_signal_plotting = train_df_roc_signal[index:index + plot_len + next_vals] 
    
    dd = lstm_a.sample_network(sample_signal, W_out, next_vals)
    sampled_output = dd.dot(W_out.T)
    #y_out = dd.dot(W_out.T)
    #sampled_output = 1.0 / (1.0 + np.exp(-y_out))
    
    plt.figure(2)
    plt.plot(sampled_output[:,0])
    plt.hold(True)
    plt.plot(sample_signal_plotting[:], 'r')
    plt.title('Prediction vs Actual Signal')
    
    plt.figure(3)
    plt.plot(sampled_output[plot_len:plot_len+next_vals,0])
    plt.hold(True)
    plt.plot(sample_signal_plotting[plot_len:plot_len+next_vals], 'r')
    plt.title('Prediction Mode - Blue (Prediction), Red (Actual)')
    plt.hold(False)
    plt.show()

































