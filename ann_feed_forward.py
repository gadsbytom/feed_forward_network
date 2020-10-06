#!/usr/bin/env python
# coding: utf8

import numpy as np
from sklearn.metrics import accuracy_score

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# tanh function
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# feed forward the x values through the neurons
def feed_forward(X, weights, act):

    no_layers = len(weights.keys())
    neurons = {}
    act_layers = {}


    for i in range(no_layers):
        if i == 0:
            act_layers[f'{i}'] = {}
            input_weighted_x = np.dot(X, weights[f'{i}'])
            if act=='sigmoid':
                act_layers[f'{i}']['normal'] = sigmoid(input_weighted_x)
            elif act=='tanh':
                act_layers[f'{i}']['normal']  = tanh(input_weighted_x)
            neurons[f'{i}'] = act_layers[f'{i}']['normal']
            act_layers[f'{i}']['bias']  = np.hstack([act_layers[f'{i}']['normal'] , np.ones((act_layers[f'{i}']['normal'] .shape[0], 1))])
        elif i < no_layers-1:
            act_layers[f'{i}'] = {}
            hidden_weighted_x = np.dot(act_layers[f'{i-1}']['bias'], weights[f'{i}'])
            if act=='sigmoid':
                act_layers[f'{i}']['normal'] = sigmoid(hidden_weighted_x)
            elif act=='tanh':
                act_layers[f'{i}']['normal'] = tanh(hidden_weighted_x)
            neurons[f'{i}'] = act_layers[f'{i-1}']['normal']
            act_layers[f'{i}']['bias'] = np.hstack([act_layers[f'{i}']['normal'], np.ones((act_layers[f'{i}']['normal'].shape[0], 1))])

         
        else:
            hidden_final_x = np.dot(act_layers[f'{i-1}']['bias'], weights[f'{i}'])
            #no tanh cos activation is probablity distribution
            final_y = sigmoid(hidden_final_x)
            neurons[f'{i}'] = final_y

    return neurons
