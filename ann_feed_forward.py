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
def feed_forward_one_hidden(X, weights_input, weights_m, act):

    """Input: Observation data X and proceses one feed-forward loop
    Output: probablity distribution for y classes"""

    weighted_x = np.dot(X, weights_input)
    if act=='sigmoid':
        act_x = sigmoid(weighted_x)
    elif act=='tanh':
        act_x = tanh(weighted_x)
    act_x_bias = np.hstack([act_x, np.ones((act_x.shape[0], 1))])
    hidden_weighted_x = np.dot(act_x_bias, weights_m)
    #no tanh cos activation is probablity distribution
    final_y = sigmoid(hidden_weighted_x)
    return act_x, final_y


# feed forward the x values through the neurons
def feed_forward_two_plus_hidden(X, weights, act):

    #dict_keys(['input', 'hidden_0', 'output'])

    no_layers = len(weights.keys())
    neurons = {}
    act_layers = {}


    for i in range(no_layers):
        if i == 0:
            act_layers['input'] = {}
            input_weighted_x = np.dot(X, weights['input'])
            if act=='sigmoid':
                act_layers['input']['normal'] = sigmoid(input_weighted_x)
            elif act=='tanh':
                act_layers['input']['normal']  = tanh(input_weighted_x)
            neurons['input'] = act_layers['input']['normal']
            act_layers['input']['bias']  = np.hstack([act_layers['input']['normal'] , np.ones((act_layers['input']['normal'] .shape[0], 1))])
            
        elif i ==1:
            act_layers[f"hidden_{i}"] = {}
            hidden_weighted_x = np.dot(act_layers['input']['bias'], weights[f'hidden_{i}'])
            if act=='sigmoid':
                act_layers[f"hidden_{i}"]['normal'] = sigmoid(hidden_weighted_x)
            elif act=='tanh':
                act_layers[f"hidden_{i}"]['normal'] = tanh(hidden_weighted_x)
            neurons[f'hidden_{i}'] = act_layers[f'hidden_{i}']['normal']
            act_layers[f"hidden_{i}"]['bias'] = np.hstack([act_layers[f"hidden_{i}"]['normal'], np.ones((act_layers[f"hidden_{i}"]['normal'].shape[0], 1))])
         
        elif i < no_layers-1:
            act_layers[f"hidden_{i}"] = {}
            hidden_weighted_x = np.dot(act_layers[f"hidden_{i-1}"]['bias'], weights[f'hidden_{i}'])
            if act=='sigmoid':
                act_layers[f"hidden_{i}"]['normal'] = sigmoid(hidden_weighted_x)
            elif act=='tanh':
                act_layers[f"hidden_{i}"]['normal'] = tanh(hidden_weighted_x)
            neurons[f'hidden_{i}'] = act_layers[f'hidden_{i}']['normal']
            act_layers[f"hidden_{i}"]['bias'] = np.hstack([act_layers[f"hidden_{i}"]['normal'], np.ones((act_layers[f"hidden_{i}"]['normal'].shape[0], 1))])
        
        else:
            hidden_final_x = np.dot(act_layers[f"hidden_{i-1}"]['bias'], weights['output'])
            #no tanh cos activation is probablity distribution
            final_y = sigmoid(hidden_final_x)
            neurons['output'] = final_y
    return neurons
