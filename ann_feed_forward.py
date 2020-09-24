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
def feed_forward(X, weights_input, weights_m, act):

    """Input: Observation data X and proceses one feed-forward loop
    Output: probablity distribution for y classes"""

    weighted_x = np.dot(X, weights_input)
    if act=='sigmoid':
        act_y = sigmoid(weighted_x)
    elif act=='tanh':
        act_y = tanh(weighted_x)
    act_y_bias = np.hstack([act_y, np.ones((act_y.shape[0], 1))])
    hidden_weighted_x = np.dot(act_y_bias, weights_m)
    #no tanh cos activation is probablity distribution
    final_y = sigmoid(hidden_weighted_x)
    return act_y, final_y


# feed forward the x values through the neurons
def CUST_feed_forward(X, weights_input, weights_m, act):

    no_layers = 2
    weights_matrices = [weights_input, weights_m]
    act_layers = {}

    for i in range(no_layers):
        if i == 0:
            act_layers[i] = {}
            input_weighted_x = np.dot(X, weights_matrices[i])
            if act=='sigmoid':
                act_layers[i]['normal'] = sigmoid(input_weighted_x)
            elif act=='tanh':
                act_layers[i]['normal']  = tanh(input_weighted_x)
            act_layers[i]['bias']  = np.hstack([act_layers[i]['normal'] , np.ones((act_layers[i]['normal'] .shape[0], 1))])
        # elif i < no_layers-1:
        #     act_layers[i] = {}
        #     hidden_weighted_x = np.dot(act_layers[i-1]['bias'], weights_matrices[i])
        #     if act=='sigmoid':
        #         act_layers[i]['normal'] = sigmoid(hidden_weighted_x)
        #     elif act=='tanh':
        #         act_layers[i]['normal'] = tanh(hidden_weighted_x)
        #     act_layers[i]['bias'] = np.hstack([act_layers[i]['normal'], np.ones((act_layers[i]['normal'].shape[0], 1))])
        else:
            hidden_weighted_x = np.dot(act_layers[i-1]['bias'], weights_matrices[1])
            #no tanh cos activation is probablity distribution
            final_y = sigmoid(hidden_weighted_x)
            return act_layers[i-1]['normal'], final_y

#x np.dot(input_s,output_s)

#y activation of x

#z np.hstack of y


initial_weights = np.random.randn(3, 2)
initial_m_weights = np.random.randn(3, 1)
