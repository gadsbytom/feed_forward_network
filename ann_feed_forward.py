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
def feed_forward(X, weights_input, weights_m, act='sigmoid'):

    """Input: Observation data X and proceses one feed-forward loop
    Output: probablity distribution for y classes"""
    weighted_x = np.dot(X, weights_input)
    if act=='sigmoid':
        act_y = sigmoid(weighted_x)
    elif act=='tanh':
        act_y = tanh(weighted_x)
    act_y_bias = np.hstack([act_y, np.ones((act_y.shape[0], 1))])
    hidden_weighted_x = np.dot(act_y_bias, weights_m)
    if act=='sigmoid':
        final_y = sigmoid(hidden_weighted_x)
    elif act=='tanh':
        final_y = tanh(hidden_weighted_x)
    return act_y, final_y
