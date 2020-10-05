#!/usr/bin/env python
# coding: utf8

from ann_feed_forward import feed_forward, sigmoid, tanh
import logging
import math
import numpy as np
import pandas as pd


# define log loss
def log_loss(ytrue, ypred):
    ytrue = ytrue.reshape(-1, 1)
    ypred = ypred.reshape(-1, 1)
    loss = -(ytrue * np.log(ypred) + (1 - ytrue) * np.log(1 - ypred))
    return loss

# derivative of the sigmoid
def sigmoid_prime(x):
    derivative = sigmoid(x) * (1 - sigmoid(x))
    return derivative

# derivative of the tanh function
def tanh_prime(x):
    return 1-((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))**2


# backpropagation once through
def backprop(
    weights, neurons, ytrue, X_input, LR, act
):


    ytrue = ytrue.reshape(-1, 1)

    no_layers = len(weights)

    inverse_loop = list(range(no_layers))[::-1]

    gradients = {}

    for i in inverse_loop:

        if i == inverse_loop[0]:

            print(f'Backprop at layer: {i} has these shapes') 

            error = (neurons[f'{i}'] - ytrue) * log_loss(ytrue, neurons[f'{i}'])  
            act_deriv = sigmoid_prime(neurons[f'{i}'])

            print('act_deriv shape ' + str(act_deriv.shape))
            gradients[f'{i}'] = act_deriv * error

            hidden_2_with_bias = np.hstack(
                [neurons[f'{i-1}'], np.ones((neurons[f'{i-1}'].shape[0], 1))]
            )

            print('hidden_2_bias shape ' + str(hidden_2_with_bias.shape))

            delta_wH2 = -np.dot(gradients[f'{i}'].transpose(), hidden_2_with_bias) * LR

            print('delta wh2 shape ' + str(delta_wH2.shape))

  
            weights[f'{i}']= weights[f'{i}'] + delta_wH2.transpose()

            print('new weights shape ' + str(weights[f'{i}'].shape))

        
        elif i > 0:

            print(f'Backprop at layer: {i} has these shapes') 

            if act=='sigmoid':
                act_deriv_2 = sigmoid_prime(neurons[f'{i}'])
            elif act=='tanh':
                act_deriv_2 = tanh_prime(neurons[f'{i}'])

            print('act_deriv 2 shape ' + str(act_deriv_2.shape))

            gradients[f'{i}'] = act_deriv_2 * np.dot(
                gradients[f'{i+1}'], weights[f'{i+1}'][:2].transpose()
            )  
            hidden_1_with_bias = np.hstack(
                [neurons[f'{i}'], np.ones((neurons[f'{i}'].shape[0], 1))]
                )

            print('hidden_1_bias shape ' + str(hidden_1_with_bias.shape))

            delta_wH1 = -np.dot(gradients[f'{i}'].transpose(), hidden_1_with_bias) * LR

            print('delta wh1 shape ' + str(delta_wH1.shape))

            weights[f'{i}'] = weights[f'{i}'] + delta_wH1.transpose() 

            print('new weights shape ' + str(weights[f'{i}'].shape))

        else:
            print(f'Backprop at layer: {i} has these shapes') 
            if act=='sigmoid':
                act_deriv_1 = sigmoid_prime(neurons[f'{i}'])
            elif act=='tanh':
                act_deriv_1 = tanh_prime(neurons[f'{i}'])

            print('act_deriv 1 shape ' + str(act_deriv_1.shape))

            gradients[f'{i}'] = act_deriv_1 * np.dot(
                gradients[f'{i+1}'], weights[f'{i+1}'][:2].transpose()
            )  
            print('X_input' + str(X_input.shape))

            delta_wH = -np.dot(gradients[f'{i}'].transpose(), X_input) * LR

            print('delta wh shape ' + str(delta_wH.shape))

            weights[f'{i}'] = weights[f'{i}'] + delta_wH.transpose()  
            print('new weights shape ' + str(weights[f'{i}'].shape))

    return weights 