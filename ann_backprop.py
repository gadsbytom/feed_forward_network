#!/usr/bin/env python
# coding: utf8

from ann_feed_forward import feed_forward_two_plus_hidden, sigmoid, tanh
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
def backprop_one_hidden(
    input_weights, output_weights, output_hidden, ypred, ytrue, X_input, LR_O, LR_H, act
):
    # separate learning rates for outer and inner weights.
    wH = input_weights
    wO = output_weights
    ytrue = ytrue.reshape(-1, 1)
    error = (ypred - ytrue) * log_loss(ytrue, ypred)  # although order doesn't matter
    # no option here as output layer is probability distribution
    act_deriv = sigmoid_prime(ypred)
    # derivative of the sigmoid function with respect to the hidden output *
    y_grad = act_deriv * error
    hidden_out_with_bias = np.hstack(
        [output_hidden, np.ones((output_hidden.shape[0], 1))]
    )
    # include bias
    delta_wo = -np.dot(y_grad.transpose(), hidden_out_with_bias) * LR_O
    # old weights + delta weights -> new weights!
    wO_new = wO + delta_wo.transpose()
    if act=='sigmoid':
        act_deriv_2 = sigmoid_prime(output_hidden)
    elif act=='tanh':
        act_deriv_2 = tanh_prime(output_hidden)
    H_grad = act_deriv_2 * np.dot(
        y_grad, wO_new[:2].transpose()
    )  # this is updating the hidden layer
    # exclude the bias (3rd column) of the outer weights, since it is not backpropagated!
    delta_wH = -np.dot(H_grad.transpose(), X_input) * LR_H
    wH_new = wH + delta_wH.transpose()  # old weights + delta weights -> new weights!

    return wH_new, wO_new



# backpropagation once through
def backprop_two_hidden(
    weights, neurons, ytrue, X_input, LR, act
):


    ytrue = ytrue.reshape(-1, 1)

    no_layers = len(weights)

    inverse_loop = list(range(no_layers))[::-1]

    for i in inverse_loop:

        if i == inverse_loop[0]:

            error = (neurons[f'{i}'] - ytrue) * log_loss(ytrue, neurons[f'{i}'])  # although order doesn't matter
            # no option here as output layer is probability distribution
            act_deriv = sigmoid_prime(neurons[f'{i}'])
            # derivative of the sigmoid function with respect to the hidden output *
            y_grad = act_deriv * error

            hidden_2_with_bias = np.hstack(
                [neurons[f'{i-1}'], np.ones((neurons[f'{i-1}'].shape[0], 1))]
            )# include bias

            delta_wH2 = -np.dot(y_grad.transpose(), hidden_2_with_bias) * LR

            # old weights + delta weights -> new weights!
            weights[f'{i}']= weights[f'{i}'] + delta_wH2.transpose()
        
        elif i > 0:

            if act=='sigmoid':
                act_deriv_2 = sigmoid_prime(neurons[f'{i}'])
            elif act=='tanh':
                act_deriv_2 = tanh_prime(neurons[f'{i}'])


            H2_grad = act_deriv_2 * np.dot(
                y_grad, weights[f'{i+1}'][:2].transpose()
            )  # this is updating the hidden layer
            # exclude the bias (3rd column) of the outer weights, since it is not backpropagated!

            hidden_1_with_bias = np.hstack(
                [neurons[f'{i}'], np.ones((neurons[f'{i}'].shape[0], 1))]
                )

            delta_wH1 = -np.dot(H2_grad.transpose(), hidden_1_with_bias) * LR
            weights[f'{i}'] = weights[f'{i}'] + delta_wH1.transpose()  # old weights + delta weights -> new weights!

        else:
            if act=='sigmoid':
                act_deriv_1 = sigmoid_prime(neurons[f'{i}'])
            elif act=='tanh':
                act_deriv_1 = tanh_prime(neurons[f'{i}'])


            H1_grad = act_deriv_1 * np.dot(
                H2_grad, weights[f'{i+1}'][:2].transpose()
            )  # this is updating the hidden layer
            # exclude the bias (3rd column) of the outer weights, since it is not backpropagated!


            delta_wH = -np.dot(H1_grad.transpose(), X_input) * LR
            weights[f'{i}'] = weights[f'{i}'] + delta_wH.transpose()  # old weights + delta weights -> new weights!

    
    return weights #input weight m, hidden 1 matrix, hidden 2 matrix respectively
