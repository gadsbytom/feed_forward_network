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
    input_weights, output_weights, output_hidden, ypred, ytrue, X_input, LR_O, LR_H, act
):
    # separate learning rates for outer and inner weights.

    wH = input_weights
    wO = output_weights


    ytrue = ytrue.reshape(-1, 1)
    error = (ypred - ytrue) * log_loss(ytrue, ypred)  # although order doesn't matter
    # logging.debug(f'shape of error is {error.shape}')

    # no option here as output layer is probability distribution
    act_deriv = sigmoid_prime(ypred)

    # derivative of the sigmoid function with respect to the hidden output *
    y_grad = act_deriv * error
    # logging.debug(f'shape of ygrad is {y_grad.shape}')


    hidden_out_with_bias = np.hstack(
        [output_hidden, np.ones((output_hidden.shape[0], 1))]
    )
    # logging.debug(f'shape of hidden_out_with_bias is {hidden_out_with_bias.shape}')
    # include bias
    delta_wo = -np.dot(y_grad.transpose(), hidden_out_with_bias) * LR_O
    # logging.debug(f'shape of delta_wo is {delta_wo.shape}')

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


def epoch(X, y, num_epochs, input_weights, output_weights, lr_o, lr_h, act):
    average_log_loss = []
    for i in range(num_epochs):
        hidden_output, ypred = feed_forward(X, input_weights, output_weights, act)
        input_weights, output_weights = backprop(
            input_weights, output_weights, hidden_output, ypred, y, X, lr_o, lr_h, act
        )
        average_log_loss.append(np.sum(log_loss(y, ypred)))
        # logging.debug(f'the shape of the log_loss is {log_loss(y,ypred).shape}')
    return average_log_loss, input_weights, output_weights
