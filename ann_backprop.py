from ann_feed_forward import feed_forward, sigmoid, tanh
import logging
import math
import numpy as np
import pandas as pd


# logging.basicConfig(level=logging.DEBUG)
# logging.debug("test")


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


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# tanh function
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


# backpropagation once through
def backprop(
    input_weights, output_weights, output_hidden, ypred, ytrue, X_input, LR_O, LR_H
):
    # separate learning rates for outer and inner weights.

    wH = input_weights
    wO = output_weights

    # STEP A:
    ytrue = ytrue.reshape(-1, 1)
    error = (ypred - ytrue) * log_loss(ytrue, ypred)  # although order doesn't matter
    # logging.debug(f'shape of error is {error.shape}')

    # STEP B:
    sig_deriv = sigmoid_prime(ypred)
    # ogging.debug(f'shape of sig deriv is {sig_deriv.shape}')

    # derivative of the sigmoid function with respect to the hidden output *
    y_grad = sig_deriv * error
    # logging.debug(f'shape of ygrad is {y_grad.shape}')

    # STEP C:
    hidden_out_with_bias = np.hstack(
        [output_hidden, np.ones((output_hidden.shape[0], 1))]
    )
    # logging.debug(f'shape of hidden_out_with_bias is {hidden_out_with_bias.shape}')
    # don't forget the bias!
    delta_wo = -np.dot(y_grad.transpose(), hidden_out_with_bias) * LR_O
    # logging.debug(f'shape of delta_wo is {delta_wo.shape}')

    # and finally, old weights + delta weights -> new weights!
    wO_new = wO + delta_wo.transpose()

    # STEP D:
    sig_deriv_2 = sigmoid_prime(output_hidden)
    H_grad = sig_deriv_2 * np.dot(
        y_grad, wO_new[:2].transpose()
    )  # this is updating the hidden layer
    # logging.debug(f'shape of H_grad is {H_grad.shape}')

    # exclude the bias (3rd column) of the outer weights, since it is not backpropagated!

    # STEP E:
    delta_wH = -np.dot(H_grad.transpose(), X_input) * LR_H
    wH_new = wH + delta_wH.transpose()  # old weights + delta weights -> new weights!

    return wH_new, wO_new


def epoch(X, y, num_epochs, input_weights, output_weights, lr_o, lr_h):
    average_log_loss = []
    for i in range(num_epochs):
        hidden_output, ypred = feed_forward(X, input_weights, output_weights)
        input_weights, output_weights = backprop(
            input_weights, output_weights, hidden_output, ypred, y, X, lr_o, lr_h
        )
        average_log_loss.append(np.sum(log_loss(y, ypred)))
        # logging.debug(f'the shape of the log_loss is {log_loss(y,ypred).shape}')
    return average_log_loss, input_weights, output_weights
