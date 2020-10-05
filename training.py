import numpy as np
from ann_feed_forward import feed_forward, sigmoid, tanh
from ann_backprop import backprop, log_loss


def epoch_training(X, y, num_epochs, weights, LR, act):


    average_log_loss = []
    for i in range(num_epochs):

        neurons = feed_forward(X, weights, act)

        weights = backprop(
            weights, neurons, y, X, LR, act
        )

        ypred = neurons[list(neurons.keys())[-1]]
        average_log_loss.append(np.sum(log_loss(y, ypred))) #neurons[-1] is ypred
        # logging.debug(f'the shape of the log_loss is {log_loss(y,ypred).shape}')
    return average_log_loss, num_epochs