import numpy as np
from ann_feed_forward import feed_forward_two_plus_hidden, sigmoid, tanh
from ann_backprop import backprop_two_hidden, log_loss



def epoch_two(X, y, num_epochs, weights, LR, act):


    average_log_loss = []
    for i in range(num_epochs):

        neurons = feed_forward_two_plus_hidden(X, weights, act)

        weights = backprop_two_hidden(
            weights, neurons, y, X, LR, act
        )

        ypred = neurons[list(neurons.keys())[-1]]
        average_log_loss.append(np.sum(log_loss(y, ypred))) #neurons[-1] is ypred
        # logging.debug(f'the shape of the log_loss is {log_loss(y,ypred).shape}')
    return average_log_loss, num_epochs