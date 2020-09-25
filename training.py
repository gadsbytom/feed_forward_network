import numpy as np
from ann_feed_forward import feed_forward_two_plus_hidden, sigmoid, tanh
from ann_backprop import CUST_backprop, log_loss



def epoch(X, y, num_epochs, input_weights, hidden_1_weights, hidden_2_weights, LR, lr_h, act):


    average_log_loss = []
    for i in range(num_epochs):

        output1_hidden, output2_hidden, ypred = feed_forward_two_plus_hidden(X, input_weights, hidden_1_weights, hidden_2_weights, act)

        input_weights, hidden_1_weights, hidden_2_weights = CUST_backprop(
            input_weights, hidden_1_weights, hidden_2_weights, output1_hidden, output2_hidden, ypred, y, X, LR, lr_h, act
        )

        average_log_loss.append(np.sum(log_loss(y, ypred)))
        # logging.debug(f'the shape of the log_loss is {log_loss(y,ypred).shape}')
    return average_log_loss, num_epochs, hidden_1_weights, hidden_2_weights

#
#
# def epoch(X, y, num_epochs, input_weights, hidden_1_weights, hidden_2_weights, LR, lr_h, act):
#
#
#     average_log_loss = []
#     for i in range(num_epochs):
#
#         output1_hidden, output2_hidden, ypred = feed_forward_two_plus_hidden(X, input_weights, hidden_1_weights, hidden_2_weights, act)
#
#         input_weights, hidden_1_weights, hidden_2_weights = CUST_backprop(
#             input_weights, hidden_1_weights, hidden_2_weights, output1_hidden, output2_hidden, ypred, y, X, LR, lr_h, act
#         )
#
#         average_log_loss.append(np.sum(log_loss(y, ypred)))
#         # logging.debug(f'the shape of the log_loss is {log_loss(y,ypred).shape}')
#     return average_log_loss, num_epochs, hidden_1_weights, hidden_2_weights
