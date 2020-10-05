
#!/usr/bin/env python
# coding: utf8
from sklearn.datasets import make_moons
from training import epoch_training
import time
from matplotlib import pyplot as plt
import numpy as np


def generate_data():
    X, y = make_moons(n_samples=50, noise=0.2, random_state=42)
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    return X,y


if __name__ == "__main__":


    X, y = generate_data()

    print(X.shape, y.shape)
    weights = {}

    print('How many hidden layers would you like for the network?\n')
    hidden_layers = int(input())
    no_layers = hidden_layers + 2 #no_layers = intput + hidden_layers + output
    time.sleep(0.5)

    print(f"\nWhich activation function would you like to use in the Hidden Layers?\nChoose between 'sigmoid' & 'tanh'\n")
    activation = input()
    time.sleep(0.5)

    #input layer
    input_shape = X.shape[1]
    weights['0'] = {}

    for i in range(hidden_layers):
        #need to debug shape errors in backprop
        #print(f"\nHow many neurons would you like in the {i+1}'st hidden layer\n")
        #neurons = int(input())
        neurons = 2
        weights[f'{i}'] = np.random.randn(input_shape, neurons)
        input_shape = neurons +1
        weights[f'{i+1}'] = {}
        print(f'shape for layer {i} is:')
        print(weights[f'{i}'].shape)

    #output layer
    weights[f'{i+1}']= np.random.randn(input_shape, neurons)
    print(f'shape for layer {i+1} is:')
    print(weights[f'{i+1}'].shape)
    input_shape = neurons +1
    output_shape = 1 #need to customise this for new datasets
    weights[f'{i+2}'] = {}
    weights[f'{i+2}'] = np.random.randn(input_shape, output_shape)
    print(f'shape for layer {i+2} is:')
    print(weights[f'{i+2}'].shape)


    print(weights.items())

    epoch_logloss, num_epochs = epoch_training(
        X, y, 5000, weights, 0.01, activation
    )


    plt.figure(figsize=(10, 8))
    plt.plot(epoch_logloss)
    plt.legend(f'Log loss after {num_epochs} epochs of SGD ')
    plt.show()



