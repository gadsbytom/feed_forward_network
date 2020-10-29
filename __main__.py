
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
    weights = {}

    print('How many hidden layers would you like for the network?\n')
    hidden_layers = int(input())
    no_layers = hidden_layers + 2 #no_layers = intput + hidden_layers + output
    time.sleep(0.5)

    correct_function = False
    activations = ['sigmoid', 'tanh']
    while not correct_function:
        print(f"\nWhich activation function would you like to use in the Hidden Layers?\nChoose between 'sigmoid' & 'tanh'\n")
        activation = input()
        if activation in activations:
            correct_function = True
        else:
            print("Sorry, we dont know that one!")
    time.sleep(0.5)
    #input layer
    input_shape = X.shape[1]
    weights['0'] = {}

    for i in range(hidden_layers):
        #need to debug shape errors in backprop - for now leave fixed at 3 neurons per layer
        neurons = 3
        weights[f'{i}'] = np.random.randn(input_shape, neurons)
        input_shape = neurons +1
        weights[f'{i+1}'] = {}

    #output layer
    weights[f'{i+1}']= np.random.randn(input_shape, neurons)
    input_shape = neurons +1
    output_shape = 1
    weights[f'{i+2}'] = {}
    weights[f'{i+2}'] = np.random.randn(input_shape, output_shape)

    epoch_logloss, num_epochs = epoch_training(
        X, y, 5000, weights, 0.01, activation
    )

    plt.figure(figsize=(10, 8))
    plt.plot(epoch_logloss)
    plt.title(f'Log loss after {num_epochs} epochs of SGD ')
    plt.show()



