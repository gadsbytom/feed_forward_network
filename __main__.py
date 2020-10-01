#!/usr/bin/env python
# coding: utf8
from sklearn.datasets import make_moons
from training import epoch_two
import time
from matplotlib import pyplot as plt
import numpy as np


if __name__ == "__main__":


    # architecture = {}
    # no_layers = 0
    # neurons = 0
    # activation = ''



    # print('How many hidden layers would you like for the network?\n')
    # no_layers = int(input())
    # time.sleep(0.5)

    # architecture['no_layers'] = no_layers

    # print(f"\nWhich activation function would you like to use in the Hidden Layers?\nChoose between 'sigmoid' & 'tanh'\n")
    # activation = input()
    # time.sleep(0.5)


    # architecture['activation'] = activation

    # for i in range(no_layers):
    #     print(f'\nHow many neurons would you like in layer #{i}\n')
    #     neurons = int(input())
    #     time.sleep(0.5)

    #     architecture[f'layer_{i}'] = {}
    #     architecture[f'layer_{i}']['neurons'] = neurons

    # print(architecture.items())


    # layers = {}

    # #shape of weight matrix
    # for i in range(no_layers):
    #     if i == 0:
    #         layers[f'layer_{i}'] = {}
    #         layers[f'layer_{i}']['input'] = X.shape[1]
    #         layers[f'layer_{i}']['output'] = architecture[f'layer_{i}']['neurons'] +1
    #     else:
    #         layers[f'layer_{i}'] = {}
    #         layers[f'layer_{i}']['input'] = architecture[f'layer_{i-1}']['neurons']
    #         layers[f'layer_{i}']['output'] = architecture[f'layer_{i}']['neurons'] +1

    # print(layers.items())


    def generate_data():
        X, y = make_moons(n_samples=50, noise=0.2, random_state=42)
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        return X,y

    X, y = generate_data()

    hidden_layers = 1

    weights = {}
    weights['input'] = np.random.randn(3, 2)
    weights['hidden_1'] = np.random.randn(3, 2)
    weights['output'] = np.random.randn(3, 1)

    print(weights.keys())

    epoch_logloss, num_epochs = epoch_two(
        X, y, 5000, weights, 0.01, 'sigmoid'
    )


    plt.figure(figsize=(10, 8))
    plt.plot(epoch_logloss)
    plt.legend(f'Log loss after {num_epochs} epochs of SGD ')
    plt.show()
