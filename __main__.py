#!/usr/bin/env python
# coding: utf8
import argparse
from sklearn.datasets import make_moons
from ann_feed_forward import feed_forward, sigmoid, tanh
from ann_backprop import backprop, epoch
from matplotlib import pyplot as plt
import numpy as np


if __name__ == "__main__":


    # Use argparse give command line-based documentation.
    parser = argparse.ArgumentParser(description="""Customise your own Neural Network,
                                    and try and solve the a binary classification problem.""")

    parser.add_argument('-a', '--activation',
                        type=str,
                        help='Choose activation: sigmoid or tanh.')

    args = parser.parse_args()

    X, y = make_moons(n_samples=50, noise=0.2, random_state=42)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    X.shape, y.shape

    # make the feed forward network
    X = np.hstack(
        [X, np.ones((X.shape[0], 1))]
    )  # adding an extra dimension for the bias

    initial_weights = np.random.randn(3, 2)
    initial_m_weights = np.random.randn(3, 1)

    epoch_200_logloss, _, _ = epoch(
        X, y, 5000, initial_weights, initial_m_weights, 0.01, 0.01, args.activation
    )

    plt.figure(figsize=(10, 10))
    plt.plot(epoch_200_logloss)
    plt.show()
