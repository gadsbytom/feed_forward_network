import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=50, noise=0.2, random_state=42)

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# tanh function
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


# feed forward the x values through the neurons
def feed_forward(X, weights_input, weights_m):
    """ Input: Observation data X and proceses one feed-forward loop
        Output: probablity distribution for y classes"""
    weighted_x = np.dot(X, weights_input)
    sigmoid_y = sigmoid(weighted_x)
    sigmoid_y_bias = np.hstack([sigmoid_y, np.ones((sigmoid_y.shape[0], 1))])
    hidden_weighted_x = np.dot(sigmoid_y_bias, weights_m)
    final_y = sigmoid(hidden_weighted_x)
    return sigmoid_y, final_y
