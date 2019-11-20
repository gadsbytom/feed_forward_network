from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
import numpy as np
from math import e

X,y = make_moons(n_samples=50, noise=0.2, random_state=42)

def sigmoid(x):
    return 1 / (1 + e** -x)

def tanh(x):
    return (e**x - e**-x) / (e**x + e**-x)


def feed_forward_sigmoid(X, w0,):
    """ Input: Observation data X and proceses one feed-forward loop
        Output: probablity distribution for y classes"""
    #step 1: add a bias column, so X turns from 50,2 into 50,3
    X = np.hstack((X,np.ones((X.shape[0],1))))
    #step 2: parse the dimensions of the input (plus bias) data, and initialise the weight layers
    dimensions = []
    for i in range(len(X.shape)):
        dimensions.append(X.shape[i])
    weights0 = np.random.rand(dimensions[1],2)
    weights1 = np.random.rand(dimensions[1],1)
    #step 3: dot product of X with weights0
    d1 = np.dot(X,weights0)
    #step 4: apply sigmoid function to each value in d1
    s1 = sigmoid(d1)
    #step 5: add the bias column to the sigmoid layer
    h1 = np.hstack((s1,np.ones((s1.shape[0],1))))
    #step 6: dot product of h1 * weights1
    d2 = np.dot(h1, weights1)
    #step 7: pass the second layer through a sigmoid activation
    s2 = sigmoid(d2)
    #step 8: convert the probability distributions to firm results
    ypred = [1 if x>0.5 else 0 for x in s2]

    return s1, s2, ypred

if __name__ == '__main__':
    X,y = make_moons(n_samples=50, noise=0.2, random_state=42)
    _ , _, ypred = feed_forward_sigmoid(X)
    accuracy = accuracy_score(y,ypred)
    print(f'feed forward score is: {accuracy}')
