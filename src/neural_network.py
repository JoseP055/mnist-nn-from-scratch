import numpy as np
import matplotlib.pyplot as plt
from proccesMNIST import get_images

# MNIST Path
mnist_path = './data/MNIST/'

x_train_num, y_train_num, x_test_num, y_test_num = get_images(mnist_path)

# Prepare x/y train, validation and test dataframes

# Train data: 50,000 numbers and labels
x_train = x_train_num[:50000].reshape(50000, -1).astype(np.float32) / 255
y_train = y_train_num[:50000].reshape(50000, 1)

# Validation data: 10,000 numbers and labels
x_val = x_train_num[50000:].reshape(10000, -1).astype(np.float32) / 255
y_val = y_train_num[50000:].reshape(10000, 1)

# Test data: 10,000 numbers and labels
x_test = x_test_num.copy().reshape(10000, -1).astype(np.float32) / 255
y_test = y_test_num.copy().reshape(10000, 1)

def create_minibatches(mb_size, x, y, shuffle=True):
    """
    x: # samples, 784
    y: # samples, 1
    """
    assert x.shape[0] == y.shape[0], 'ERROR in the quantity of samples'
    total_data = x.shape[0]
    if shuffle:
        idxs = np.arange(total_data)
        np.random.shuffle(idxs)
        x = x[idxs]
        y = y[idxs]
    
    return ((x[i:i+mb_size], y[i:i+mb_size]) for i in range(0, total_data, mb_size))

# INIT PARAMETERS

def init_parameters(input_size, neurons):
    """
    input_size = input elements, 784
    neurons = list [200, 10] neurons per layer
    """
    W1 = np.random.randn(neurons[0], input_size) * 0.001
    b1 = np.zeros((neurons[0], 1))

    W2 = np.random.randn(neurons[1], neurons[0]) * 0.001
    b2 = np.zeros((neurons[1], 1))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

parameters = init_parameters(28*28, [200, 10])

# print(parameters["W1"].shape)
# print(parameters["b1"].shape)
# print(parameters["W2"].shape)
# print(parameters["b2"].shape)

# FUNCTIONS

#RELU
def relu(x):
    return np.maximum(0, x)

#Scores
def scores(x, paremeters, activation_fnc):
    """
    x has (#pixels, num samples) shape
    """
    z1 = parameters["W1"] @ x + parameters["b1"]
    a1 = activation_fnc(z1)
    z2 = parameters["W2"] @ a1 + parameters["b2"]

    return z2, z1, a1

scores, z1, a1 = scores(x_train[:64].T, parameters, relu)
# print(x_train[:64].T.shape) # (784, 64)

# Softmax
def softmax(x):
    exp_scores= np.exp(x)
    sum_exp_scores = np.sum(exp_scores, axis=0)
    probs = exp_scores/sum_exp_scores

    return probs

# Loss Entropy

def x_entropy(scores, y, batch_size=64):
    probs = softmax(scores)
    y_hat = probs[y.squeeze(), np.arange(batch_size)]
    cost = np.sum(-np.log(y_hat)) / batch_size

    return probs, cost

# Back Propagation // Backward
def backward(probs, x, y, z1, a1, parameters, batch_size=64):
    grads = {}
    probs[y.squeeze(), np.arange(batch_size)] -= 1 # y_hat - y
    dz2 = probs.copy()

    dW2 = dz2 @ a1.T / batch_size
    db2 = np.sum(dz2, axis=1, keepdims=True) / batch_size
    da1 = parameters['W2'].T @ dz2

    dz1 = da1.copy()
    dz1[z1 <= 0] = 0

    dW1 = dz1 @ x 
    db1 = np.sum(dz1, axis=1, keepdims=True)

    assert parameters['W1'].shape == dW1.shape, "W1 not same shape"
    assert parameters['W2'].shape == dW2.shape, "W2 not same shape"
    assert parameters['b1'].shape == db1.shape, "b1 not same shape"
    assert parameters['b2'].shape == db2.shape, "b2 not same shape"

    grads = {'W1': dW1, "b1":db1, "W2": dW2, "b2": db2}
    
    return grads

y_hat, cost = x_entropy(scores, y_train[:64])
#print(y_hat.shape)
grads = backward(y_hat, x_train[:64], y_train[:64], z1, a1, parameters)
