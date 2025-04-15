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
