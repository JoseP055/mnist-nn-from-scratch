import os
import numpy as np
import matplotlib.pyplot as plt

mnist_path = './data/MNIST/'

def list_files(mnist_path):
    return [os.path.join(mnist_path, f) for f in os.listdir(mnist_path) if os.path.isfile(os.path.join(mnist_path, f))]

def get_images(mnist_path):
    x_train = y_train = x_test = y_test = None

    for f in list_files(mnist_path):
        if 'train-images' in f:
            with open(f, 'rb') as data:
                _ = int.from_bytes(data.read(4), 'big')
                num_images = int.from_bytes(data.read(4), 'big')
                rows = int.from_bytes(data.read(4), 'big')
                cols = int.from_bytes(data.read(4), 'big')
                train_images = data.read()
                x_train = np.frombuffer(train_images, dtype=np.uint8).reshape((num_images, rows, cols))
        elif 'train-labels' in f:
            with open(f, 'rb') as data:
                data.read(8)
                y_train = np.frombuffer(data.read(), dtype=np.uint8)
        elif 't10k-images' in f:
            with open(f, 'rb') as data:
                _ = int.from_bytes(data.read(4), 'big')
                num_images = int.from_bytes(data.read(4), 'big')
                rows = int.from_bytes(data.read(4), 'big')
                cols = int.from_bytes(data.read(4), 'big')
                test_images = data.read()
                x_test = np.frombuffer(test_images, dtype=np.uint8).reshape((num_images, rows, cols))
        elif 't10k-labels' in f:
            with open(f, 'rb') as data:
                data.read(8)
                y_test = np.frombuffer(data.read(), dtype=np.uint8)

    return x_train, y_train, x_test, y_test


def show_number(x, y=None):
    plt.imshow(x, cmap='gray')
    if y is not None:
        plt.title(f'Digit: {y}')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = get_images(mnist_path)

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    for i in range(5):
        rand_idx = np.random.randint(len(y_test))
        show_number(x_test[rand_idx], y_test[rand_idx])
