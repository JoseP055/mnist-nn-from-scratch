import numpy as np
import pickle
import matplotlib.pyplot as plt
from src.neural_network import scores, relu, load_data

_, _, _, _, x_test, y_test, x_test_num, y_test_num = load_data()

with open('mnist_model.pkl', 'rb') as f:
    parameters = pickle.load(f)

def predict(x_input):
    scores2, _, _ = scores(x_input, parameters, relu)
    return int(np.argmax(scores2, axis=0))

for i in range(5):
    idx = np.random.randint(len(x_test))
    image = x_test[idx]
    image_raw = x_test_num[idx]
    true_label = y_test[idx][0]
    predicted_label = predict(image.reshape(-1, 1))

    plt.figure(figsize=(4, 4))
    plt.imshow(image_raw, cmap='gray')
    plt.title(f"Predicted: {predicted_label} | True: {true_label}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


