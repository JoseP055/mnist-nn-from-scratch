import numpy as np
import matplotlib.pyplot as plt
from src.neural_network import load_data, train, init_parameters

# For save a model 
import pickle

if __name__ == "__main__":

    x_train, y_train, x_val, y_val, x_test, y_test, _, _ = load_data()


    mb_size = 512
    learning_rate = 1e-2
    epochs = 30
    parameters = init_parameters(28*28, [200, 10])

    parameters, cost_history, accuracy_history = train(x_data=x_train, y_data=y_train, epochs=epochs, parameters=parameters, mb_size=mb_size, learning_rate=learning_rate)

    # Save the trained model to a file
    with open('mnist_model.pkl', 'wb') as f:
        pickle.dump(parameters, f)

    print("✅ Model successfully saved as 'mnist_model.pkl'")

    plt.figure(figsize=(12, 5))

    # Cost
    plt.subplot(1, 2, 1)
    plt.plot(cost_history, label='Cost', marker='o')
    plt.title('Cost over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history, label='Accuracy', color='green', marker='s')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()



