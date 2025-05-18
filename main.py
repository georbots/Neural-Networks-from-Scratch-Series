import numpy as np
from perceptron import Perceptron

if __name__ == "__main__":
    # AND gate inputs and outputs
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([0, 0, 0, 1])

    perceptron = Perceptron(shape=(2,))
    perceptron.train(X, y, num_epochs=1000)

    print("\nTesting trained perceptron on AND gate:")
    for x_input, y_true in zip(X, y):
        y_pred = perceptron.predict(x_input)
        pred_class = 1 if y_pred >= 0.5 else 0
        print(f"Input: {x_input} Predicted: {pred_class} Actual: {y_true}")
