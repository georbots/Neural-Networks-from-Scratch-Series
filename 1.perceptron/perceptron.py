import numpy as np

class Perceptron:
    def __init__(self, shape, learning_rate=0.001, activation='sigmoid'):
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=shape)
        self.bias = 0
        self.learning_rate = learning_rate
        self.activation = activation

    def predict(self, x, activation=None):
        if activation is None:
            activation = self.activation

        z = np.dot(self.weights, x) + self.bias

        if activation == 'sigmoid':
            y_hat = 1 / (1 + np.exp(-z))
        elif activation == 'relu':
            y_hat = np.maximum(0, z)
        elif activation == 'step':
            y_hat = 1 if z >= 0 else 0
        else:
            raise KeyError(f"Unknown activation: {activation}")
        return y_hat

    def train(self, X, y, num_epochs, callback=None):
        loss_history = []
        acc_history = []
        for i in range(num_epochs):
            epoch_loss = 0
            for x, y_true in zip(X, y):
                y_hat = self.predict(x, activation=self.activation)
                y_hat = np.clip(y_hat, 1e-8, 1 - 1e-8)  # to avoid nan loss with relu
                loss = -(y_true * np.log(y_hat + 1e-8) + (1 - y_true) * np.log(1 - y_hat + 1e-8))
                epoch_loss += loss

                grad_w = (y_hat - y_true) * x
                grad_b = y_hat - y_true

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            num_correct = sum(
                (1 if self.predict(x_i, activation=self.activation) >= 0.5 else 0) == y_i
                for x_i, y_i in zip(X, y)
            )
            accuracy = num_correct / len(X)
            loss_history.append(epoch_loss / len(X))
            acc_history.append(accuracy)

            if callback:
                callback(i, self.weights.copy(), self.bias)

            if i % 1000 == 0 or i == num_epochs - 1:
                print(f"Epoch {i}: acc = {accuracy:.4f}, loss = {epoch_loss / len(X):.4f}")

        return loss_history, acc_history