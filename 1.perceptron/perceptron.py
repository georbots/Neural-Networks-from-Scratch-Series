import numpy as np

class Perceptron:
    def __init__(self, shape):
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=shape)
        self.bias = 0
        self.learning_rate = 0.001
        pass

    def predict(self, x):
        z = np.dot(self.weights,x)+self.bias # calculate neuron output

        y_hat = 1 / (1 + np.exp(-z)) # implement sigmoid function
        return y_hat

    def train(self, X, y, num_epochs, callback=None):
        loss_history = []
        acc_history = []
        for i in range(num_epochs):
            epoch_loss = 0
            for x, y_true in zip(X, y):
                y_hat = self.predict(x)
                loss = -(y_true * np.log(y_hat + 1e-8) + (1 - y_true) * np.log(1 - y_hat + 1e-8))
                epoch_loss += loss

                grad_w = (y_hat - y_true) * x
                grad_b = y_hat - y_true

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            num_correct = sum(
                (1 if self.predict(x_i) >= 0.5 else 0) == y_i
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