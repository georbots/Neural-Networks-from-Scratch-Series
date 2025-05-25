import numpy as np

class Perceptron:
    def __init__(self, shape, learning_rate=0.001, activation='sigmoid'):
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=shape)
        self.bias = 0
        self.learning_rate = learning_rate
        self.activation = activation

    def predict(self, x_batch, activation=None):
        if activation is None:
            activation = self.activation

        z = x_batch.dot(self.weights) + self.bias

        if activation == 'sigmoid':
            y_hat = 1 / (1 + np.exp(-z))
        elif activation == 'relu':
            y_hat = np.maximum(0, z)
        elif activation == 'step':
            y_hat = np.where(z >= 0, 1, 0)
        else:
            raise KeyError(f"Unknown activation: {activation}")
        return y_hat

    def train(self, X_batches, y_batches, num_epochs, callback=None):
        loss_history = []
        acc_history = []

        for epoch in range(num_epochs):
            epoch_loss = 0

            for x_batch, y_batch in zip(X_batches, y_batches):
                y_hat = self.predict(x_batch, activation=self.activation)
                y_hat = np.clip(y_hat, 1e-8, 1 - 1e-8)  # to avoid nan loss with relu
                loss = -(y_batch * np.log(y_hat + 1e-8) + (1 - y_batch) * np.log(1 - y_hat + 1e-8))
                epoch_loss += np.mean(loss)

                grad_w = x_batch.T @ (y_hat - y_batch) / len(x_batch)
                grad_b = np.mean(y_hat - y_batch)

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            X_full = np.vstack(X_batches)
            y_full = np.hstack(y_batches)

            y_pred = self.predict(X_full, activation=self.activation)
            y_pred_labels = (y_pred >= 0.5).astype(int)
            num_correct = np.sum(y_pred_labels == y_full)

            accuracy = num_correct / len(y_full)
            loss_history.append(epoch_loss / len(X_batches))
            acc_history.append(accuracy)

            if callback:
                callback(epoch, self.weights.copy(), self.bias)

            print(f"Epoch {epoch}: acc = {accuracy:.4f}, loss = {epoch_loss / len(X_full):.4f}")

        return loss_history, acc_history