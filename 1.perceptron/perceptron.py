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

    def train(self, X, y, num_epochs):
        for i in range(num_epochs):
            epoch_loss = 0
            for x, y_true in zip(X,y):
                y_hat = self.predict(x)
                loss = -(y_true * np.log(y_hat) + (1-y_true)*np.log(1-y_hat)) # implement cross entropy loss fix later
                epoch_loss += loss
                grad_w = (y_hat-y_true)*x
                grad_b = y_hat - y_true

                self.weights = self.weights - self.learning_rate * grad_w
                self.bias = self.bias - self.learning_rate * grad_b

            num_correct = 0
            for x, y_true in zip(X,y):
                y_hat_at = self.predict(x)
                pred_class = 1 if y_hat_at>=0.5 else 0

                if pred_class==y_true: num_correct+=1

            accuracy = num_correct/len(X)

            print(f'Epoch {i} train accuracy: {accuracy}')
            print(f'Epoch {i} train loss: {epoch_loss/len(X)}')

        pass  # code for training loop
