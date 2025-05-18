import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron


X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

# Initialize perceptron
model = Perceptron(shape=(2,))

# Train model and capture history
loss_history, acc_history = model.train(X, y, num_epochs=10000)

# Plot Loss and Accuracy
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(loss_history, label='Loss')
plt.title('Training Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(acc_history, label='Accuracy')
plt.title('Training Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
