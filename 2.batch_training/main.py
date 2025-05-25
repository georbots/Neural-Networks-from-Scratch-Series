import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate 150 points around (1,1) for class 0
class0 = np.random.randn(150, 2) + np.array([1, 1])

# Generate 150 points around (4,4) for class 1
class1 = np.random.randn(150, 2) + np.array([4, 4])


X = np.vstack((class0, class1))
y = np.hstack((np.zeros(150), np.ones(150)))


plt.figure(figsize=(8,6))
plt.scatter(class0[:, 0], class0[:, 1], color='red', label='Class 0')
plt.scatter(class1[:, 0], class1[:, 1], color='blue', label='Class 1')
plt.title("Linearly Separable 2D Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
